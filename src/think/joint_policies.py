"""PyTorch policy modules for synchronized PPO/DQN joint MARL training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
from typing import Any, Final

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.discrete_action_mapper import (
    DISCRETE_ACTION_COUNT,
    DiscreteActionMapper,
    DispatchDecision,
    ModeDecision,
    ReorderDecision,
    RouteDecision,
)
from src.act.observation_builder import OBSERVATION_DIM


LOG_STD_MIN: Final[float] = -5.0
LOG_STD_MAX: Final[float] = 2.0
TANH_EPSILON: Final[float] = 1e-6
CHECKPOINT_VERSION: Final[str] = "torch_joint_policy_v1"
FLAT_DQN_ARCHITECTURE: Final[str] = "flat_v1"
HIERARCHICAL_DQN_ARCHITECTURE: Final[str] = "hierarchical_v1"
SUPPORTED_DQN_ARCHITECTURES: Final[frozenset[str]] = frozenset(
    {FLAT_DQN_ARCHITECTURE, HIERARCHICAL_DQN_ARCHITECTURE}
)


@dataclass(frozen=True, slots=True)
class PPOActionOutput:
    """Continuous PPO action sample and associated training quantities."""

    action: Tensor
    log_prob: Tensor
    value: Tensor
    entropy: Tensor
    mean: Tensor
    log_std: Tensor


@dataclass(frozen=True, slots=True)
class DQNActionOutput:
    """Discrete DQN action selection and lightweight exploration metadata."""

    action: Tensor
    q_values: Tensor
    greedy_action: Tensor
    exploratory: Tensor
    epsilon: float


class JointFeatureExtractor(nn.Module):
    """Shared-style MLP feature extractor for normalized 5PL observations."""

    def __init__(
        self,
        *,
        input_dim: int = OBSERVATION_DIM,
        hidden_dim: int = 256,
        hidden_layers: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not hidden_layers:
            raise ValueError("hidden_layers must not be empty.")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for layer_dim in hidden_layers:
            if layer_dim <= 0:
                raise ValueError("hidden layer dimensions must be positive.")
            layers.extend(
                (
                    nn.Linear(previous_dim, layer_dim),
                    nn.LayerNorm(layer_dim),
                    nn.SiLU(),
                )
            )
            previous_dim = layer_dim
        layers.append(nn.Linear(previous_dim, hidden_dim))
        layers.append(nn.SiLU())

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, observation: Tensor) -> Tensor:
        observation = _as_float_batch(observation, expected_dim=self.input_dim)
        return self.net(observation)


class ContinuousActor(nn.Module):
    """Tanh-squashed Gaussian actor for PPO strategic continuous controls."""

    def __init__(
        self,
        *,
        latent_dim: int = 256,
        action_dim: int = CONTINUOUS_ACTION_DIM,
        hidden_layers: tuple[int, ...] = (256,),
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.trunk = _mlp(latent_dim, hidden_layers, hidden_layers[-1] if hidden_layers else latent_dim)
        output_dim = hidden_layers[-1] if hidden_layers else latent_dim
        self.mean_head = nn.Linear(output_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5, dtype=torch.float32))

    def forward(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        latent = _as_float_batch(latent, expected_dim=self.latent_dim)
        mean = self.mean_head(self.trunk(latent))
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        return mean, log_std

    def distribution(self, latent: Tensor) -> Normal:
        mean, log_std = self(latent)
        return Normal(mean, log_std.exp())

    def sample(self, latent: Tensor, *, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        mean, log_std = self(latent)
        normal = Normal(mean, log_std.exp())
        pre_tanh = mean if deterministic else normal.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = _squashed_log_prob(normal, pre_tanh, action)
        entropy = normal.entropy().sum(dim=-1)
        return action, log_prob, entropy, mean, log_std

    def evaluate_actions(self, latent: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        latent = _as_float_batch(latent, expected_dim=self.latent_dim)
        actions = _as_float_batch(actions, expected_dim=self.action_dim).clamp(
            min=-1.0 + TANH_EPSILON,
            max=1.0 - TANH_EPSILON,
        )
        if actions.shape[0] != latent.shape[0]:
            raise ValueError(f"action batch size {actions.shape[0]} does not match latent batch size {latent.shape[0]}.")
        mean, log_std = self(latent)
        normal = Normal(mean, log_std.exp())
        pre_tanh = torch.atanh(actions)
        log_prob = _squashed_log_prob(normal, pre_tanh, actions)
        entropy = normal.entropy().sum(dim=-1)
        return log_prob, entropy


class ValueCritic(nn.Module):
    """Scalar state-value critic used by the PPO strategic controller."""

    def __init__(
        self,
        *,
        latent_dim: int = 256,
        hidden_layers: tuple[int, ...] = (256,),
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        output_dim = hidden_layers[-1] if hidden_layers else latent_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            _mlp(latent_dim, hidden_layers, output_dim),
            nn.Linear(output_dim, 1),
        )

    def forward(self, latent: Tensor) -> Tensor:
        latent = _as_float_batch(latent, expected_dim=self.latent_dim)
        return self.net(latent).squeeze(-1)


class DiscreteQNetwork(nn.Module):
    """Q-network for the DQN tactical dispatch controller."""

    def __init__(
        self,
        *,
        observation_dim: int = OBSERVATION_DIM,
        action_count: int = DISCRETE_ACTION_COUNT,
        hidden_layers: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        if action_count <= 0:
            raise ValueError("action_count must be positive.")

        output_dim = hidden_layers[-1] if hidden_layers else observation_dim
        self.observation_dim = observation_dim
        self.action_count = action_count
        self.dqn_architecture = FLAT_DQN_ARCHITECTURE
        self.net = nn.Sequential(
            _mlp(observation_dim, hidden_layers, output_dim),
            nn.Linear(output_dim, action_count),
        )

    def forward(self, observation: Tensor) -> Tensor:
        observation = _as_float_batch(observation, expected_dim=self.observation_dim)
        return self.net(observation)

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "external_q": self.action_count,
            "composition": "flat_external_q",
        }


class HierarchicalDQNQNetwork(nn.Module):
    """Factorized DQN that composes internal action heads into external 0..47 Q values."""

    def __init__(
        self,
        *,
        observation_dim: int = OBSERVATION_DIM,
        action_count: int = DISCRETE_ACTION_COUNT,
        hidden_layers: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        if action_count != DISCRETE_ACTION_COUNT:
            raise ValueError(f"hierarchical_v1 requires external action_count {DISCRETE_ACTION_COUNT}.")

        output_dim = hidden_layers[-1] if hidden_layers else observation_dim
        self.observation_dim = observation_dim
        self.action_count = action_count
        self.dqn_architecture = HIERARCHICAL_DQN_ARCHITECTURE
        self.trunk = _mlp(observation_dim, hidden_layers, output_dim)
        self.dispatch_head = nn.Linear(output_dim, len(DispatchDecision))
        self.route_head = nn.Linear(output_dim, len(RouteDecision))
        self.mode_head = nn.Linear(output_dim, len(ModeDecision))
        self.reorder_head = nn.Linear(output_dim, len(ReorderDecision))

        components = _external_action_component_indices()
        self.register_buffer("_dispatch_indices", components["dispatch"], persistent=False)
        self.register_buffer("_route_indices", components["route"], persistent=False)
        self.register_buffer("_mode_indices", components["mode"], persistent=False)
        self.register_buffer("_reorder_indices", components["reorder"], persistent=False)
        self.register_buffer("_dispatch_mask", components["dispatch_mask"], persistent=False)

    def forward(self, observation: Tensor) -> Tensor:
        latent = self._latent(observation)
        dispatch_q = self.dispatch_head(latent)
        route_q = self.route_head(latent)
        mode_q = self.mode_head(latent)
        reorder_q = self.reorder_head(latent)
        dispatch_component = dispatch_q[:, self._dispatch_indices]
        reorder_component = reorder_q[:, self._reorder_indices]
        route_mode_component = route_q[:, self._route_indices] + mode_q[:, self._mode_indices]
        return dispatch_component + reorder_component + (self._dispatch_mask.unsqueeze(0) * route_mode_component)

    def dispatch_q_values(self, observation: Tensor) -> Tensor:
        return self.dispatch_head(self._latent(observation))

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "dispatch": len(DispatchDecision),
            "route": len(RouteDecision),
            "mode": len(ModeDecision),
            "reorder": len(ReorderDecision),
            "composition": "dispatch + reorder + dispatch_mask(route + mode)",
        }

    def _latent(self, observation: Tensor) -> Tensor:
        observation = _as_float_batch(observation, expected_dim=self.observation_dim)
        return self.trunk(observation)


def build_dqn_q_network(
    *,
    observation_dim: int = OBSERVATION_DIM,
    action_count: int = DISCRETE_ACTION_COUNT,
    dqn_architecture: str = FLAT_DQN_ARCHITECTURE,
) -> nn.Module:
    if dqn_architecture == FLAT_DQN_ARCHITECTURE:
        return DiscreteQNetwork(observation_dim=observation_dim, action_count=action_count)
    if dqn_architecture == HIERARCHICAL_DQN_ARCHITECTURE:
        return HierarchicalDQNQNetwork(observation_dim=observation_dim, action_count=action_count)
    raise ValueError(f"unsupported dqn_architecture: {dqn_architecture!r}.")


class JointPolicyBundle(nn.Module):
    """Container for the synchronized PPO actor/critic and DQN online/target nets."""

    def __init__(
        self,
        *,
        observation_dim: int = OBSERVATION_DIM,
        continuous_action_dim: int = CONTINUOUS_ACTION_DIM,
        discrete_action_count: int = DISCRETE_ACTION_COUNT,
        hidden_dim: int = 256,
        dqn_architecture: str = FLAT_DQN_ARCHITECTURE,
    ) -> None:
        super().__init__()
        if dqn_architecture not in SUPPORTED_DQN_ARCHITECTURES:
            raise ValueError(f"unsupported dqn_architecture: {dqn_architecture!r}.")
        self.observation_dim = observation_dim
        self.continuous_action_dim = continuous_action_dim
        self.discrete_action_count = discrete_action_count
        self.hidden_dim = hidden_dim
        self.dqn_architecture = dqn_architecture

        self.ppo_feature_extractor = JointFeatureExtractor(
            input_dim=observation_dim,
            hidden_dim=hidden_dim,
        )
        self.ppo_actor = ContinuousActor(latent_dim=hidden_dim, action_dim=continuous_action_dim)
        self.ppo_critic = ValueCritic(latent_dim=hidden_dim)
        self.dqn_q_network = build_dqn_q_network(
            observation_dim=observation_dim,
            action_count=discrete_action_count,
            dqn_architecture=dqn_architecture,
        )
        self.dqn_target_q_network = build_dqn_q_network(
            observation_dim=observation_dim,
            action_count=discrete_action_count,
            dqn_architecture=dqn_architecture,
        )
        self.hard_update_dqn_target()

    def select_ppo_action(self, observation: Tensor, *, deterministic: bool = False) -> PPOActionOutput:
        latent = self.ppo_feature_extractor(observation)
        action, log_prob, entropy, mean, log_std = self.ppo_actor.sample(latent, deterministic=deterministic)
        value = self.ppo_critic(latent)
        return PPOActionOutput(
            action=action,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            mean=mean,
            log_std=log_std,
        )

    def evaluate_ppo_actions(self, observation: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        latent = self.ppo_feature_extractor(observation)
        log_prob, entropy = self.ppo_actor.evaluate_actions(latent, actions)
        value = self.ppo_critic(latent)
        return log_prob, entropy, value

    @torch.no_grad()
    def select_dqn_action(self, observation: Tensor, *, epsilon: float = 0.0) -> DQNActionOutput:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be within [0, 1].")

        q_values = self.dqn_q_network(observation)
        greedy_action = q_values.argmax(dim=-1)
        batch_size = greedy_action.shape[0]
        device = q_values.device
        random_action = torch.randint(
            low=0,
            high=self.discrete_action_count,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )
        exploratory = torch.rand(batch_size, device=device) < epsilon
        action = torch.where(exploratory, random_action, greedy_action)
        return DQNActionOutput(
            action=action,
            q_values=q_values,
            greedy_action=greedy_action,
            exploratory=exploratory,
            epsilon=float(epsilon),
        )

    @torch.no_grad()
    def hard_update_dqn_target(self) -> None:
        self.dqn_target_q_network.load_state_dict(self.dqn_q_network.state_dict())

    @torch.no_grad()
    def soft_update_dqn_target(self, tau: float) -> None:
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be within [0, 1].")
        for target_param, online_param in zip(
            self.dqn_target_q_network.parameters(),
            self.dqn_q_network.parameters(),
            strict=True,
        ):
            target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)

    def build_checkpoint(
        self,
        *,
        global_step: int,
        config: dict[str, Any],
        ppo_optimizer_state: dict[str, Any] | None = None,
        dqn_optimizer_state: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if global_step < 0:
            raise ValueError("global_step must not be negative.")
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "global_step": int(global_step),
            "config": config,
            "observation_dim": self.observation_dim,
            "continuous_action_dim": self.continuous_action_dim,
            "discrete_action_count": self.discrete_action_count,
            "dqn_architecture": self.dqn_architecture,
            "external_discrete_action_count": self.discrete_action_count,
            "internal_heads": self.dqn_q_network.architecture_metadata(),
            "model_state_dicts": {
                "ppo_feature_extractor": _snapshot_state_dict(self.ppo_feature_extractor),
                "ppo_actor": _snapshot_state_dict(self.ppo_actor),
                "ppo_critic": _snapshot_state_dict(self.ppo_critic),
                "dqn_q_network": _snapshot_state_dict(self.dqn_q_network),
                "dqn_target_q_network": _snapshot_state_dict(self.dqn_target_q_network),
            },
            "optimizer_state_dicts": {
                "ppo": _snapshot_payload(ppo_optimizer_state),
                "dqn": _snapshot_payload(dqn_optimizer_state),
            },
            "extra": extra or {},
        }

    def load_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        version = checkpoint.get("checkpoint_version")
        if version != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported checkpoint_version: expected {CHECKPOINT_VERSION!r}, got {version!r}.")
        expected_dimensions = {
            "observation_dim": self.observation_dim,
            "continuous_action_dim": self.continuous_action_dim,
            "discrete_action_count": self.discrete_action_count,
        }
        for key, expected_value in expected_dimensions.items():
            actual_value = checkpoint.get(key)
            if actual_value != expected_value:
                raise ValueError(f"checkpoint {key} mismatch: expected {expected_value}, got {actual_value}.")
        checkpoint_architecture = str(checkpoint.get("dqn_architecture", FLAT_DQN_ARCHITECTURE))
        if checkpoint_architecture != self.dqn_architecture:
            raise ValueError(
                "checkpoint dqn_architecture mismatch: "
                f"expected {self.dqn_architecture!r}, got {checkpoint_architecture!r}."
            )
        external_action_count = checkpoint.get("external_discrete_action_count")
        if external_action_count is not None and int(external_action_count) != self.discrete_action_count:
            raise ValueError(
                "checkpoint external_discrete_action_count mismatch: "
                f"expected {self.discrete_action_count}, got {external_action_count}."
            )

        states = checkpoint.get("model_state_dicts")
        if not isinstance(states, dict):
            raise ValueError("checkpoint is missing model_state_dicts.")

        self.ppo_feature_extractor.load_state_dict(states["ppo_feature_extractor"])
        self.ppo_actor.load_state_dict(states["ppo_actor"])
        self.ppo_critic.load_state_dict(states["ppo_critic"])
        self.dqn_q_network.load_state_dict(states["dqn_q_network"])
        self.dqn_target_q_network.load_state_dict(states["dqn_target_q_network"])


def _mlp(input_dim: int, hidden_layers: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for layer_dim in hidden_layers:
        if layer_dim <= 0:
            raise ValueError("hidden layer dimensions must be positive.")
        layers.extend((nn.Linear(previous_dim, layer_dim), nn.SiLU()))
        previous_dim = layer_dim
    if previous_dim != output_dim:
        layers.append(nn.Linear(previous_dim, output_dim))
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def _external_action_component_indices() -> dict[str, Tensor]:
    mapper = DiscreteActionMapper()
    dispatch_indices: list[int] = []
    route_indices: list[int] = []
    mode_indices: list[int] = []
    reorder_indices: list[int] = []
    dispatch_mask: list[float] = []
    for action_id in range(DISCRETE_ACTION_COUNT):
        action = mapper.map(action_id)
        dispatch_indices.append(int(action.dispatch))
        route_indices.append(int(action.route))
        mode_indices.append(int(action.mode))
        reorder_indices.append(int(action.reorder))
        dispatch_mask.append(1.0 if action.dispatch == DispatchDecision.DISPATCH else 0.0)
    return {
        "dispatch": torch.tensor(dispatch_indices, dtype=torch.long),
        "route": torch.tensor(route_indices, dtype=torch.long),
        "mode": torch.tensor(mode_indices, dtype=torch.long),
        "reorder": torch.tensor(reorder_indices, dtype=torch.long),
        "dispatch_mask": torch.tensor(dispatch_mask, dtype=torch.float32),
    }


def _as_float_batch(tensor: Tensor, *, expected_dim: int) -> Tensor:
    tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"expected a 1D or 2D tensor, got {tensor.ndim} dimensions.")
    if tensor.shape[-1] != expected_dim:
        raise ValueError(f"expected final dimension {expected_dim}, got {tensor.shape[-1]}.")
    return tensor


def _squashed_log_prob(normal: Normal, pre_tanh: Tensor, action: Tensor) -> Tensor:
    del action
    correction = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
    return (normal.log_prob(pre_tanh) - correction).sum(dim=-1)


def _snapshot_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _snapshot_payload(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_payload(item) for item in value)
    return value
