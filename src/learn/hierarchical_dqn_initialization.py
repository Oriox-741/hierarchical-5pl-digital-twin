"""Hierarchical DQN initialization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import torch
from torch import Tensor
import torch.nn.functional as F

from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.observation_builder import OBSERVATION_DIM
from src.think.joint_policies import (
    CHECKPOINT_VERSION,
    FLAT_DQN_ARCHITECTURE,
    HIERARCHICAL_DQN_ARCHITECTURE,
    JointPolicyBundle,
)


HIERARCHICAL_INIT_CONFIG_KEY: Final[str] = "hierarchical_initialization"
HIERARCHICAL_INIT_METHOD_KEY: Final[str] = "hierarchical_init_method"
HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION: Final[str] = "flat_teacher_distillation_v1"
HIERARCHICAL_INIT_MDP_CONTRACT_VERSION: Final[str] = "physical_reality_v5_route_candidate_visibility"


@dataclass(frozen=True, slots=True)
class LoadedFlatDQNTeacher:
    name: str
    checkpoint_path: Path
    policy: JointPolicyBundle


@dataclass(frozen=True, slots=True)
class HierarchicalDQNInitializationConfig:
    enabled: bool = False
    method: str = HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION
    teacher_checkpoint: Path | None = None
    observation_sample_count: int = 4096
    distillation_steps: int = 512
    batch_size: int = 128
    learning_rate: float = 1.0e-4
    max_loss: float = 1.0
    q_loss_weight: float = 1.0
    action_ce_weight: float = 1.0
    temperature: float = 1.0
    seed: int = 42
    transfer_dqn_trunk: bool = True

    def __post_init__(self) -> None:
        if self.method != HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION:
            raise ValueError(f"unsupported hierarchical initialization method: {self.method!r}.")
        if self.enabled and self.teacher_checkpoint is None:
            raise ValueError("enabled hierarchical initialization requires teacher_checkpoint.")
        if self.observation_sample_count <= 0:
            raise ValueError("hierarchical initialization observation_sample_count must be positive.")
        if self.distillation_steps < 0:
            raise ValueError("hierarchical initialization distillation_steps must not be negative.")
        if self.batch_size <= 0:
            raise ValueError("hierarchical initialization batch_size must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("hierarchical initialization learning_rate must be positive.")
        if self.max_loss <= 0.0:
            raise ValueError("hierarchical initialization max_loss must be positive.")
        if self.q_loss_weight < 0.0:
            raise ValueError("hierarchical initialization q_loss_weight must not be negative.")
        if self.action_ce_weight < 0.0:
            raise ValueError("hierarchical initialization action_ce_weight must not be negative.")
        if self.temperature <= 0.0:
            raise ValueError("hierarchical initialization temperature must be positive.")

    def with_teacher_checkpoint(self, checkpoint: Path) -> "HierarchicalDQNInitializationConfig":
        return replace(self, teacher_checkpoint=Path(checkpoint))


@dataclass(frozen=True, slots=True)
class HierarchicalDQNInitializationLossResult:
    loss: Tensor
    metrics: dict[str, Any]


def hierarchical_dqn_initialization_config_from_training_config(
    config: Mapping[str, Any],
) -> HierarchicalDQNInitializationConfig:
    section = config.get(HIERARCHICAL_INIT_CONFIG_KEY, {})
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError(f"{HIERARCHICAL_INIT_CONFIG_KEY} config section must be an object.")

    enabled = bool(section.get("enabled", False))
    explicit_enable = bool(section.get("explicit_enable", False))
    if enabled and not explicit_enable:
        raise ValueError("hierarchical_initialization.enabled requires explicit_enable=true.")

    checkpoint_value = section.get("teacher_checkpoint")
    teacher_checkpoint = Path(str(checkpoint_value)) if checkpoint_value not in (None, "") else None
    return HierarchicalDQNInitializationConfig(
        enabled=enabled,
        method=str(section.get("method", HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION)),
        teacher_checkpoint=teacher_checkpoint,
        observation_sample_count=int(section.get("observation_sample_count", 4096)),
        distillation_steps=int(section.get("distillation_steps", 512)),
        batch_size=int(section.get("batch_size", 128)),
        learning_rate=float(section.get("learning_rate", 1.0e-4)),
        max_loss=float(section.get("max_loss", 1.0)),
        q_loss_weight=float(section.get("q_loss_weight", 1.0)),
        action_ce_weight=float(section.get("action_ce_weight", 1.0)),
        temperature=float(section.get("temperature", 1.0)),
        seed=int(section.get("seed", 42)),
        transfer_dqn_trunk=bool(section.get("transfer_dqn_trunk", True)),
    )


def load_flat_dqn_teacher_checkpoint(
    path: Path,
    *,
    device: torch.device | str,
    teacher_name: str,
) -> LoadedFlatDQNTeacher:
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("flat teacher checkpoint must be a dictionary.")
    validate_flat_teacher_checkpoint_contract(checkpoint)
    policy = JointPolicyBundle(dqn_architecture=FLAT_DQN_ARCHITECTURE).to(device)
    policy.load_checkpoint_state(checkpoint)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return LoadedFlatDQNTeacher(name=teacher_name, checkpoint_path=checkpoint_path, policy=policy)


def validate_flat_teacher_checkpoint_contract(checkpoint: Mapping[str, Any]) -> None:
    config = checkpoint.get("config")
    contract = config.get("mdp_contract_version") if isinstance(config, Mapping) else None
    if contract != HIERARCHICAL_INIT_MDP_CONTRACT_VERSION:
        raise ValueError(
            "flat teacher checkpoint MDP contract mismatch: expected "
            f"{HIERARCHICAL_INIT_MDP_CONTRACT_VERSION!r}, got {contract!r}."
        )
    version = checkpoint.get("checkpoint_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported flat teacher checkpoint_version: expected {CHECKPOINT_VERSION!r}, got {version!r}.")
    if checkpoint.get("observation_dim") != OBSERVATION_DIM:
        raise ValueError(f"flat teacher checkpoint observation_dim mismatch: expected {OBSERVATION_DIM}.")
    if checkpoint.get("discrete_action_count") != DISCRETE_ACTION_COUNT:
        raise ValueError(f"flat teacher checkpoint discrete_action_count mismatch: expected {DISCRETE_ACTION_COUNT}.")
    architecture = str(checkpoint.get("dqn_architecture", FLAT_DQN_ARCHITECTURE))
    if architecture != FLAT_DQN_ARCHITECTURE:
        raise ValueError(f"flat teacher checkpoint must use {FLAT_DQN_ARCHITECTURE!r}, got {architecture!r}.")


def sample_hierarchical_init_observation_bank(
    *,
    sample_count: int,
    seed: int,
    device: torch.device | str,
) -> Tensor:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    observations = torch.rand((int(sample_count), OBSERVATION_DIM), generator=generator, dtype=torch.float32)
    if sample_count >= 1:
        observations[0].zero_()
    if sample_count >= 2:
        observations[1].fill_(1.0)
    if sample_count >= 3:
        observations[2].fill_(0.5)
    return observations.to(device=device)


def compute_flat_to_hierarchical_distillation_loss(
    *,
    student_policy: JointPolicyBundle,
    teacher_policy: JointPolicyBundle,
    observations: Tensor,
    config: HierarchicalDQNInitializationConfig,
) -> HierarchicalDQNInitializationLossResult:
    if student_policy.dqn_architecture != HIERARCHICAL_DQN_ARCHITECTURE:
        raise ValueError("student_policy must use hierarchical_v1 for flat-to-hierarchical distillation.")
    if teacher_policy.dqn_architecture != FLAT_DQN_ARCHITECTURE:
        raise ValueError("teacher_policy must use flat_v1 for flat-to-hierarchical distillation.")
    observations = _as_observation_batch(observations)
    student_q = student_policy.dqn_q_network(observations)
    with torch.no_grad():
        teacher_q = teacher_policy.dqn_q_network(observations)
    if student_q.shape != teacher_q.shape or tuple(student_q.shape)[1] != DISCRETE_ACTION_COUNT:
        raise ValueError("distillation Q tensors must both have shape (batch, 48).")
    if not torch.isfinite(student_q).all() or not torch.isfinite(teacher_q).all():
        raise ValueError("distillation Q tensors must be finite.")

    temperature = float(config.temperature)
    teacher_actions = teacher_q.argmax(dim=1)
    student_actions = student_q.argmax(dim=1)
    q_loss = F.smooth_l1_loss(student_q, teacher_q.detach())
    action_ce = F.cross_entropy(student_q / temperature, teacher_actions)
    raw_loss = (float(config.q_loss_weight) * q_loss) + (float(config.action_ce_weight) * action_ce)
    bounded_loss = _soft_bounded_loss(raw_loss, max_loss=float(config.max_loss))
    metrics = {
        "hierarchical_distillation_loss": float(bounded_loss.detach().cpu().item()),
        "hierarchical_distillation_raw_loss": float(raw_loss.detach().cpu().item()),
        "hierarchical_distillation_q_loss": float(q_loss.detach().cpu().item()),
        "hierarchical_distillation_action_ce": float(action_ce.detach().cpu().item()),
        "hierarchical_distillation_action_match_rate": float(
            (student_actions == teacher_actions).to(dtype=torch.float32).mean().detach().cpu().item()
        ),
        "hierarchical_distillation_teacher_entropy": float(
            torch.distributions.Categorical(logits=teacher_q.detach()).entropy().mean().detach().cpu().item()
        ),
    }
    return HierarchicalDQNInitializationLossResult(loss=bounded_loss, metrics=metrics)


def distill_hierarchical_dqn_from_flat_teacher(
    *,
    student_policy: JointPolicyBundle,
    teacher_policy: JointPolicyBundle,
    observations: Tensor,
    config: HierarchicalDQNInitializationConfig,
) -> dict[str, Any]:
    observations = _as_observation_batch(observations)
    optimizer = torch.optim.Adam(student_policy.dqn_q_network.parameters(), lr=float(config.learning_rate))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))
    last_result = compute_flat_to_hierarchical_distillation_loss(
        student_policy=student_policy,
        teacher_policy=teacher_policy,
        observations=observations,
        config=config,
    )
    for _step in range(int(config.distillation_steps)):
        indices = torch.randint(
            low=0,
            high=observations.shape[0],
            size=(int(config.batch_size),),
            generator=generator,
            dtype=torch.long,
        )
        batch = observations.index_select(0, indices.to(device=observations.device))
        result = compute_flat_to_hierarchical_distillation_loss(
            student_policy=student_policy,
            teacher_policy=teacher_policy,
            observations=batch,
            config=config,
        )
        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(student_policy.dqn_q_network.parameters(), max_norm=10.0, error_if_nonfinite=True)
        optimizer.step()
        last_result = result
    student_policy.hard_update_dqn_target()
    final_result = compute_flat_to_hierarchical_distillation_loss(
        student_policy=student_policy,
        teacher_policy=teacher_policy,
        observations=observations,
        config=config,
    )
    return {
        **final_result.metrics,
        "hierarchical_distillation_initial_loss": float(last_result.metrics["hierarchical_distillation_loss"]),
        "hierarchical_distillation_steps": int(config.distillation_steps),
        "hierarchical_distillation_observation_sample_count": int(observations.shape[0]),
        "hierarchical_distillation_batch_size": int(config.batch_size),
    }


def load_hierarchical_distillation_initial_checkpoint(
    path: Path,
    *,
    policies: JointPolicyBundle,
    state: Any,
    device: torch.device,
    init_config: HierarchicalDQNInitializationConfig,
) -> dict[str, Any]:
    if policies.dqn_architecture != HIERARCHICAL_DQN_ARCHITECTURE:
        raise ValueError("hierarchical distillation initialization requires a hierarchical_v1 student policy.")
    if not init_config.enabled:
        raise ValueError("hierarchical distillation initialization requires enabled config.")

    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("initial checkpoint must be a dictionary.")
    validate_flat_teacher_checkpoint_contract(checkpoint)
    teacher = load_flat_dqn_teacher_checkpoint(checkpoint_path, device=device, teacher_name="flat_teacher")
    _load_ppo_state_from_checkpoint(policies, checkpoint)
    trunk_copied = False
    trunk_tensor_count = 0
    if init_config.transfer_dqn_trunk:
        trunk_tensor_count = copy_compatible_flat_dqn_trunk_to_hierarchical(
            teacher_policy=teacher.policy,
            student_policy=policies,
        )
        trunk_copied = trunk_tensor_count > 0
    observations = sample_hierarchical_init_observation_bank(
        sample_count=init_config.observation_sample_count,
        seed=init_config.seed,
        device=device,
    )
    metrics = distill_hierarchical_dqn_from_flat_teacher(
        student_policy=policies,
        teacher_policy=teacher.policy,
        observations=observations,
        config=init_config,
    )
    state.global_step = 0
    state.episode_count = 0
    state.dqn_update_count = 0
    state.last_ppo_loss = None
    state.last_dqn_loss = None

    config = checkpoint.get("config")
    shared_cfg = config.get("shared_global_parameters", {}) if isinstance(config, Mapping) else {}
    source_env_id = str(shared_cfg.get("environment_id", "")) if isinstance(shared_cfg, Mapping) else ""
    source_contract = config.get("mdp_contract_version") if isinstance(config, Mapping) else None
    return {
        "initialized_from_checkpoint": str(checkpoint_path),
        "initialized_from_env_id": source_env_id,
        "initialized_from_global_step": int(checkpoint.get("global_step", 0)),
        "initialized_from_contract": source_contract,
        "fine_tune_parent_checkpoint": str(checkpoint_path),
        "resume_mode": "init_from_joint_checkpoint",
        "replay_restore_status": "fresh_replay_by_design",
        HIERARCHICAL_INIT_METHOD_KEY: HIERARCHICAL_INIT_METHOD_FLAT_TEACHER_DISTILLATION,
        "hierarchical_teacher_checkpoint": str(checkpoint_path),
        "hierarchical_init_source_dqn_architecture": FLAT_DQN_ARCHITECTURE,
        "hierarchical_init_target_dqn_architecture": HIERARCHICAL_DQN_ARCHITECTURE,
        "hierarchical_init_ppo_loaded_from_teacher": True,
        "hierarchical_init_dqn_trunk_loaded_from_teacher": trunk_copied,
        "hierarchical_init_dqn_trunk_tensor_count": trunk_tensor_count,
        **metrics,
    }


def copy_compatible_flat_dqn_trunk_to_hierarchical(
    *,
    teacher_policy: JointPolicyBundle,
    student_policy: JointPolicyBundle,
) -> int:
    if teacher_policy.dqn_architecture != FLAT_DQN_ARCHITECTURE:
        raise ValueError("teacher_policy must use flat_v1.")
    if student_policy.dqn_architecture != HIERARCHICAL_DQN_ARCHITECTURE:
        raise ValueError("student_policy must use hierarchical_v1.")
    source_state = teacher_policy.dqn_q_network.net.state_dict()
    target_state = student_policy.dqn_q_network.trunk.state_dict()
    copied = 0
    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if isinstance(source_tensor, Tensor) and tuple(source_tensor.shape) == tuple(target_tensor.shape):
            target_state[key] = source_tensor.detach().clone().to(dtype=target_tensor.dtype)
            copied += 1
    student_policy.dqn_q_network.trunk.load_state_dict(target_state)
    return copied


def _load_ppo_state_from_checkpoint(policy: JointPolicyBundle, checkpoint: Mapping[str, Any]) -> None:
    states = checkpoint.get("model_state_dicts")
    if not isinstance(states, Mapping):
        raise ValueError("checkpoint is missing model_state_dicts.")
    policy.ppo_feature_extractor.load_state_dict(states["ppo_feature_extractor"])
    policy.ppo_actor.load_state_dict(states["ppo_actor"])
    policy.ppo_critic.load_state_dict(states["ppo_critic"])


def _as_observation_batch(observations: Tensor) -> Tensor:
    observations = observations.to(dtype=torch.float32)
    if observations.ndim == 1:
        observations = observations.unsqueeze(0)
    if observations.ndim != 2 or observations.shape[1] != OBSERVATION_DIM:
        raise ValueError(f"expected observations with shape (batch, {OBSERVATION_DIM}).")
    if observations.shape[0] <= 0:
        raise ValueError("observation batch must not be empty.")
    if not torch.isfinite(observations).all():
        raise ValueError("distillation observations must be finite.")
    return observations


def _soft_bounded_loss(raw_loss: Tensor, *, max_loss: float) -> Tensor:
    if max_loss <= 0.0:
        raise ValueError("max_loss must be positive.")
    if not torch.isfinite(raw_loss).all():
        raise ValueError("raw distillation loss must be finite.")
    max_loss_tensor = torch.as_tensor(float(max_loss), dtype=raw_loss.dtype, device=raw_loss.device)
    return max_loss_tensor * (raw_loss / (raw_loss + max_loss_tensor))
