"""Runtime loader for synchronized Torch PPO+DQN joint checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.observation_builder import OBSERVATION_DIM
from src.think.joint_policies import CHECKPOINT_VERSION, FLAT_DQN_ARCHITECTURE, JointPolicyBundle


JOINT_ARTIFACT_KINDS = frozenset({"joint", "joint_final"})
EXPECTED_MDP_CONTRACT: Final[str] = "physical_reality_v5_route_candidate_visibility"


@dataclass(frozen=True, slots=True)
class TorchJointAction:
    continuous: np.ndarray
    discrete: int


@dataclass(slots=True)
class TorchJointRuntimePolicy:
    checkpoint_path: Path
    checkpoint: Mapping[str, Any]
    policies: JointPolicyBundle
    device: torch.device

    def predict_joint(self, observation: Any, *, deterministic: bool = True) -> TorchJointAction:
        observation_array = np.asarray(observation, dtype=np.float32).reshape(-1)
        if observation_array.shape != (OBSERVATION_DIM,):
            raise ValueError(f"observation shape mismatch: expected ({OBSERVATION_DIM},), got {observation_array.shape}.")

        observation_tensor = torch.as_tensor(
            observation_array,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            ppo_output = self.policies.select_ppo_action(observation_tensor, deterministic=deterministic)
            dqn_output = self.policies.select_dqn_action(observation_tensor, epsilon=0.0)

        continuous = ppo_output.action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if continuous.shape != (CONTINUOUS_ACTION_DIM,):
            raise ValueError(
                f"continuous action shape mismatch: expected ({CONTINUOUS_ACTION_DIM},), got {continuous.shape}."
            )
        if not np.all(np.isfinite(continuous)):
            raise ValueError("continuous action contains NaN or Inf values.")
        if np.any(continuous < -1.0) or np.any(continuous > 1.0):
            raise ValueError("continuous action must be bounded within [-1, 1].")

        discrete = int(dqn_output.action.squeeze(0).detach().cpu().item())
        if not 0 <= discrete < DISCRETE_ACTION_COUNT:
            raise ValueError(f"discrete action must be in [0, {DISCRETE_ACTION_COUNT - 1}], got {discrete}.")

        return TorchJointAction(continuous=continuous, discrete=discrete)


def load_torch_joint_policy(path: Path | str, *, device: torch.device | str = "cpu") -> TorchJointRuntimePolicy:
    checkpoint_path = Path(path)
    torch_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping.")

    _validate_torch_joint_checkpoint(checkpoint)
    policies = JointPolicyBundle(
        observation_dim=int(checkpoint["observation_dim"]),
        continuous_action_dim=int(checkpoint["continuous_action_dim"]),
        discrete_action_count=int(checkpoint["discrete_action_count"]),
        dqn_architecture=str(checkpoint.get("dqn_architecture", FLAT_DQN_ARCHITECTURE)),
    ).to(torch_device)
    policies.load_checkpoint_state(dict(checkpoint))
    policies.eval()

    return TorchJointRuntimePolicy(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        policies=policies,
        device=torch_device,
    )


def _validate_torch_joint_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    checkpoint_version = checkpoint.get("checkpoint_version")
    if checkpoint_version != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint_version mismatch: expected {CHECKPOINT_VERSION!r}, got {checkpoint_version!r}."
        )

    artifact_kind = checkpoint.get("artifact_kind")
    if artifact_kind not in JOINT_ARTIFACT_KINDS:
        raise ValueError(f"artifact_kind mismatch: expected one of {sorted(JOINT_ARTIFACT_KINDS)}, got {artifact_kind!r}.")

    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("config metadata must be a mapping.")

    contract = config.get("mdp_contract_version")
    if contract != EXPECTED_MDP_CONTRACT:
        raise ValueError(f"contract mismatch: expected {EXPECTED_MDP_CONTRACT!r}, got {contract!r}.")

    _require_int_field(checkpoint, "observation_dim", OBSERVATION_DIM)
    _require_int_field(checkpoint, "continuous_action_dim", CONTINUOUS_ACTION_DIM)
    _require_int_field(checkpoint, "discrete_action_count", DISCRETE_ACTION_COUNT)

    shared = config.get("shared_global_parameters", {})
    if isinstance(shared, Mapping) and "observation_dim" in shared:
        configured_dim = int(shared["observation_dim"])
        if configured_dim != OBSERVATION_DIM:
            raise ValueError(
                f"config shared_global_parameters.observation_dim mismatch: "
                f"expected {OBSERVATION_DIM}, got {configured_dim}."
            )


def _require_int_field(checkpoint: Mapping[str, Any], field: str, expected: int) -> None:
    if field not in checkpoint:
        raise ValueError(f"{field} metadata is required.")
    actual = int(checkpoint[field])
    if actual != expected:
        raise ValueError(f"{field} mismatch: expected {expected}, got {actual}.")
