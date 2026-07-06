"""Policy loading and inference service for active PPO/DQN checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.env_5pl import ActionMode
from src.act.observation_builder import OBSERVATION_DIM
from src.learn.model_registry import Algorithm, ModelRecord, ModelRegistry
from src.orchestration.torch_joint_runtime import (
    EXPECTED_MDP_CONTRACT,
    TorchJointRuntimePolicy,
    load_torch_joint_policy,
)
from src.think.evaluate_policy import HeuristicPolicy, PredictableModel


PPO_AGENT_ROLE = "continuous_control"
DQN_AGENT_ROLE = "tactical_dispatch"


@dataclass(frozen=True, slots=True)
class PolicyInference:
    action: np.ndarray | int | dict[str, Any]
    algorithm: Algorithm | str
    model_path: str | dict[str, str | None] | None
    deterministic: bool
    agent_id: str | None = None
    agent_role: str | None = None


@dataclass(frozen=True, slots=True)
class _TorchJointActivePair:
    joint_checkpoint: Path
    ppo_path: Path
    dqn_path: Path
    logical_model_id: str


@dataclass(frozen=True, slots=True)
class _ActiveRegistryEntry:
    path: Path
    record: ModelRecord | None


class PolicyService:
    """Load active registry models and run non-blocking inference from orchestration."""

    def __init__(self, registry: ModelRegistry | None = None, *, device: torch.device | str = "cpu") -> None:
        self.registry = registry or ModelRegistry()
        self.device = torch.device(device)
        self._loaded: dict[tuple[Algorithm, str | None], tuple[PredictableModel, Path]] = {}
        self._loaded_joint: dict[tuple[Path, str], TorchJointRuntimePolicy] = {}

    def load_active(
        self,
        algorithm: Algorithm,
        *,
        agent_role: str | None = None,
    ) -> tuple[PredictableModel, Path]:
        path = self.registry.active_path_for(algorithm, agent_role=agent_role)
        metadata = self._registry_metadata_for_path(algorithm, path, agent_role=agent_role)
        _ensure_sb3_compatible_artifact(path, metadata)
        cache_key = (algorithm, agent_role)
        cached = self._loaded.get(cache_key)
        if cached is not None and cached[1] == path:
            return cached

        model: BaseAlgorithm
        if algorithm == "ppo":
            model = PPO.load(path)
        elif algorithm == "dqn":
            model = DQN.load(path)
        else:
            raise ValueError(f"unsupported algorithm: {algorithm}")

        loaded = (cast(PredictableModel, model), path)
        self._loaded[cache_key] = loaded
        return loaded

    def load_active_joint(self) -> tuple[TorchJointRuntimePolicy, _TorchJointActivePair] | None:
        pair = self._resolve_active_torch_joint_pair()
        if pair is None:
            return None
        cache_key = (pair.joint_checkpoint.resolve(strict=False), str(self.device))
        policy = self._loaded_joint.get(cache_key)
        if policy is None:
            policy = load_torch_joint_policy(pair.joint_checkpoint, device=self.device)
            self._loaded_joint[cache_key] = policy
        return policy, pair

    def _registry_metadata_for_path(
        self,
        algorithm: Algorithm,
        path: Path,
        *,
        agent_role: str | None = None,
    ) -> dict[str, Any]:
        for record in self.registry.records(algorithm=algorithm, agent_role=agent_role):
            if _same_path(record.path, path):
                return dict(record.metadata)
        if agent_role is not None:
            for record in self.registry.records(algorithm=algorithm):
                if _same_path(record.path, path):
                    return dict(record.metadata)
        return {}

    def _active_entry_for_path(
        self,
        algorithm: Algorithm,
        path: Path,
        *,
        agent_role: str,
    ) -> _ActiveRegistryEntry:
        for record in self.registry.records(algorithm=algorithm, agent_role=agent_role, status="active"):
            if _same_path(record.path, path):
                return _ActiveRegistryEntry(path=path, record=record)
        for record in self.registry.records(algorithm=algorithm, status="active"):
            if _same_path(record.path, path):
                return _ActiveRegistryEntry(path=path, record=record)
        return _ActiveRegistryEntry(path=path, record=None)

    def _resolve_active_torch_joint_pair(self) -> _TorchJointActivePair | None:
        ppo_path = self._explicit_active_path_or_none("ppo", agent_role=PPO_AGENT_ROLE)
        dqn_path = self._explicit_active_path_or_none("dqn", agent_role=DQN_AGENT_ROLE)
        if ppo_path is None and dqn_path is None:
            return None
        if ppo_path is None or dqn_path is None:
            present_path = ppo_path if ppo_path is not None else dqn_path
            algorithm: Algorithm = "ppo" if ppo_path is not None else "dqn"
            agent_role = PPO_AGENT_ROLE if ppo_path is not None else DQN_AGENT_ROLE
            present_entry = self._active_entry_for_path(algorithm, present_path, agent_role=agent_role)
            if not _active_entry_looks_like_torch_joint(present_entry):
                return None
            missing = "ppo" if ppo_path is None else "dqn"
            raise FileNotFoundError(f"incomplete torch_joint active pair: missing explicit active {missing} model.")

        ppo_entry = self._active_entry_for_path("ppo", ppo_path, agent_role=PPO_AGENT_ROLE)
        dqn_entry = self._active_entry_for_path("dqn", dqn_path, agent_role=DQN_AGENT_ROLE)
        if not _looks_like_torch_joint_active_pair(ppo_entry, dqn_entry):
            return None
        if ppo_entry.record is None:
            raise ValueError(f"active torch_joint ppo registry record not found for role {PPO_AGENT_ROLE}: {ppo_path}")
        if dqn_entry.record is None:
            raise ValueError(f"active torch_joint dqn registry record not found for role {DQN_AGENT_ROLE}: {dqn_path}")
        return _validate_torch_joint_active_pair(
            ppo_entry.record,
            dqn_entry.record,
            ppo_path=ppo_path,
            dqn_path=dqn_path,
        )

    def _explicit_active_path_or_none(self, algorithm: Algorithm, *, agent_role: str) -> Path | None:
        try:
            return self.registry.active_path_for(algorithm, agent_role=agent_role)
        except FileNotFoundError as exc:
            if str(exc).startswith(f"No explicit active {algorithm} model configured"):
                return None
            raise

    def predict(
        self,
        observation: np.ndarray,
        *,
        algorithm: Algorithm,
        deterministic: bool = True,
        fallback_to_heuristic: bool = True,
        agent_id: str | None = None,
        agent_role: str | None = None,
    ) -> PolicyInference:
        try:
            model, path = self.load_active(algorithm, agent_role=agent_role)
            action_array, _ = model.predict(observation, deterministic=deterministic)
            return PolicyInference(
                action=_coerce_action(action_array, algorithm),
                algorithm=algorithm,
                model_path=str(path),
                deterministic=deterministic,
                agent_id=agent_id,
                agent_role=agent_role,
            )
        except FileNotFoundError:
            if not fallback_to_heuristic:
                raise
            mode: ActionMode = "continuous" if algorithm == "ppo" else "discrete"
            heuristic = HeuristicPolicy(mode)
            action_array, _ = heuristic.predict(observation, deterministic=True)
            return PolicyInference(
                action=_coerce_action(action_array, algorithm),
                algorithm=f"heuristic_{algorithm}",
                model_path=None,
                deterministic=True,
                agent_id=agent_id,
                agent_role=agent_role,
            )

    def predict_joint(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
        fallback_to_heuristic: bool = True,
        agent_id: str | None = "joint_controller",
        agent_role: str | None = "control_tower",
    ) -> PolicyInference:
        loaded_joint = self.load_active_joint()
        if loaded_joint is not None:
            policy, pair = loaded_joint
            action = policy.predict_joint(observation, deterministic=deterministic)
            return PolicyInference(
                action={
                    "continuous": np.asarray(action.continuous, dtype=np.float32).reshape(-1),
                    "discrete": int(action.discrete),
                },
                algorithm="torch_joint",
                model_path={
                    "joint": str(pair.joint_checkpoint),
                    "ppo": str(pair.ppo_path),
                    "dqn": str(pair.dqn_path),
                },
                deterministic=deterministic,
                agent_id=agent_id,
                agent_role=agent_role,
            )

        ppo_inference = self.predict(
            observation,
            algorithm="ppo",
            deterministic=deterministic,
            fallback_to_heuristic=fallback_to_heuristic,
            agent_id="ppo_strategic_controller",
            agent_role="continuous_control",
        )
        dqn_inference = self.predict(
            observation,
            algorithm="dqn",
            deterministic=deterministic,
            fallback_to_heuristic=fallback_to_heuristic,
            agent_id="dqn_tactical_controller",
            agent_role="tactical_dispatch",
        )
        return PolicyInference(
            action={
                "continuous": np.asarray(ppo_inference.action, dtype=np.float32).reshape(-1),
                "discrete": int(np.asarray(dqn_inference.action).reshape(-1)[0]),
            },
            algorithm="joint",
            model_path={
                "ppo": ppo_inference.model_path if isinstance(ppo_inference.model_path, str) else None,
                "dqn": dqn_inference.model_path if isinstance(dqn_inference.model_path, str) else None,
            },
            deterministic=deterministic,
            agent_id=agent_id,
            agent_role=agent_role,
        )


def _coerce_action(action_array: np.ndarray, algorithm: Algorithm) -> np.ndarray | int:
    if algorithm == "dqn":
        return int(np.asarray(action_array).item())
    return np.asarray(action_array, dtype=np.float32)


def _ensure_sb3_compatible_artifact(path: Path, metadata: dict[str, Any]) -> None:
    runtime_loader = str(metadata.get("runtime_loader", "")).lower()
    framework = str(metadata.get("framework", "")).lower()
    artifact_kind = str(metadata.get("artifact_kind", "")).lower()
    suffix = path.suffix.lower()

    if runtime_loader and runtime_loader != "sb3":
        raise ValueError(
            f"PolicyService SB3 loader cannot load runtime_loader={runtime_loader!r} artifact: {path}"
        )
    if framework and framework != "sb3":
        raise ValueError(f"PolicyService SB3 loader cannot load framework={framework!r} artifact: {path}")
    if artifact_kind.startswith("joint") or "torch" in artifact_kind:
        raise ValueError(
            f"PolicyService SB3 loader cannot load artifact_kind={artifact_kind!r} artifact: {path}"
        )
    if suffix != ".zip":
        raise ValueError(f"PolicyService SB3 loader can only load .zip artifacts, got {suffix!r}: {path}")


def _active_entry_looks_like_torch_joint(entry: _ActiveRegistryEntry) -> bool:
    metadata = entry.record.metadata if entry.record is not None else {}
    return (
        Path(entry.path).suffix.lower() == ".pt"
        or metadata.get("joint_checkpoint") is not None
        or str(metadata.get("framework", "")).lower() == "torch_joint"
        or str(metadata.get("runtime_loader", "")).lower() == "joint_torch"
    )


def _looks_like_torch_joint_active_pair(ppo_entry: _ActiveRegistryEntry, dqn_entry: _ActiveRegistryEntry) -> bool:
    entries = (ppo_entry, dqn_entry)
    return any(
        _active_entry_looks_like_torch_joint(entry)
        for entry in entries
    )


def _validate_torch_joint_active_pair(
    ppo_record: ModelRecord,
    dqn_record: ModelRecord,
    *,
    ppo_path: Path,
    dqn_path: Path,
) -> _TorchJointActivePair:
    ppo_metadata = ppo_record.metadata
    dqn_metadata = dqn_record.metadata
    _require_metadata_value(ppo_metadata, "agent_role", PPO_AGENT_ROLE, role="ppo")
    _require_metadata_value(dqn_metadata, "agent_role", DQN_AGENT_ROLE, role="dqn")
    _require_metadata_value(ppo_metadata, "framework", "torch_joint", role="ppo")
    _require_metadata_value(dqn_metadata, "framework", "torch_joint", role="dqn")
    _require_metadata_value(ppo_metadata, "runtime_loader", "joint_torch", role="ppo")
    _require_metadata_value(dqn_metadata, "runtime_loader", "joint_torch", role="dqn")
    _require_metadata_value(ppo_metadata, "artifact_kind", "ppo_final", role="ppo")
    _require_metadata_value(dqn_metadata, "artifact_kind", "dqn_final", role="dqn")
    _require_metadata_value(ppo_metadata, "contract", EXPECTED_MDP_CONTRACT, role="ppo")
    _require_metadata_value(dqn_metadata, "contract", EXPECTED_MDP_CONTRACT, role="dqn")
    _require_metadata_int(ppo_metadata, "obs_dim", OBSERVATION_DIM, role="ppo")
    _require_metadata_int(dqn_metadata, "obs_dim", OBSERVATION_DIM, role="dqn")
    _require_metadata_int(ppo_metadata, "action_count", DISCRETE_ACTION_COUNT, role="ppo")
    _require_metadata_int(dqn_metadata, "action_count", DISCRETE_ACTION_COUNT, role="dqn")

    ppo_logical_id = _required_metadata_string(ppo_metadata, "logical_model_id", role="ppo")
    dqn_logical_id = _required_metadata_string(dqn_metadata, "logical_model_id", role="dqn")
    if ppo_logical_id != dqn_logical_id:
        raise ValueError(
            f"torch_joint active pair logical_model_id mismatch: ppo={ppo_logical_id!r}, dqn={dqn_logical_id!r}."
        )

    ppo_joint_checkpoint = Path(_required_metadata_string(ppo_metadata, "joint_checkpoint", role="ppo"))
    dqn_joint_checkpoint = Path(_required_metadata_string(dqn_metadata, "joint_checkpoint", role="dqn"))
    if not _same_path(ppo_joint_checkpoint, dqn_joint_checkpoint):
        raise ValueError(
            "torch_joint active pair joint_checkpoint mismatch: "
            f"ppo={ppo_joint_checkpoint}, dqn={dqn_joint_checkpoint}."
        )
    if not ppo_joint_checkpoint.exists():
        raise FileNotFoundError(f"torch_joint active pair joint_checkpoint does not exist: {ppo_joint_checkpoint}")

    return _TorchJointActivePair(
        joint_checkpoint=ppo_joint_checkpoint,
        ppo_path=ppo_path,
        dqn_path=dqn_path,
        logical_model_id=ppo_logical_id,
    )


def _required_metadata_string(metadata: dict[str, Any], key: str, *, role: str) -> str:
    value = metadata.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"{role} torch_joint metadata field {key!r} is required.")
    return str(value)


def _require_metadata_value(metadata: dict[str, Any], key: str, expected: str, *, role: str) -> None:
    value = _required_metadata_string(metadata, key, role=role)
    if value.lower() != expected.lower():
        raise ValueError(f"{role} torch_joint metadata {key} mismatch: expected {expected!r}, got {value!r}.")


def _require_metadata_int(metadata: dict[str, Any], key: str, expected: int, *, role: str) -> None:
    if key not in metadata:
        raise ValueError(f"{role} torch_joint metadata field {key!r} is required.")
    value = int(metadata[key])
    if value != expected:
        raise ValueError(f"{role} torch_joint metadata {key} mismatch: expected {expected}, got {value}.")


def _same_path(left: Path | str, right: Path | str) -> bool:
    return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)
