from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.observation_builder import OBSERVATION_DIM
from src.learn.model_registry import ModelRegistry
from src.orchestration.policy_service import PolicyService
from src.orchestration.torch_joint_runtime import EXPECTED_MDP_CONTRACT


PPO_ROLE = "continuous_control"
DQN_ROLE = "tactical_dispatch"


class _FakeTorchAction:
    def __init__(self, *, continuous: np.ndarray | None = None, discrete: int = 7) -> None:
        self.continuous = (
            np.zeros(CONTINUOUS_ACTION_DIM, dtype=np.float32)
            if continuous is None
            else np.asarray(continuous, dtype=np.float32)
        )
        self.discrete = discrete


class _FakeTorchPolicy:
    def __init__(self, action: _FakeTorchAction | None = None) -> None:
        self.action = action or _FakeTorchAction()
        self.calls = 0

    def predict_joint(self, observation: np.ndarray, *, deterministic: bool = True) -> _FakeTorchAction:
        self.calls += 1
        return self.action


class _FakeSb3Model:
    def __init__(self, action: np.ndarray | int) -> None:
        self.action = action

    def predict(self, observation: np.ndarray, *, deterministic: bool = True):
        return self.action, None


class PolicyServiceTorchJointTests(unittest.TestCase):
    def test_candidate_records_do_not_activate_joint_runtime(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = ModelRegistry(root / "registry")
            paths = _touch_paths(root)
            metadata = _torch_joint_metadata(root)

            registry.register(
                algorithm="ppo",
                path=paths["ppo"],
                agent_role=PPO_ROLE,
                status="candidate",
                metadata={**metadata, "artifact_kind": "ppo_final"},
            )
            registry.register(
                algorithm="dqn",
                path=paths["dqn"],
                agent_role=DQN_ROLE,
                status="candidate",
                metadata={**metadata, "artifact_kind": "dqn_final"},
            )
            service = PolicyService(registry=registry)

            with patch("src.orchestration.policy_service.load_torch_joint_policy") as loader:
                inference = service.predict_joint(
                    np.zeros(OBSERVATION_DIM, dtype=np.float32),
                    fallback_to_heuristic=True,
                )

            loader.assert_not_called()
            self.assertEqual("joint", inference.algorithm)
            self.assertEqual({"ppo": None, "dqn": None}, inference.model_path)

    def test_missing_active_entries_fallback_when_allowed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            service = PolicyService(registry=ModelRegistry(Path(temp_dir) / "registry"))

            with patch("src.orchestration.policy_service.load_torch_joint_policy") as loader:
                inference = service.predict_joint(
                    np.zeros(OBSERVATION_DIM, dtype=np.float32),
                    fallback_to_heuristic=True,
                )

            loader.assert_not_called()
            self.assertEqual("joint", inference.algorithm)
            self.assertEqual({"ppo": None, "dqn": None}, inference.model_path)

            with self.assertRaisesRegex(FileNotFoundError, "No explicit active"):
                service.predict_joint(
                    np.zeros(OBSERVATION_DIM, dtype=np.float32),
                    fallback_to_heuristic=False,
                )

    def test_active_torch_joint_pair_uses_joint_loader_not_sb3(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = ModelRegistry(root / "registry")
            paths = _register_active_torch_pair(root, registry)
            fake_policy = _FakeTorchPolicy(_FakeTorchAction(discrete=11))
            service = PolicyService(registry=registry)

            with patch("src.orchestration.policy_service.load_torch_joint_policy", return_value=fake_policy) as loader:
                with patch("src.orchestration.policy_service.PPO.load") as ppo_load:
                    with patch("src.orchestration.policy_service.DQN.load") as dqn_load:
                        inference = service.predict_joint(
                            np.zeros(OBSERVATION_DIM, dtype=np.float32),
                            deterministic=True,
                            fallback_to_heuristic=False,
                        )

            loader.assert_called_once_with(paths["joint"], device=torch.device("cpu"))
            ppo_load.assert_not_called()
            dqn_load.assert_not_called()
            self.assertEqual("torch_joint", inference.algorithm)
            self.assertEqual({"joint": str(paths["joint"]), "ppo": str(paths["ppo"]), "dqn": str(paths["dqn"])}, inference.model_path)
            self.assertEqual((CONTINUOUS_ACTION_DIM,), np.asarray(inference.action["continuous"]).shape)
            self.assertEqual(11, inference.action["discrete"])

    def test_active_hierarchical_torch_joint_metadata_uses_joint_loader_not_sb3(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = ModelRegistry(root / "registry")
            hierarchical_metadata = {
                "logical_model_id": "joint_torch_v5_prod_hierarchical_v1_1m_20260611",
                "dqn_architecture": "hierarchical_v1",
                "hierarchical_init_method": "flat_teacher_distillation_v1",
                "external_discrete_action_count": DISCRETE_ACTION_COUNT,
                "training_step": 1_000_000,
                "exact_resume_capable": True,
                "long_run_gate_verdict": "PASS",
                "registry_scope": "candidate_only",
            }
            paths = _register_active_torch_pair(
                root,
                registry,
                ppo_metadata=hierarchical_metadata,
                dqn_metadata=hierarchical_metadata,
            )
            fake_policy = _FakeTorchPolicy(_FakeTorchAction(discrete=1))
            service = PolicyService(registry=registry)

            with patch("src.orchestration.policy_service.load_torch_joint_policy", return_value=fake_policy) as loader:
                with patch("src.orchestration.policy_service.PPO.load") as ppo_load:
                    with patch("src.orchestration.policy_service.DQN.load") as dqn_load:
                        inference = service.predict_joint(
                            np.zeros(OBSERVATION_DIM, dtype=np.float32),
                            deterministic=True,
                            fallback_to_heuristic=False,
                        )

            loader.assert_called_once_with(paths["joint"], device=torch.device("cpu"))
            ppo_load.assert_not_called()
            dqn_load.assert_not_called()
            self.assertEqual("torch_joint", inference.algorithm)
            self.assertEqual(1, inference.action["discrete"])

    def test_active_torch_joint_pair_is_cached(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = ModelRegistry(root / "registry")
            paths = _register_active_torch_pair(root, registry)
            fake_policy = _FakeTorchPolicy()
            service = PolicyService(registry=registry, device=torch.device("cpu"))

            with patch("src.orchestration.policy_service.load_torch_joint_policy", return_value=fake_policy) as loader:
                service.predict_joint(np.zeros(OBSERVATION_DIM, dtype=np.float32), fallback_to_heuristic=False)
                service.predict_joint(np.zeros(OBSERVATION_DIM, dtype=np.float32), fallback_to_heuristic=False)

            loader.assert_called_once_with(paths["joint"], device=torch.device("cpu"))
            self.assertEqual(2, fake_policy.calls)

    def test_mismatched_logical_model_id_raises(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = ModelRegistry(root / "registry")
            _register_active_torch_pair(
                root,
                registry,
                ppo_metadata={"logical_model_id": "ppo_model"},
                dqn_metadata={"logical_model_id": "dqn_model"},
            )
            service = PolicyService(registry=registry)

            with patch("src.orchestration.policy_service.load_torch_joint_policy") as loader:
                with self.assertRaisesRegex(ValueError, "logical_model_id"):
                    service.predict_joint(np.zeros(OBSERVATION_DIM, dtype=np.float32), fallback_to_heuristic=False)

            loader.assert_not_called()

    def test_mismatched_joint_checkpoint_raises(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            other_joint = root / "other_joint.pt"
            other_joint.write_text("placeholder", encoding="utf-8")
            registry = ModelRegistry(root / "registry")
            _register_active_torch_pair(root, registry, dqn_metadata={"joint_checkpoint": str(other_joint)})
            service = PolicyService(registry=registry)

            with patch("src.orchestration.policy_service.load_torch_joint_policy") as loader:
                with self.assertRaisesRegex(ValueError, "joint_checkpoint"):
                    service.predict_joint(np.zeros(OBSERVATION_DIM, dtype=np.float32), fallback_to_heuristic=False)

            loader.assert_not_called()

    def test_missing_or_wrong_runtime_loader_raises_without_sb3(self) -> None:
        for bad_metadata in ({"runtime_loader": "sb3"}, {"framework": "sb3"}):
            with self.subTest(bad_metadata=bad_metadata):
                with TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    registry = ModelRegistry(root / "registry")
                    _register_active_torch_pair(root, registry, ppo_metadata=bad_metadata)
                    service = PolicyService(registry=registry)

                    with patch("src.orchestration.policy_service.load_torch_joint_policy") as loader:
                        with patch("src.orchestration.policy_service.PPO.load") as ppo_load:
                            with patch("src.orchestration.policy_service.DQN.load") as dqn_load:
                                with self.assertRaisesRegex(ValueError, "runtime_loader|framework"):
                                    service.predict_joint(
                                        np.zeros(OBSERVATION_DIM, dtype=np.float32),
                                        fallback_to_heuristic=False,
                                    )

                    loader.assert_not_called()
                    ppo_load.assert_not_called()
                    dqn_load.assert_not_called()

    def test_wrong_artifact_kind_raises(self) -> None:
        for bad_metadata in ({"artifact_kind": "dqn_final"}, {"artifact_kind": "ppo_final"}):
            with self.subTest(bad_metadata=bad_metadata):
                with TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    registry = ModelRegistry(root / "registry")
                    if bad_metadata["artifact_kind"] == "dqn_final":
                        _register_active_torch_pair(root, registry, ppo_metadata=bad_metadata)
                    else:
                        _register_active_torch_pair(root, registry, dqn_metadata=bad_metadata)
                    service = PolicyService(registry=registry)

                    with patch("src.orchestration.policy_service.load_torch_joint_policy") as loader:
                        with self.assertRaisesRegex(ValueError, "artifact_kind"):
                            service.predict_joint(np.zeros(OBSERVATION_DIM, dtype=np.float32), fallback_to_heuristic=False)

                    loader.assert_not_called()

    def test_wrong_contract_or_dim_raises(self) -> None:
        bad_cases = (
            {"contract": "old_contract"},
            {"obs_dim": OBSERVATION_DIM - 1},
            {"action_count": DISCRETE_ACTION_COUNT + 1},
        )
        for bad_metadata in bad_cases:
            with self.subTest(bad_metadata=bad_metadata):
                with TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    registry = ModelRegistry(root / "registry")
                    _register_active_torch_pair(root, registry, ppo_metadata=bad_metadata)
                    service = PolicyService(registry=registry)

                    with patch("src.orchestration.policy_service.load_torch_joint_policy") as loader:
                        with self.assertRaisesRegex(ValueError, "contract|obs_dim|action_count"):
                            service.predict_joint(np.zeros(OBSERVATION_DIM, dtype=np.float32), fallback_to_heuristic=False)

                    loader.assert_not_called()

    def test_sb3_active_zip_path_still_uses_existing_sb3_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            registry = ModelRegistry(root / "registry")
            ppo_zip = root / "ppo.zip"
            dqn_zip = root / "dqn.zip"
            ppo_zip.write_text("placeholder", encoding="utf-8")
            dqn_zip.write_text("placeholder", encoding="utf-8")
            registry.register(
                algorithm="ppo",
                path=ppo_zip,
                agent_role=PPO_ROLE,
                status="active",
                metadata={"framework": "sb3", "runtime_loader": "sb3"},
            )
            registry.register(
                algorithm="dqn",
                path=dqn_zip,
                agent_role=DQN_ROLE,
                status="active",
                metadata={"framework": "sb3", "runtime_loader": "sb3"},
            )
            service = PolicyService(registry=registry)

            with patch("src.orchestration.policy_service.PPO.load", return_value=_FakeSb3Model(np.ones(CONTINUOUS_ACTION_DIM, dtype=np.float32))) as ppo_load:
                ppo_inference = service.predict(
                    np.zeros(OBSERVATION_DIM, dtype=np.float32),
                    algorithm="ppo",
                    agent_role=PPO_ROLE,
                    fallback_to_heuristic=False,
                )
            with patch("src.orchestration.policy_service.DQN.load", return_value=_FakeSb3Model(np.array([3], dtype=np.int64))) as dqn_load:
                dqn_inference = service.predict(
                    np.zeros(OBSERVATION_DIM, dtype=np.float32),
                    algorithm="dqn",
                    agent_role=DQN_ROLE,
                    fallback_to_heuristic=False,
                )

            ppo_load.assert_called_once_with(ppo_zip)
            dqn_load.assert_called_once_with(dqn_zip)
            self.assertEqual("ppo", ppo_inference.algorithm)
            self.assertEqual("dqn", dqn_inference.algorithm)
            self.assertEqual(3, dqn_inference.action)

    def test_partial_sb3_active_state_still_uses_sb3_and_heuristic_joint_path(self) -> None:
        for agent_role in (PPO_ROLE, None):
            with self.subTest(agent_role=agent_role):
                with TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    registry = ModelRegistry(root / "registry")
                    ppo_zip = root / "ppo.zip"
                    ppo_zip.write_text("placeholder", encoding="utf-8")
                    registry.register(
                        algorithm="ppo",
                        path=ppo_zip,
                        agent_role=agent_role,
                        status="active",
                        metadata={"framework": "sb3", "runtime_loader": "sb3"},
                    )
                    service = PolicyService(registry=registry)

                    with patch("src.orchestration.policy_service.load_torch_joint_policy") as joint_loader:
                        with patch(
                            "src.orchestration.policy_service.PPO.load",
                            return_value=_FakeSb3Model(np.ones(CONTINUOUS_ACTION_DIM, dtype=np.float32)),
                        ) as ppo_load:
                            with patch("src.orchestration.policy_service.DQN.load") as dqn_load:
                                inference = service.predict_joint(
                                    np.zeros(OBSERVATION_DIM, dtype=np.float32),
                                    fallback_to_heuristic=True,
                                )

                    joint_loader.assert_not_called()
                    ppo_load.assert_called_once_with(ppo_zip)
                    dqn_load.assert_not_called()
                    self.assertEqual("joint", inference.algorithm)
                    self.assertEqual({"ppo": str(ppo_zip), "dqn": None}, inference.model_path)


def _touch_paths(root: Path) -> dict[str, Path]:
    paths = {
        "joint": root / "joint_torch_latest.pt",
        "ppo": root / "ppo_torch_joint_final.pt",
        "dqn": root / "dqn_torch_joint_final.pt",
    }
    for path in paths.values():
        path.write_text("placeholder", encoding="utf-8")
    return paths


def _torch_joint_metadata(root: Path, **overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "logical_model_id": "paired_active",
        "joint_checkpoint": str(root / "joint_torch_latest.pt"),
        "framework": "torch_joint",
        "runtime_loader": "joint_torch",
        "contract": EXPECTED_MDP_CONTRACT,
        "obs_dim": OBSERVATION_DIM,
        "action_count": DISCRETE_ACTION_COUNT,
    }
    metadata.update(overrides)
    return metadata


def _register_active_torch_pair(
    root: Path,
    registry: ModelRegistry,
    *,
    ppo_metadata: dict[str, object] | None = None,
    dqn_metadata: dict[str, object] | None = None,
) -> dict[str, Path]:
    paths = _touch_paths(root)
    base_metadata = _torch_joint_metadata(root)
    registry.register(
        algorithm="ppo",
        path=paths["ppo"],
        agent_role=PPO_ROLE,
        status="active",
        metadata={**base_metadata, **(ppo_metadata or {}), "artifact_kind": ppo_metadata.get("artifact_kind", "ppo_final") if ppo_metadata else "ppo_final"},
    )
    registry.register(
        algorithm="dqn",
        path=paths["dqn"],
        agent_role=DQN_ROLE,
        status="active",
        metadata={**base_metadata, **(dqn_metadata or {}), "artifact_kind": dqn_metadata.get("artifact_kind", "dqn_final") if dqn_metadata else "dqn_final"},
    )
    return paths


if __name__ == "__main__":
    unittest.main()
