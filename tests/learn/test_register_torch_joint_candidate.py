from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
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
from src.think.joint_policies import HIERARCHICAL_DQN_ARCHITECTURE, JointPolicyBundle
from src.orchestration.torch_joint_runtime import EXPECTED_MDP_CONTRACT


LOGICAL_MODEL_ID = "joint_torch_v5_balanced_retention_ft_200k_20260608"
HIERARCHICAL_LOGICAL_MODEL_ID = "joint_torch_v5_prod_hierarchical_v1_1m_20260611"
HIERARCHICAL_INIT_METHOD = "flat_teacher_distillation_v1"
PARENT_ENV_ID = "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean"
ENV_ID = "joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k"
PPO_ROLE = "continuous_control"
DQN_ROLE = "tactical_dispatch"
DELETE_CHECKPOINT_KEY = object()


class RegisterTorchJointCandidateTests(unittest.TestCase):
    def test_dry_run_does_not_write_registry(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, registry, records = _create_registry_with_pair(Path(temp_dir))
            before_index = registry.index_path.read_text(encoding="utf-8")
            before_active = registry.active_path.read_text(encoding="utf-8")

            code, output = _run_cli(
                "--logical-model-id",
                LOGICAL_MODEL_ID,
                "--ppo-candidate-id",
                records["ppo"].model_id,
                "--dqn-candidate-id",
                records["dqn"].model_id,
                "--registry-dir",
                str(registry.registry_dir),
                "--dry-run",
            )

            self.assertEqual(0, code)
            self.assertIn("DRY_RUN_OK", output)
            self.assertIn("would activate", output.lower())
            self.assertEqual(before_index, registry.index_path.read_text(encoding="utf-8"))
            self.assertEqual(before_active, registry.active_path.read_text(encoding="utf-8"))
            self.assertFalse((root / "models" / "production").exists())
            self.assertFalse((root / "models" / "baselines").exists())

    def test_activate_writes_two_active_records_and_active_models(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, registry, records = _create_registry_with_pair(Path(temp_dir))

            code, output = _run_cli(
                "--logical-model-id",
                LOGICAL_MODEL_ID,
                "--ppo-candidate-id",
                records["ppo"].model_id,
                "--dqn-candidate-id",
                records["dqn"].model_id,
                "--registry-dir",
                str(registry.registry_dir),
                "--activate",
            )

            self.assertEqual(0, code)
            self.assertIn("ACTIVATION_COMPLETE", output)
            lines = [json.loads(line) for line in registry.index_path.read_text(encoding="utf-8").splitlines()]
            active_rows = [row for row in lines if row["status"] == "active"]
            self.assertEqual(2, len(active_rows))
            active = json.loads(registry.active_path.read_text(encoding="utf-8"))
            self.assertEqual(str(root / "ppo_torch_joint_final.pt"), active[f"ppo:{PPO_ROLE}"])
            self.assertEqual(str(root / "dqn_torch_joint_final.pt"), active[f"dqn:{DQN_ROLE}"])
            by_algorithm = {row["algorithm"]: row for row in active_rows}
            self.assertEqual(records["ppo"].model_id, by_algorithm["ppo"]["metadata"]["promoted_from_candidate_id"])
            self.assertEqual(records["dqn"].model_id, by_algorithm["dqn"]["metadata"]["promoted_from_candidate_id"])
            self.assertEqual("registry_active_only", by_algorithm["ppo"]["metadata"]["activation_scope"])
            self.assertFalse(by_algorithm["ppo"]["metadata"]["production_promotion"])
            self.assertFalse(by_algorithm["ppo"]["metadata"]["baseline_update"])
            self.assertFalse((root / "models" / "production").exists())
            self.assertFalse((root / "models" / "baselines").exists())

    def test_omitted_mode_defaults_to_dry_run(self) -> None:
        with TemporaryDirectory() as temp_dir:
            _, registry, records = _create_registry_with_pair(Path(temp_dir))
            before_index = registry.index_path.read_text(encoding="utf-8")
            before_active = registry.active_path.read_text(encoding="utf-8")

            code, output = _run_cli(
                "--logical-model-id",
                LOGICAL_MODEL_ID,
                "--ppo-candidate-id",
                records["ppo"].model_id,
                "--dqn-candidate-id",
                records["dqn"].model_id,
                "--registry-dir",
                str(registry.registry_dir),
            )

            self.assertEqual(0, code)
            self.assertIn("DRY_RUN_OK", output)
            self.assertEqual(before_index, registry.index_path.read_text(encoding="utf-8"))
            self.assertEqual(before_active, registry.active_path.read_text(encoding="utf-8"))

    def test_dry_run_missing_registry_dir_does_not_create_it(self) -> None:
        with TemporaryDirectory() as temp_dir:
            missing_registry = Path(temp_dir) / "missing_registry"

            with self.assertRaisesRegex(FileNotFoundError, "registry"):
                _run_cli(
                    "--logical-model-id",
                    LOGICAL_MODEL_ID,
                    "--ppo-candidate-id",
                    "missing-ppo",
                    "--dqn-candidate-id",
                    "missing-dqn",
                    "--registry-dir",
                    str(missing_registry),
                    "--dry-run",
                )

            self.assertFalse(missing_registry.exists())

    def test_refuses_both_dry_run_and_activate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            _, registry, records = _create_registry_with_pair(Path(temp_dir))

            with self.assertRaisesRegex(ValueError, "dry-run.*activate|activate.*dry-run"):
                _run_cli(
                    "--logical-model-id",
                    LOGICAL_MODEL_ID,
                    "--ppo-candidate-id",
                    records["ppo"].model_id,
                    "--dqn-candidate-id",
                    records["dqn"].model_id,
                    "--registry-dir",
                    str(registry.registry_dir),
                    "--dry-run",
                    "--activate",
                )

    def test_refuses_wrong_candidate_ids(self) -> None:
        with TemporaryDirectory() as temp_dir:
            _, registry, records = _create_registry_with_pair(Path(temp_dir))

            with self.assertRaisesRegex(ValueError, "candidate"):
                _run_cli(
                    "--logical-model-id",
                    LOGICAL_MODEL_ID,
                    "--ppo-candidate-id",
                    "missing-ppo",
                    "--dqn-candidate-id",
                    records["dqn"].model_id,
                    "--registry-dir",
                    str(registry.registry_dir),
                    "--activate",
                )

    def test_refuses_mismatched_logical_model_id(self) -> None:
        self._assert_refuses_pair(
            ppo_metadata={"logical_model_id": "ppo_model"},
            dqn_metadata={"logical_model_id": "dqn_model"},
            message="logical_model_id",
        )

    def test_refuses_mismatched_joint_checkpoint(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            other_joint = root / "other_joint.pt"
            other_joint.write_text("placeholder", encoding="utf-8")
            _, registry, records = _create_registry_with_pair(root, dqn_metadata={"joint_checkpoint": str(other_joint)})

            with self.assertRaisesRegex(ValueError, "joint_checkpoint"):
                _activate(registry, records)

    def test_refuses_non_candidate_status(self) -> None:
        with TemporaryDirectory() as temp_dir:
            _, registry, records = _create_registry_with_pair(Path(temp_dir), ppo_status="archived")

            with self.assertRaisesRegex(ValueError, "status"):
                _activate(registry, records)

    def test_refuses_failed_eval_verdict(self) -> None:
        self._assert_refuses_pair(ppo_metadata={"eval_verdict": "FAIL"}, message="eval_verdict")

    def test_refuses_nonzero_hard_blocker_status(self) -> None:
        self._assert_refuses_pair(dqn_metadata={"hard_blocker_status": "nonzero"}, message="hard_blocker_status")

    def test_refuses_wrong_framework_or_runtime_loader(self) -> None:
        for bad_metadata in ({"framework": "sb3"}, {"runtime_loader": "sb3"}):
            with self.subTest(bad_metadata=bad_metadata):
                self._assert_refuses_pair(ppo_metadata=bad_metadata, message="framework|runtime_loader")

    def test_refuses_wrong_artifact_kind_pair(self) -> None:
        self._assert_refuses_pair(ppo_metadata={"artifact_kind": "dqn_final"}, message="artifact_kind")
        self._assert_refuses_pair(dqn_metadata={"artifact_kind": "ppo_final"}, message="artifact_kind")

    def test_refuses_wrong_contract_or_dims(self) -> None:
        for bad_metadata in (
            {"contract": "old_contract"},
            {"obs_dim": 72},
            {"action_count": 49},
        ):
            with self.subTest(bad_metadata=bad_metadata):
                self._assert_refuses_pair(ppo_metadata=bad_metadata, message="contract|obs_dim|action_count")

    def test_refuses_missing_artifact_paths(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, registry, records = _create_registry_with_pair(Path(temp_dir))
            (root / "ppo_torch_joint_final.pt").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "artifact"):
                _activate(registry, records)

    def test_idempotency_refuses_duplicate_active_activation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            _, registry, records = _create_registry_with_pair(Path(temp_dir))
            _activate(registry, records)

            with self.assertRaisesRegex(ValueError, "already active|duplicate"):
                _activate(registry, records)

    def test_activation_refuses_existing_bare_active_mapping(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, registry, records = _create_registry_with_pair(Path(temp_dir))
            registry.active_path.write_text(json.dumps({"ppo": str(root / "legacy_ppo.zip")}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "active registry mapping"):
                _activate(registry, records)

    def test_activation_dry_run_refuses_existing_active_mapping_without_replace_flag(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, registry, records = _create_hierarchical_registry_pair(Path(temp_dir))
            registry.active_path.write_text(
                json.dumps(
                    {
                        f"ppo:{PPO_ROLE}": str(root / "old_ppo.pt"),
                        f"dqn:{DQN_ROLE}": str(root / "old_dqn.pt"),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "replace-active"):
                _run_cli(
                    "--logical-model-id",
                    HIERARCHICAL_LOGICAL_MODEL_ID,
                    "--ppo-candidate-id",
                    records["ppo"].model_id,
                    "--dqn-candidate-id",
                    records["dqn"].model_id,
                    "--registry-dir",
                    str(registry.registry_dir),
                    "--dry-run",
                )

    def test_hierarchical_activation_dry_run_replace_active_writes_nothing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, registry, records = _create_hierarchical_registry_pair(Path(temp_dir))
            registry.active_path.write_text(
                json.dumps(
                    {
                        f"ppo:{PPO_ROLE}": str(root / "old_ppo.pt"),
                        f"dqn:{DQN_ROLE}": str(root / "old_dqn.pt"),
                    }
                ),
                encoding="utf-8",
            )
            before_index = registry.index_path.read_text(encoding="utf-8")
            before_active = registry.active_path.read_text(encoding="utf-8")

            code, output = _run_cli(
                "--logical-model-id",
                HIERARCHICAL_LOGICAL_MODEL_ID,
                "--ppo-candidate-id",
                records["ppo"].model_id,
                "--dqn-candidate-id",
                records["dqn"].model_id,
                "--registry-dir",
                str(registry.registry_dir),
                "--dry-run",
                "--replace-active",
            )

            self.assertEqual(0, code)
            self.assertIn("DRY_RUN_OK", output)
            self.assertIn("would replace active mappings", output)
            self.assertIn("old_ppo.pt", output)
            self.assertIn("old_dqn.pt", output)
            self.assertEqual(before_index, registry.index_path.read_text(encoding="utf-8"))
            self.assertEqual(before_active, registry.active_path.read_text(encoding="utf-8"))

    def test_activate_with_replace_active_updates_temp_active_mapping_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, registry, records = _create_hierarchical_registry_pair(Path(temp_dir))
            registry.active_path.write_text(
                json.dumps(
                    {
                        f"ppo:{PPO_ROLE}": str(root / "old_ppo.pt"),
                        f"dqn:{DQN_ROLE}": str(root / "old_dqn.pt"),
                    }
                ),
                encoding="utf-8",
            )

            code, output = _run_cli(
                "--logical-model-id",
                HIERARCHICAL_LOGICAL_MODEL_ID,
                "--ppo-candidate-id",
                records["ppo"].model_id,
                "--dqn-candidate-id",
                records["dqn"].model_id,
                "--registry-dir",
                str(registry.registry_dir),
                "--activate",
                "--replace-active",
            )

            self.assertEqual(0, code)
            self.assertIn("ACTIVATION_COMPLETE", output)
            rows = [json.loads(line) for line in registry.index_path.read_text(encoding="utf-8").splitlines()]
            active_rows = [
                row
                for row in rows
                if row["status"] == "active"
                and row["metadata"].get("logical_model_id") == HIERARCHICAL_LOGICAL_MODEL_ID
            ]
            self.assertEqual(2, len(active_rows))
            active = json.loads(registry.active_path.read_text(encoding="utf-8"))
            self.assertEqual(str(root / "ppo_torch_joint_final.pt"), active[f"ppo:{PPO_ROLE}"])
            self.assertEqual(str(root / "dqn_torch_joint_final.pt"), active[f"dqn:{DQN_ROLE}"])

    def test_refuses_non_false_production_or_baseline_flags(self) -> None:
        for bad_metadata in (
            {"production_ready": True},
            {"production_ready": None},
            {"production_ready": "false"},
            {"baseline_update": True},
            {"baseline_update": ""},
            {"baseline_update": "none"},
        ):
            with self.subTest(bad_metadata=bad_metadata):
                self._assert_refuses_pair(ppo_metadata=bad_metadata, message="production_ready|baseline_update")

    def test_policy_service_can_resolve_temp_activated_pair(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, registry, records = _create_registry_with_pair(Path(temp_dir))
            _activate(registry, records)

            from src.orchestration.policy_service import PolicyService

            fake_action = type("FakeAction", (), {"continuous": np.zeros(5, dtype=np.float32), "discrete": 7})()
            fake_policy = type("FakeTorchPolicy", (), {"predict_joint": lambda self, observation, deterministic=True: fake_action})()
            service = PolicyService(registry=registry)
            with patch("src.orchestration.policy_service.load_torch_joint_policy", return_value=fake_policy) as loader:
                with patch("src.orchestration.policy_service.PPO.load") as ppo_load:
                    with patch("src.orchestration.policy_service.DQN.load") as dqn_load:
                        inference = service.predict_joint(
                            np.zeros(OBSERVATION_DIM, dtype=np.float32),
                            fallback_to_heuristic=False,
                        )

            self.assertEqual("torch_joint", inference.algorithm)
            loader.assert_called_once_with(root / "joint_torch_latest.pt", device=service.device)
            ppo_load.assert_not_called()
            dqn_load.assert_not_called()

    def test_hierarchical_candidate_dry_run_validates_metadata_and_writes_nothing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, registry, artifacts = _create_hierarchical_candidate_artifacts(Path(temp_dir))
            before_index = registry.index_path.read_text(encoding="utf-8")
            before_active = registry.active_path.read_text(encoding="utf-8")

            code, output = _run_cli(
                "--logical-model-id",
                HIERARCHICAL_LOGICAL_MODEL_ID,
                "--registry-dir",
                str(registry.registry_dir),
                "--joint-checkpoint",
                str(artifacts["joint"]),
                "--ppo-path",
                str(artifacts["ppo"]),
                "--dqn-path",
                str(artifacts["dqn"]),
                "--eval-output-dir",
                str(artifacts["eval"]),
                "--gate-result",
                str(artifacts["gate"]),
                "--dqn-architecture",
                HIERARCHICAL_DQN_ARCHITECTURE,
                "--hierarchical-init-method",
                HIERARCHICAL_INIT_METHOD,
                "--dry-run",
            )

            self.assertEqual(0, code)
            self.assertIn("DRY_RUN_OK", output)
            self.assertIn("would register Torch joint candidate rows", output)
            self.assertIn(f"logical_model_id: {HIERARCHICAL_LOGICAL_MODEL_ID}", output)
            self.assertIn("dqn_architecture: hierarchical_v1", output)
            self.assertIn("hierarchical_init_method: flat_teacher_distillation_v1", output)
            self.assertIn("training_step: 1000000", output)
            self.assertIn("long_run_gate_verdict: PASS", output)
            self.assertIn("registry_scope: candidate_only", output)
            self.assertEqual(before_index, registry.index_path.read_text(encoding="utf-8"))
            self.assertEqual(before_active, registry.active_path.read_text(encoding="utf-8"))

    def test_hierarchical_candidate_dry_run_requires_dqn_architecture(self) -> None:
        self._assert_refuses_hierarchical_candidate_dry_run(
            checkpoint_overrides={"dqn_architecture": DELETE_CHECKPOINT_KEY},
            cli_args={"--dqn-architecture": "hierarchical_v1"},
            message="dqn_architecture",
        )

    def test_hierarchical_candidate_dry_run_rejects_wrong_dqn_architecture(self) -> None:
        self._assert_refuses_hierarchical_candidate_dry_run(
            checkpoint_overrides={"dqn_architecture": "flat_v1"},
            cli_args={"--dqn-architecture": "hierarchical_v1"},
            message="dqn_architecture",
        )

    def test_hierarchical_candidate_dry_run_requires_init_method(self) -> None:
        self._assert_refuses_hierarchical_candidate_dry_run(
            checkpoint_overrides={"hierarchical_init_method": "random_heads"},
            cli_args={"--hierarchical-init-method": HIERARCHICAL_INIT_METHOD},
            message="hierarchical_init_method",
        )

    def test_hierarchical_candidate_dry_run_refuses_failed_eval_verdict(self) -> None:
        self._assert_refuses_hierarchical_candidate_dry_run(
            eval_overrides={"verdict": "FAIL"},
            message="eval_verdict|scenario",
        )

    def test_hierarchical_candidate_dry_run_refuses_nonzero_hard_blocker(self) -> None:
        self._assert_refuses_hierarchical_candidate_dry_run(
            eval_overrides={"fake_dispatch_credit": 1},
            message="hard blocker|fake_dispatch_credit",
        )

    def test_hierarchical_candidate_dry_run_requires_exact_resume_capable(self) -> None:
        self._assert_refuses_hierarchical_candidate_dry_run(
            checkpoint_overrides={"resume_state": {"exact_resume_capable": False}},
            message="exact_resume_capable",
        )

    def _assert_refuses_pair(
        self,
        *,
        ppo_metadata: dict[str, object] | None = None,
        dqn_metadata: dict[str, object] | None = None,
        message: str,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            _, registry, records = _create_registry_with_pair(
                Path(temp_dir),
                ppo_metadata=ppo_metadata,
                dqn_metadata=dqn_metadata,
            )

            with self.assertRaisesRegex(ValueError, message):
                _activate(registry, records)

    def _assert_refuses_hierarchical_candidate_dry_run(
        self,
        *,
        checkpoint_overrides: dict[str, object] | None = None,
        eval_overrides: dict[str, object] | None = None,
        cli_args: dict[str, str] | None = None,
        message: str,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            _, registry, artifacts = _create_hierarchical_candidate_artifacts(
                Path(temp_dir),
                checkpoint_overrides=checkpoint_overrides,
                eval_overrides=eval_overrides,
            )
            args = [
                "--logical-model-id",
                HIERARCHICAL_LOGICAL_MODEL_ID,
                "--registry-dir",
                str(registry.registry_dir),
                "--joint-checkpoint",
                str(artifacts["joint"]),
                "--ppo-path",
                str(artifacts["ppo"]),
                "--dqn-path",
                str(artifacts["dqn"]),
                "--eval-output-dir",
                str(artifacts["eval"]),
                "--gate-result",
                str(artifacts["gate"]),
                "--dry-run",
            ]
            if cli_args:
                for key, value in cli_args.items():
                    args.extend([key, value])
            with self.assertRaisesRegex(ValueError, message):
                _run_cli(*args)


def _run_cli(*args: str) -> tuple[int, str]:
    from src.learn.register_torch_joint_candidate import main

    output = StringIO()
    with redirect_stdout(output):
        code = main(list(args))
    return code, output.getvalue()


def _activate(registry: ModelRegistry, records: dict[str, object]) -> tuple[int, str]:
    return _run_cli(
        "--logical-model-id",
        LOGICAL_MODEL_ID,
        "--ppo-candidate-id",
        records["ppo"].model_id,
        "--dqn-candidate-id",
        records["dqn"].model_id,
        "--registry-dir",
        str(registry.registry_dir),
        "--activate",
    )


def _create_registry_with_pair(
    root: Path,
    *,
    ppo_metadata: dict[str, object] | None = None,
    dqn_metadata: dict[str, object] | None = None,
    ppo_status: str = "candidate",
    dqn_status: str = "candidate",
) -> tuple[Path, ModelRegistry, dict[str, object]]:
    paths = _touch_candidate_paths(root)
    registry = ModelRegistry(root / "registry")
    registry.active_path.write_text("{}\n", encoding="utf-8")
    base_metadata = _candidate_metadata(root)
    ppo_record = registry.register(
        algorithm="ppo",
        path=paths["ppo"],
        agent_role=PPO_ROLE,
        status=ppo_status,
        metrics={"scenario_pass_rate": 1.0},
        metadata={**base_metadata, **(ppo_metadata or {}), "artifact_kind": (ppo_metadata or {}).get("artifact_kind", "ppo_final")},
    )
    dqn_record = registry.register(
        algorithm="dqn",
        path=paths["dqn"],
        agent_role=DQN_ROLE,
        status=dqn_status,
        metrics={"scenario_pass_rate": 1.0},
        metadata={**base_metadata, **(dqn_metadata or {}), "artifact_kind": (dqn_metadata or {}).get("artifact_kind", "dqn_final")},
    )
    return root, registry, {"ppo": ppo_record, "dqn": dqn_record}


def _create_hierarchical_registry_pair(root: Path) -> tuple[Path, ModelRegistry, dict[str, object]]:
    paths = _touch_candidate_paths(root)
    registry = ModelRegistry(root / "registry")
    registry.active_path.write_text("{}\n", encoding="utf-8")
    metadata = _hierarchical_candidate_metadata(root)
    ppo_record = registry.register(
        algorithm="ppo",
        path=paths["ppo"],
        agent_role=PPO_ROLE,
        status="candidate",
        metrics={"scenario_pass_rate": 1.0},
        metadata={**metadata, "artifact_kind": "ppo_final"},
    )
    dqn_record = registry.register(
        algorithm="dqn",
        path=paths["dqn"],
        agent_role=DQN_ROLE,
        status="candidate",
        metrics={"scenario_pass_rate": 1.0},
        metadata={**metadata, "artifact_kind": "dqn_final"},
    )
    return root, registry, {"ppo": ppo_record, "dqn": dqn_record}


def _touch_candidate_paths(root: Path) -> dict[str, Path]:
    paths = {
        "joint": root / "joint_torch_latest.pt",
        "ppo": root / "ppo_torch_joint_final.pt",
        "dqn": root / "dqn_torch_joint_final.pt",
    }
    for path in paths.values():
        path.write_text("placeholder", encoding="utf-8")
    (root / "eval").mkdir()
    return paths


def _candidate_metadata(root: Path) -> dict[str, object]:
    return {
        "logical_model_id": LOGICAL_MODEL_ID,
        "env_id": ENV_ID,
        "parent_env_id": PARENT_ENV_ID,
        "joint_checkpoint": str(root / "joint_torch_latest.pt"),
        "parent_checkpoint": str(root / "parent_joint_torch_latest.pt"),
        "framework": "torch_joint",
        "runtime_loader": "joint_torch",
        "contract": EXPECTED_MDP_CONTRACT,
        "obs_dim": 73,
        "action_count": 48,
        "training_step": 200000,
        "parent_step": 1000000,
        "eval_output_dir": str(root / "eval"),
        "eval_verdict": "PASS",
        "hard_blocker_status": "zero",
        "production_ready": False,
        "baseline_update": False,
    }


def _hierarchical_candidate_metadata(root: Path) -> dict[str, object]:
    return {
        "logical_model_id": HIERARCHICAL_LOGICAL_MODEL_ID,
        "joint_checkpoint": str(root / "joint_torch_latest.pt"),
        "framework": "torch_joint",
        "runtime_loader": "joint_torch",
        "contract": EXPECTED_MDP_CONTRACT,
        "obs_dim": 73,
        "action_count": 48,
        "external_discrete_action_count": 48,
        "dqn_architecture": HIERARCHICAL_DQN_ARCHITECTURE,
        "hierarchical_init_method": HIERARCHICAL_INIT_METHOD,
        "training_step": 1_000_000,
        "exact_resume_capable": True,
        "eval_output_dir": str(root / "eval"),
        "eval_verdict": "PASS",
        "hard_blocker_status": "zero",
        "long_run_gate_verdict": "PASS",
        "registry_scope": "candidate_only",
        "production_ready": False,
        "baseline_update": False,
    }


def _create_hierarchical_candidate_artifacts(
    root: Path,
    *,
    checkpoint_overrides: dict[str, object] | None = None,
    eval_overrides: dict[str, object] | None = None,
) -> tuple[Path, ModelRegistry, dict[str, Path]]:
    registry = ModelRegistry(root / "registry")
    registry.active_path.write_text("{}\n", encoding="utf-8")
    registry.index_path.write_text("", encoding="utf-8")
    artifacts = {
        "joint": root / "joint_torch_latest.pt",
        "ppo": root / "ppo_torch_joint_final_hierarchical_v1_1m.pt",
        "dqn": root / "dqn_torch_joint_final_hierarchical_v1_1m.pt",
        "eval": root / "eval",
        "gate": root / "gate_result.json",
    }
    _write_hierarchical_checkpoint(artifacts["joint"], checkpoint_overrides=checkpoint_overrides)
    artifacts["ppo"].write_bytes(b"ppo")
    artifacts["dqn"].write_bytes(b"dqn")
    _write_eval_dir(artifacts["eval"], eval_overrides=eval_overrides)
    artifacts["gate"].write_text(json.dumps({"decision": "PASS", "fatal_failures": []}), encoding="utf-8")
    return root, registry, artifacts


def _write_hierarchical_checkpoint(path: Path, *, checkpoint_overrides: dict[str, object] | None = None) -> None:
    policy = JointPolicyBundle(dqn_architecture=HIERARCHICAL_DQN_ARCHITECTURE)
    checkpoint = policy.build_checkpoint(
        global_step=1_000_000,
        config={
            "mdp_contract_version": EXPECTED_MDP_CONTRACT,
            "shared_global_parameters": {"observation_dim": OBSERVATION_DIM},
        },
    )
    checkpoint.update(
        {
            "artifact_kind": "joint_final",
            "hierarchical_init_method": HIERARCHICAL_INIT_METHOD,
            "resume_state": {"exact_resume_capable": True},
            "dqn_replay_buffer_state": {
                "size": 500_000,
                "position": 0,
                "total_added": 1_000_000,
            },
            "rng_state": {"python": "present", "numpy": "present", "torch": "present"},
        }
    )
    if checkpoint_overrides:
        for key, value in checkpoint_overrides.items():
            if value is DELETE_CHECKPOINT_KEY:
                checkpoint.pop(key, None)
            else:
                checkpoint[key] = value
    torch.save(checkpoint, path)


def _write_eval_dir(path: Path, *, eval_overrides: dict[str, object] | None = None) -> None:
    path.mkdir(parents=True)
    scenario = {
        "scenario_id": "scenario",
        "verdict": "PASS",
        "hard_blocker_verdict": "PASS",
        "episodes": 20,
        "fake_dispatch_credit": 0,
        "customer_revisited": 0,
        "route_failure_positive_dispatch_credit": 0,
        "nan_inf_detected": False,
        "dqn_local_negative_positive_train_rows": 0,
        "no_current_work_dqn_delivery_credit": 0,
        "hold_delivery_credit_leak": 0,
        "action8_route_or_delivery_credit_leak": 0,
        "unsafe_24_25_candidate_credit": 0,
        "no_work_positive_dqn_local": 0,
        "emergency_zero_useful_positive_credit": 0,
    }
    if eval_overrides:
        scenario.update(eval_overrides)
    scenarios = [{**scenario, "scenario_id": f"scenario_{idx}"} for idx in range(8)]
    if eval_overrides:
        scenarios[0].update(eval_overrides)
    (path / "scenario_summary.json").write_text(
        json.dumps(
            {
                "checkpoint": "joint_torch_latest.pt",
                "contract": EXPECTED_MDP_CONTRACT,
                "observation_dim": OBSERVATION_DIM,
                "scenarios": scenarios,
            }
        ),
        encoding="utf-8",
    )
    (path / "episode_metrics.jsonl").write_text(
        "\n".join(json.dumps({"episode": idx}) for idx in range(160)) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
