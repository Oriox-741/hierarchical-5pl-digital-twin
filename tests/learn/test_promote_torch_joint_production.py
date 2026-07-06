from __future__ import annotations

import json
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.observation_builder import OBSERVATION_DIM
from src.learn.model_registry import ModelRegistry
from src.orchestration.policy_service import DQN_AGENT_ROLE, PPO_AGENT_ROLE
from src.orchestration.torch_joint_runtime import EXPECTED_MDP_CONTRACT, load_torch_joint_policy
from src.think.joint_policies import JointPolicyBundle


LOGICAL_MODEL_ID = "joint_torch_v5_balanced_retention_ft_200k_20260608"
ENV_ID = "joint_curriculum_v5_clean_after_rewardfix_perfclean_balanced_retention_ft_200k"
PARENT_ENV_ID = "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean"


@dataclass(frozen=True, slots=True)
class PromotionFixture:
    root: Path
    registry: ModelRegistry
    production_root: Path
    baseline_marker: Path
    eval_output_dir: Path
    joint_checkpoint: Path
    ppo_artifact: Path
    dqn_artifact: Path
    ppo_model_id: str
    dqn_model_id: str

    @property
    def production_dir(self) -> Path:
        return self.production_root / LOGICAL_MODEL_ID


class PromoteTorchJointProductionTests(unittest.TestCase):
    def test_dry_run_does_not_create_production_dir(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fixture = _create_active_fixture(Path(temp_dir))
            before_index = fixture.registry.index_path.read_text(encoding="utf-8")
            before_active = fixture.registry.active_path.read_text(encoding="utf-8")

            code, output = _run_cli(
                "--logical-model-id",
                LOGICAL_MODEL_ID,
                "--registry-dir",
                str(fixture.registry.registry_dir),
                "--production-root",
                str(fixture.production_root),
                "--dry-run",
            )

            self.assertEqual(0, code)
            self.assertIn("DRY_RUN_OK", output)
            self.assertIn("planned copy", output.lower())
            self.assertIn("production_manifest.json", output)
            self.assertFalse(fixture.production_dir.exists())
            self.assertFalse(fixture.production_root.exists())
            self.assertEqual(before_index, fixture.registry.index_path.read_text(encoding="utf-8"))
            self.assertEqual(before_active, fixture.registry.active_path.read_text(encoding="utf-8"))

    def test_promote_copies_exact_three_artifacts_and_manifest(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fixture = _create_active_fixture(Path(temp_dir))

            code, output = _promote(fixture)

            self.assertEqual(0, code)
            self.assertIn("PRODUCTION_PROMOTION_COMPLETE", output)
            files = {path.name for path in fixture.production_dir.iterdir()}
            self.assertEqual(
                {
                    "joint_torch_latest.pt",
                    "ppo_torch_joint_final_balanced_retention_ft_200k.pt",
                    "dqn_torch_joint_final_balanced_retention_ft_200k.pt",
                    "production_manifest.json",
                },
                files,
            )
            self.assertEqual(
                fixture.joint_checkpoint.stat().st_size,
                (fixture.production_dir / "joint_torch_latest.pt").stat().st_size,
            )
            self.assertEqual(
                fixture.ppo_artifact.stat().st_size,
                (fixture.production_dir / "ppo_torch_joint_final_balanced_retention_ft_200k.pt").stat().st_size,
            )
            self.assertEqual(
                fixture.dqn_artifact.stat().st_size,
                (fixture.production_dir / "dqn_torch_joint_final_balanced_retention_ft_200k.pt").stat().st_size,
            )

    def test_manifest_contains_required_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fixture = _create_active_fixture(Path(temp_dir))
            _promote(fixture)

            manifest = json.loads((fixture.production_dir / "production_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(LOGICAL_MODEL_ID, manifest["logical_model_id"])
            self.assertEqual(fixture.ppo_model_id, manifest["active_ppo_model_id"])
            self.assertEqual(fixture.dqn_model_id, manifest["active_dqn_model_id"])
            self.assertEqual(str(fixture.joint_checkpoint), manifest["source_joint_checkpoint"])
            self.assertEqual(str(fixture.ppo_artifact), manifest["source_ppo_artifact"])
            self.assertEqual(str(fixture.dqn_artifact), manifest["source_dqn_artifact"])
            self.assertEqual(str(fixture.production_dir / "joint_torch_latest.pt"), manifest["production_joint_checkpoint"])
            self.assertEqual(
                str(fixture.production_dir / "ppo_torch_joint_final_balanced_retention_ft_200k.pt"),
                manifest["production_ppo_artifact"],
            )
            self.assertEqual(
                str(fixture.production_dir / "dqn_torch_joint_final_balanced_retention_ft_200k.pt"),
                manifest["production_dqn_artifact"],
            )
            self.assertEqual(EXPECTED_MDP_CONTRACT, manifest["contract"])
            self.assertEqual(OBSERVATION_DIM, manifest["observation_dim"])
            self.assertEqual(CONTINUOUS_ACTION_DIM, manifest["continuous_action_dim"])
            self.assertEqual(DISCRETE_ACTION_COUNT, manifest["discrete_action_count"])
            self.assertEqual(200000, manifest["training_step"])
            self.assertEqual(1000000, manifest["parent_step"])
            self.assertEqual(str(fixture.eval_output_dir), manifest["eval_output_dir"])
            self.assertEqual("PASS", manifest["eval_verdict"])
            self.assertEqual("zero", manifest["hard_blocker_status"])
            self.assertIn("route_disruption_congestion.high_resilience_selected_when_not_best_count=411", manifest["residual_watches"])
            self.assertTrue(manifest["residual_watches_accepted"])
            self.assertEqual("egeme", manifest["human_approver"])
            self.assertIn("promoted_at", manifest)
            self.assertTrue(manifest["registry_active_scope"])
            self.assertEqual("copy_only", manifest["production_promotion_scope"])
            self.assertFalse(manifest["baseline_update"])
            self.assertFalse(manifest["db_mutation"])

    def test_refuses_without_human_approver_on_promote(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fixture = _create_active_fixture(Path(temp_dir))

            with self.assertRaisesRegex(ValueError, "human approver"):
                _run_cli(
                    "--logical-model-id",
                    LOGICAL_MODEL_ID,
                    "--registry-dir",
                    str(fixture.registry.registry_dir),
                    "--production-root",
                    str(fixture.production_root),
                    "--accept-residual-watches",
                    "--promote",
                )
            self.assertFalse(fixture.production_root.exists())

    def test_refuses_without_watch_acceptance_on_promote(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fixture = _create_active_fixture(Path(temp_dir))

            with self.assertRaisesRegex(ValueError, "residual watch"):
                _run_cli(
                    "--logical-model-id",
                    LOGICAL_MODEL_ID,
                    "--registry-dir",
                    str(fixture.registry.registry_dir),
                    "--production-root",
                    str(fixture.production_root),
                    "--human-approver",
                    "egeme",
                    "--promote",
                )
            self.assertFalse(fixture.production_root.exists())

    def test_dry_run_allows_no_approver_but_reports_missing_approval(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fixture = _create_active_fixture(Path(temp_dir))

            code, output = _run_cli(
                "--logical-model-id",
                LOGICAL_MODEL_ID,
                "--registry-dir",
                str(fixture.registry.registry_dir),
                "--production-root",
                str(fixture.production_root),
                "--dry-run",
            )

            self.assertEqual(0, code)
            self.assertIn("DRY_RUN_OK", output)
            self.assertIn("human_approver_missing: true", output)
            self.assertIn("residual_watches_accepted: false", output)
            self.assertFalse(fixture.production_root.exists())

    def test_omitted_mode_defaults_to_dry_run(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fixture = _create_active_fixture(Path(temp_dir))

            code, output = _run_cli(
                "--logical-model-id",
                LOGICAL_MODEL_ID,
                "--registry-dir",
                str(fixture.registry.registry_dir),
                "--production-root",
                str(fixture.production_root),
            )

            self.assertEqual(0, code)
            self.assertIn("DRY_RUN_OK", output)
            self.assertFalse(fixture.production_root.exists())

    def test_refuses_both_dry_run_and_promote(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fixture = _create_active_fixture(Path(temp_dir))

            with self.assertRaisesRegex(ValueError, "dry-run.*promote|promote.*dry-run"):
                _run_cli(
                    "--logical-model-id",
                    LOGICAL_MODEL_ID,
                    "--registry-dir",
                    str(fixture.registry.registry_dir),
                    "--production-root",
                    str(fixture.production_root),
                    "--dry-run",
                    "--promote",
                )
            self.assertFalse(fixture.production_root.exists())

    def test_refuses_existing_production_dir(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fixture = _create_active_fixture(Path(temp_dir))
            fixture.production_dir.mkdir(parents=True)

            with self.assertRaisesRegex(FileExistsError, "production target"):
                _promote(fixture)

    def test_refuses_unsafe_production_dir_name(self) -> None:
        for unsafe_name in ("..\\baselines", "../baselines", "nested/name", "C:\\temp\\prod"):
            with self.subTest(unsafe_name=unsafe_name):
                with TemporaryDirectory() as temp_dir:
                    fixture = _create_active_fixture(Path(temp_dir))

                    with self.assertRaisesRegex(ValueError, "production-dir-name"):
                        _run_cli(
                            "--logical-model-id",
                            LOGICAL_MODEL_ID,
                            "--registry-dir",
                            str(fixture.registry.registry_dir),
                            "--production-root",
                            str(fixture.production_root),
                            "--production-dir-name",
                            unsafe_name,
                            "--dry-run",
                        )
                    self.assertFalse(fixture.production_root.exists())

    def test_refuses_non_pt_source_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fixture = _create_active_fixture(Path(temp_dir))
            ppo_zip = fixture.ppo_artifact.with_suffix(".zip")
            ppo_zip.write_bytes(fixture.ppo_artifact.read_bytes())
            _rewrite_active_artifact_path(fixture, "ppo", ppo_zip)

            with self.assertRaisesRegex(ValueError, r"\.pt"):
                _dry_run(fixture)

    def test_refuses_duplicate_destination_basenames(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fixture = _create_active_fixture(Path(temp_dir))
            ppo_duplicate = fixture.root / "ppo" / "model.pt"
            dqn_duplicate = fixture.root / "dqn" / "model.pt"
            ppo_duplicate.parent.mkdir()
            dqn_duplicate.parent.mkdir()
            ppo_duplicate.write_bytes(b"ppo")
            dqn_duplicate.write_bytes(b"dqn")
            _rewrite_active_artifact_path(fixture, "ppo", ppo_duplicate)
            _rewrite_active_artifact_path(fixture, "dqn", dqn_duplicate)

            with self.assertRaisesRegex(ValueError, "unique|duplicate"):
                _dry_run(fixture)

    def test_refuses_extra_active_registry_keys(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fixture = _create_active_fixture(Path(temp_dir))
            active = json.loads(fixture.registry.active_path.read_text(encoding="utf-8"))
            active["ppo:shadow"] = str(fixture.ppo_artifact)
            fixture.registry.active_path.write_text(json.dumps(active), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "only intended"):
                _dry_run(fixture)

    def test_refuses_incomplete_active_pair(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fixture = _create_active_fixture(Path(temp_dir))
            active = json.loads(fixture.registry.active_path.read_text(encoding="utf-8"))
            active.pop(f"dqn:{DQN_AGENT_ROLE}")
            fixture.registry.active_path.write_text(json.dumps(active), encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "active dqn"):
                _dry_run(fixture)

    def test_refuses_wrong_contract_or_dims(self) -> None:
        for metadata in (
            {"contract": "old_contract"},
            {"obs_dim": OBSERVATION_DIM - 1},
            {"action_count": DISCRETE_ACTION_COUNT + 1},
        ):
            with self.subTest(metadata=metadata):
                with TemporaryDirectory() as temp_dir:
                    fixture = _create_active_fixture(Path(temp_dir), ppo_metadata=metadata)

                    with self.assertRaisesRegex(ValueError, "contract|obs_dim|action_count"):
                        _dry_run(fixture)

    def test_refuses_eval_not_pass_or_hard_blockers_nonzero(self) -> None:
        for metadata in (
            {"eval_verdict": "FAIL"},
            {"hard_blocker_status": "nonzero"},
        ):
            with self.subTest(metadata=metadata):
                with TemporaryDirectory() as temp_dir:
                    fixture = _create_active_fixture(Path(temp_dir), dqn_metadata=metadata)

                    with self.assertRaisesRegex(ValueError, "eval_verdict|hard_blocker_status"):
                        _dry_run(fixture)

    def test_does_not_touch_registry_or_baselines(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fixture = _create_active_fixture(Path(temp_dir))
            before_index = fixture.registry.index_path.read_text(encoding="utf-8")
            before_active = fixture.registry.active_path.read_text(encoding="utf-8")
            before_baseline = fixture.baseline_marker.read_text(encoding="utf-8")
            before_sources = {
                fixture.joint_checkpoint: fixture.joint_checkpoint.read_bytes(),
                fixture.ppo_artifact: fixture.ppo_artifact.read_bytes(),
                fixture.dqn_artifact: fixture.dqn_artifact.read_bytes(),
            }

            _promote(fixture)

            self.assertEqual(before_index, fixture.registry.index_path.read_text(encoding="utf-8"))
            self.assertEqual(before_active, fixture.registry.active_path.read_text(encoding="utf-8"))
            self.assertEqual(before_baseline, fixture.baseline_marker.read_text(encoding="utf-8"))
            for source, contents in before_sources.items():
                self.assertEqual(contents, source.read_bytes())

    def test_post_copy_loader_smoke(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fixture = _create_active_fixture(Path(temp_dir))
            _promote(fixture)

            policy = load_torch_joint_policy(fixture.production_dir / "joint_torch_latest.pt", device="cpu")
            action = policy.predict_joint(np.zeros(OBSERVATION_DIM, dtype=np.float32), deterministic=True)

            self.assertEqual((CONTINUOUS_ACTION_DIM,), action.continuous.shape)
            self.assertTrue(np.all(np.isfinite(action.continuous)))
            self.assertTrue(np.all(action.continuous >= -1.0))
            self.assertTrue(np.all(action.continuous <= 1.0))
            self.assertGreaterEqual(action.discrete, 0)
            self.assertLess(action.discrete, DISCRETE_ACTION_COUNT)


def _run_cli(*args: str) -> tuple[int, str]:
    from src.learn.promote_torch_joint_production import main

    output = StringIO()
    with redirect_stdout(output):
        code = main(list(args))
    return code, output.getvalue()


def _dry_run(fixture: PromotionFixture) -> tuple[int, str]:
    return _run_cli(
        "--logical-model-id",
        LOGICAL_MODEL_ID,
        "--registry-dir",
        str(fixture.registry.registry_dir),
        "--production-root",
        str(fixture.production_root),
        "--dry-run",
    )


def _promote(fixture: PromotionFixture) -> tuple[int, str]:
    return _run_cli(
        "--logical-model-id",
        LOGICAL_MODEL_ID,
        "--registry-dir",
        str(fixture.registry.registry_dir),
        "--production-root",
        str(fixture.production_root),
        "--human-approver",
        "egeme",
        "--accept-residual-watches",
        "--promote",
    )


def _create_active_fixture(
    root: Path,
    *,
    ppo_metadata: dict[str, object] | None = None,
    dqn_metadata: dict[str, object] | None = None,
) -> PromotionFixture:
    checkpoint_dir = root / "models" / "checkpoints" / "candidate"
    checkpoint_dir.mkdir(parents=True)
    joint_checkpoint = checkpoint_dir / "joint_torch_latest.pt"
    ppo_artifact = checkpoint_dir / "ppo_torch_joint_final_balanced_retention_ft_200k.pt"
    dqn_artifact = checkpoint_dir / "dqn_torch_joint_final_balanced_retention_ft_200k.pt"
    _write_joint_checkpoint(joint_checkpoint)
    ppo_artifact.write_bytes(b"ppo-artifact")
    dqn_artifact.write_bytes(b"dqn-artifact")

    eval_output_dir = root / "models" / "eval" / "candidate"
    eval_output_dir.mkdir(parents=True)
    (eval_output_dir / "scenario_summary.json").write_text('{"verdict":"PASS"}', encoding="utf-8")

    baseline_dir = root / "models" / "baselines"
    baseline_dir.mkdir(parents=True)
    baseline_marker = baseline_dir / "baseline.txt"
    baseline_marker.write_text("baseline untouched", encoding="utf-8")

    registry = ModelRegistry(root / "models" / "registry")
    base_metadata = _active_metadata(
        joint_checkpoint=joint_checkpoint,
        eval_output_dir=eval_output_dir,
    )
    ppo_record = registry.register(
        algorithm="ppo",
        path=ppo_artifact,
        agent_role=PPO_AGENT_ROLE,
        status="active",
        metrics={"scenario_pass_rate": 1.0},
        metadata={**base_metadata, **(ppo_metadata or {}), "artifact_kind": "ppo_final"},
    )
    dqn_record = registry.register(
        algorithm="dqn",
        path=dqn_artifact,
        agent_role=DQN_AGENT_ROLE,
        status="active",
        metrics={"scenario_pass_rate": 1.0},
        metadata={**base_metadata, **(dqn_metadata or {}), "artifact_kind": "dqn_final"},
    )
    return PromotionFixture(
        root=root,
        registry=registry,
        production_root=root / "models" / "production",
        baseline_marker=baseline_marker,
        eval_output_dir=eval_output_dir,
        joint_checkpoint=joint_checkpoint,
        ppo_artifact=ppo_artifact,
        dqn_artifact=dqn_artifact,
        ppo_model_id=ppo_record.model_id,
        dqn_model_id=dqn_record.model_id,
    )


def _active_metadata(*, joint_checkpoint: Path, eval_output_dir: Path) -> dict[str, object]:
    return {
        "logical_model_id": LOGICAL_MODEL_ID,
        "env_id": ENV_ID,
        "parent_env_id": PARENT_ENV_ID,
        "joint_checkpoint": str(joint_checkpoint),
        "parent_checkpoint": "models/checkpoints/parent/joint_torch_latest.pt",
        "framework": "torch_joint",
        "runtime_loader": "joint_torch",
        "contract": EXPECTED_MDP_CONTRACT,
        "obs_dim": OBSERVATION_DIM,
        "action_count": DISCRETE_ACTION_COUNT,
        "training_step": 200000,
        "parent_step": 1000000,
        "eval_output_dir": str(eval_output_dir),
        "eval_verdict": "PASS",
        "hard_blocker_status": "zero",
        "production_promotion": False,
        "baseline_update": False,
        "activation_scope": "registry_active_only",
    }


def _rewrite_active_artifact_path(fixture: PromotionFixture, algorithm: str, path: Path) -> None:
    key = f"{algorithm}:{PPO_AGENT_ROLE if algorithm == 'ppo' else DQN_AGENT_ROLE}"
    active = json.loads(fixture.registry.active_path.read_text(encoding="utf-8"))
    active[key] = str(path)
    fixture.registry.active_path.write_text(json.dumps(active), encoding="utf-8")

    lines = []
    for line in fixture.registry.index_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["algorithm"] == algorithm and row["status"] == "active":
            row["path"] = str(path)
        lines.append(json.dumps(row, separators=(",", ":")))
    fixture.registry.index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_joint_checkpoint(path: Path) -> None:
    config = {
        "mdp_contract_version": EXPECTED_MDP_CONTRACT,
        "shared_global_parameters": {
            "environment_id": ENV_ID,
            "observation_dim": OBSERVATION_DIM,
        },
    }
    checkpoint = JointPolicyBundle().build_checkpoint(global_step=200000, config=config)
    checkpoint["artifact_kind"] = "joint_final"
    torch.save(checkpoint, path)


if __name__ == "__main__":
    unittest.main()
