from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from scripts import production_artifact_health_report as health


class ProductionArtifactHealthReportTests(unittest.TestCase):
    def test_active_registry_happy_path_passes(self) -> None:
        with _fixture() as fixture:
            report = health.build_artifact_health_report(
                fixture.config,
                process_entries=[],
                run_runtime_smoke=False,
            )

            self.assertEqual("PASS", report["active_registry_checks"]["status"])
            self.assertEqual("PASS", report["models_jsonl_checks"]["status"])
            self.assertEqual("PASS", report["production_manifest_checks"]["status"])
            self.assertEqual("PASS", report["production_file_hash_checks"]["status"])
            self.assertEqual("PASS", report["protected_path_profile_checks"]["status"])
            self.assertEqual("SKIPPED", report["runtime_smoke_result"]["status"])
            self.assertEqual("ARTIFACT_HEALTH_CLEAN", report["final_classification"])

    def test_active_registry_mismatch_is_detected(self) -> None:
        with _fixture() as fixture:
            _write_json(
                fixture.active_models,
                {
                    "ppo:continuous_control": fixture.expected_ppo_path,
                    "dqn:tactical_dispatch": "models\\checkpoints\\wrong_dqn.pt",
                },
            )

            report = health.build_artifact_health_report(
                fixture.config,
                process_entries=[],
                run_runtime_smoke=False,
            )

            self.assertEqual("FAIL", report["active_registry_checks"]["status"])
            self.assertEqual(
                "ARTIFACT_HEALTH_ACTIVE_REGISTRY_MISMATCH",
                report["final_classification"],
            )

    def test_models_jsonl_candidate_and_active_row_checks_work(self) -> None:
        with _fixture() as fixture:
            rows = _read_jsonl(fixture.models_jsonl)
            rows = [row for row in rows if row["model_id"] != fixture.identity.active_dqn_id]
            _write_jsonl(fixture.models_jsonl, rows)

            report = health.build_artifact_health_report(
                fixture.config,
                process_entries=[],
                run_runtime_smoke=False,
            )

            self.assertEqual("FAIL", report["models_jsonl_checks"]["status"])
            self.assertTrue(report["models_jsonl_checks"]["missing_rows"])
            self.assertEqual(
                "ARTIFACT_HEALTH_MODELS_JSONL_MISMATCH",
                report["final_classification"],
            )

    def test_manifest_mismatch_is_detected(self) -> None:
        with _fixture() as fixture:
            manifest = json.loads(fixture.production_manifest.read_text(encoding="utf-8"))
            manifest["hierarchical_init_method"] = "wrong_init"
            _write_json(fixture.production_manifest, manifest)

            report = health.build_artifact_health_report(
                fixture.config,
                process_entries=[],
                run_runtime_smoke=False,
            )

            self.assertEqual("FAIL", report["production_manifest_checks"]["status"])
            self.assertEqual(
                "ARTIFACT_HEALTH_PRODUCTION_MANIFEST_MISMATCH",
                report["final_classification"],
            )

    def test_file_hash_mismatch_is_detected(self) -> None:
        with _fixture() as fixture:
            (fixture.production_dir / health.EXPECTED_PPO_FILENAME).write_text("changed", encoding="utf-8")

            report = health.build_artifact_health_report(
                fixture.config,
                process_entries=[],
                run_runtime_smoke=False,
            )

            self.assertEqual("FAIL", report["production_file_hash_checks"]["status"])
            self.assertEqual(
                "ARTIFACT_HEALTH_PRODUCTION_HASH_MISMATCH",
                report["final_classification"],
            )

    def test_protected_path_profile_mismatch_is_detected(self) -> None:
        with _fixture() as fixture:
            (fixture.protected_dirs["db"] / "unexpected.sqlite").write_text("drift", encoding="utf-8")

            report = health.build_artifact_health_report(
                fixture.config,
                process_entries=[],
                run_runtime_smoke=False,
            )

            self.assertEqual("FAIL", report["protected_path_profile_checks"]["status"])
            self.assertEqual(
                "ARTIFACT_HEALTH_PROTECTED_PATH_DRIFT",
                report["final_classification"],
            )

    def test_forbidden_process_result_influences_classification(self) -> None:
        with _fixture() as fixture:
            report = health.build_artifact_health_report(
                fixture.config,
                process_entries=[
                    health.ProcessEntry(
                        pid=123,
                        name="python.exe",
                        command_line="python -m src.learn.train_joint_torch",
                    )
                ],
                run_runtime_smoke=False,
            )

            self.assertEqual("FAIL", report["process_scan_result"]["status"])
            self.assertEqual(
                "ARTIFACT_HEALTH_FORBIDDEN_PROCESS_FOUND",
                report["final_classification"],
            )

    def test_cli_writes_only_requested_output_json_in_fixture_mode(self) -> None:
        with _fixture() as fixture:
            output = fixture.root / "report.json"
            before_files = _relative_files(fixture.root)
            before_hashes = {
                fixture.active_models: _sha256(fixture.active_models),
                fixture.models_jsonl: _sha256(fixture.models_jsonl),
                fixture.production_manifest: _sha256(fixture.production_manifest),
            }

            exit_code = health.main(
                [
                    "--active-models",
                    str(fixture.active_models),
                    "--models-jsonl",
                    str(fixture.models_jsonl),
                    "--production-dir",
                    str(fixture.production_dir),
                    "--production-manifest",
                    str(fixture.production_manifest),
                    "--dashboard",
                    str(fixture.dashboard),
                    "--output",
                    str(output),
                    "--skip-runtime-smoke",
                    "--expected-active-ppo-id",
                    fixture.identity.active_ppo_id,
                    "--expected-active-dqn-id",
                    fixture.identity.active_dqn_id,
                    "--expected-candidate-ppo-id",
                    fixture.identity.candidate_ppo_id,
                    "--expected-candidate-dqn-id",
                    fixture.identity.candidate_dqn_id,
                    "--expected-active-ppo-path",
                    fixture.expected_ppo_path,
                    "--expected-active-dqn-path",
                    fixture.expected_dqn_path,
                    "--protected-profile",
                    fixture.protected_profile_arg("models/baselines"),
                    "--protected-profile",
                    fixture.protected_profile_arg("db"),
                    "--expected-active-models-sha256",
                    fixture.expected_active_models_hash,
                    "--expected-models-jsonl-sha256",
                    fixture.expected_models_jsonl_hash,
                    "--expected-production-manifest-sha256",
                    fixture.expected_manifest_hash,
                    "--expected-production-dir-files",
                    ",".join(fixture.expected_production_files),
                    "--expected-production-file-sha256",
                    f"{health.EXPECTED_JOINT_FILENAME}={fixture.expected_hashes[health.EXPECTED_JOINT_FILENAME]}",
                    "--expected-production-file-sha256",
                    f"{health.EXPECTED_PPO_FILENAME}={fixture.expected_hashes[health.EXPECTED_PPO_FILENAME]}",
                    "--expected-production-file-sha256",
                    f"{health.EXPECTED_DQN_FILENAME}={fixture.expected_hashes[health.EXPECTED_DQN_FILENAME]}",
                    "--expected-production-file-sha256",
                    f"production_manifest.json={fixture.expected_hashes['production_manifest.json']}",
                ]
            )

            after_files = _relative_files(fixture.root)
            self.assertEqual(0, exit_code)
            self.assertEqual(before_files | {Path("report.json")}, after_files)
            self.assertEqual("ARTIFACT_HEALTH_CLEAN", json.loads(output.read_text(encoding="utf-8"))["final_classification"])
            for path, before_hash in before_hashes.items():
                self.assertEqual(before_hash, _sha256(path))

    def test_json_reader_accepts_utf8_bom(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text('\ufeff{"logical_model_id":"ok"}\n', encoding="utf-8")

            self.assertEqual({"logical_model_id": "ok"}, health._read_json(path))

    def test_repo_root_is_added_for_direct_script_runtime_imports(self) -> None:
        original_sys_path = list(sys.path)
        try:
            repo_root = health._repo_root()
            sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != repo_root]

            health._ensure_repo_root_on_sys_path()

            self.assertIn(str(repo_root), sys.path)
        finally:
            sys.path = original_sys_path

    def test_exact_resume_capable_is_read_from_resume_state_metadata(self) -> None:
        checkpoint = {"resume_state": {"exact_resume_capable": True}}

        self.assertTrue(health._exact_resume_capable(checkpoint))


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.registry = root / "registry"
        self.production_dir = root / "production" / health.EXPECTED_LOGICAL_MODEL_ID
        self.active_models = self.registry / "active_models.json"
        self.models_jsonl = self.registry / "models.jsonl"
        self.production_manifest = self.production_dir / "production_manifest.json"
        self.dashboard = root / "docs" / "00_PROJECT_DASHBOARD.md"
        self.protected_dirs = {
            "models/baselines": root / "protected" / "models" / "baselines",
            "db": root / "protected" / "db",
        }
        self.expected_ppo_path = "models\\checkpoints\\joint_torch_v5_prod_hierarchical_v1_1m_20260611\\ppo.pt"
        self.expected_dqn_path = "models\\checkpoints\\joint_torch_v5_prod_hierarchical_v1_1m_20260611\\dqn.pt"
        self.identity = health.ExpectedProductionIdentity(
            logical_model_id=health.EXPECTED_LOGICAL_MODEL_ID,
            active_ppo_id="active-ppo",
            active_dqn_id="active-dqn",
            candidate_ppo_id="candidate-ppo",
            candidate_dqn_id="candidate-dqn",
            active_ppo_path=self.expected_ppo_path,
            active_dqn_path=self.expected_dqn_path,
            dqn_architecture="hierarchical_v1",
            hierarchical_init_method="flat_teacher_distillation_v1",
            contract="physical_reality_v5_route_candidate_visibility",
            observation_dim=73,
            action_count=48,
            training_step=1_000_000,
        )

    def write(self) -> "_Fixture":
        self.registry.mkdir(parents=True)
        self.production_dir.mkdir(parents=True)
        (self.dashboard.parent).mkdir(parents=True)
        for directory in self.protected_dirs.values():
            directory.mkdir(parents=True)
            (directory / "sentinel.txt").write_text("abc", encoding="utf-8")

        for filename, contents in {
            health.EXPECTED_JOINT_FILENAME: "joint",
            health.EXPECTED_PPO_FILENAME: "ppo",
            health.EXPECTED_DQN_FILENAME: "dqn",
        }.items():
            (self.production_dir / filename).write_text(contents, encoding="utf-8")

        manifest = {
            "logical_model_id": self.identity.logical_model_id,
            "active_ppo_id": self.identity.active_ppo_id,
            "active_dqn_id": self.identity.active_dqn_id,
            "candidate_ppo_id": self.identity.candidate_ppo_id,
            "candidate_dqn_id": self.identity.candidate_dqn_id,
            "source_paths": {
                health.EXPECTED_JOINT_FILENAME: "source_joint",
                health.EXPECTED_PPO_FILENAME: "source_ppo",
                health.EXPECTED_DQN_FILENAME: "source_dqn",
            },
            "production_paths": {
                health.EXPECTED_JOINT_FILENAME: str(self.production_dir / health.EXPECTED_JOINT_FILENAME),
                health.EXPECTED_PPO_FILENAME: str(self.production_dir / health.EXPECTED_PPO_FILENAME),
                health.EXPECTED_DQN_FILENAME: str(self.production_dir / health.EXPECTED_DQN_FILENAME),
            },
            "sha256": {"source": {}, "production": {}},
            "dqn_architecture": self.identity.dqn_architecture,
            "hierarchical_init_method": self.identity.hierarchical_init_method,
            "contract": self.identity.contract,
            "observation_dim": self.identity.observation_dim,
            "action_dim": self.identity.action_count,
            "training_step": self.identity.training_step,
            "eval_verdict": "PASS",
            "hard_blocker_status": "zero",
            "long_run_gate_verdict": "PASS",
            "residual_watches_accepted": True,
            "baseline_update": False,
            "db_mutation": False,
            "registry_mutation": False,
        }
        hashes = {
            filename: _sha256(self.production_dir / filename)
            for filename in (
                health.EXPECTED_JOINT_FILENAME,
                health.EXPECTED_PPO_FILENAME,
                health.EXPECTED_DQN_FILENAME,
            )
        }
        manifest["sha256"]["source"] = dict(hashes)
        manifest["sha256"]["production"] = dict(hashes)
        _write_json(self.production_manifest, manifest)
        hashes["production_manifest.json"] = _sha256(self.production_manifest)
        self.expected_hashes = hashes

        _write_json(
            self.active_models,
            {
                "ppo:continuous_control": self.expected_ppo_path,
                "dqn:tactical_dispatch": self.expected_dqn_path,
            },
        )
        self.expected_active_models_hash = _sha256(self.active_models)

        rows = [
            _row(self.identity.candidate_ppo_id, "ppo", self.expected_ppo_path, "candidate", self.identity, "ppo_final", "continuous_control"),
            _row(self.identity.candidate_dqn_id, "dqn", self.expected_dqn_path, "candidate", self.identity, "dqn_final", "tactical_dispatch"),
            _row(self.identity.active_ppo_id, "ppo", self.expected_ppo_path, "active", self.identity, "ppo_final", "continuous_control"),
            _row(self.identity.active_dqn_id, "dqn", self.expected_dqn_path, "active", self.identity, "dqn_final", "tactical_dispatch"),
            {"model_id": "old-active-ppo", "algorithm": "ppo", "status": "active", "metadata": {"logical_model_id": "old-prod"}},
            {"model_id": "old-active-dqn", "algorithm": "dqn", "status": "active", "metadata": {"logical_model_id": "old-prod"}},
        ]
        _write_jsonl(self.models_jsonl, rows)
        self.expected_models_jsonl_hash = _sha256(self.models_jsonl)
        self.expected_manifest_hash = _sha256(self.production_manifest)
        self.expected_production_files = tuple(sorted(hashes))

        self.dashboard.write_text(
            "\n".join(
                [
                    health.EXPECTED_LOGICAL_MODEL_ID,
                    str(self.production_dir / health.EXPECTED_JOINT_FILENAME),
                    "docs/releases/20260611_hierarchical_v1_1m_production_handoff",
                    "docs/runbooks/20260611_hierarchical_v1_production_monitoring_runbook",
                    "docs/runs/20260611_hierarchical_v1_equal_budget_residual_watch_eval",
                    "docs/runbooks/20260611_hierarchical_v1_monitoring_kpi_dictionary",
                    "docs/runbooks/20260611_codex_model_lifecycle_governance_runbook",
                ]
            ),
            encoding="utf-8",
        )

        self.config = health.HealthReportConfig(
            active_models_path=self.active_models,
            models_jsonl_path=self.models_jsonl,
            production_dir=self.production_dir,
            production_manifest_path=self.production_manifest,
            dashboard_path=self.dashboard,
            output_path=None,
            expected_identity=self.identity,
            expected_active_models_sha256=self.expected_active_models_hash,
            expected_models_jsonl_sha256=self.expected_models_jsonl_hash,
            expected_production_manifest_sha256=self.expected_manifest_hash,
            expected_production_file_sha256=self.expected_hashes,
            expected_production_dir_files=set(self.expected_production_files),
            expected_protected_profiles={
                key: health.PathProfile(path=path, file_count=1, total_bytes=3)
                for key, path in self.protected_dirs.items()
            },
        )
        return self

    def protected_profile_arg(self, label: str) -> str:
        profile = self.config.expected_protected_profiles[label]
        return f"{label}|{profile.path}|{profile.file_count}|{profile.total_bytes}"


def _fixture():
    temp_dir = TemporaryDirectory()
    root = Path(temp_dir.name)
    fixture = _Fixture(root).write()

    class _FixtureContext:
        def __enter__(self) -> _Fixture:
            return fixture

        def __exit__(self, exc_type, exc, tb) -> None:
            temp_dir.cleanup()

    return _FixtureContext()


def _row(
    model_id: str,
    algorithm: str,
    path: str,
    status: str,
    identity: health.ExpectedProductionIdentity,
    artifact_kind: str,
    agent_role: str,
) -> dict:
    return {
        "model_id": model_id,
        "algorithm": algorithm,
        "path": path,
        "status": status,
        "metadata": {
            "logical_model_id": identity.logical_model_id,
            "framework": "torch_joint",
            "runtime_loader": "joint_torch",
            "dqn_architecture": identity.dqn_architecture,
            "hierarchical_init_method": identity.hierarchical_init_method,
            "joint_checkpoint": "joint_torch_latest.pt",
            "contract": identity.contract,
            "obs_dim": identity.observation_dim,
            "action_count": identity.action_count,
            "external_discrete_action_count": identity.action_count,
            "training_step": identity.training_step,
            "exact_resume_capable": True,
            "eval_verdict": "PASS",
            "hard_blocker_status": "zero",
            "long_run_gate_verdict": "PASS",
            "production_ready": False,
            "baseline_update": False,
            "artifact_kind": artifact_kind,
            "agent_role": agent_role,
        },
    }


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _relative_files(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


if __name__ == "__main__":
    unittest.main()
