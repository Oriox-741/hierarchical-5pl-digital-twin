from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from scripts import run_production_ops_bundle as bundle


EXPECTED_TOP_LEVEL_KEYS = {
    "bundle_metadata",
    "artifact_health",
    "monitoring_report",
    "company_data_schema",
    "protected_state",
    "process_scan",
    "decision",
}


class ProductionOpsBundleTests(unittest.TestCase):
    def test_all_clean_bundle_is_clean(self) -> None:
        report = bundle.build_ops_bundle_report(
            artifact_health=_artifact_health(),
            monitoring_report=_monitoring_report(),
            company_data_schema=_company_schema_report(),
            process_entries=[],
        )

        self.assertEqual(EXPECTED_TOP_LEVEL_KEYS, set(report))
        self.assertEqual("production_ops_bundle_v1", report["bundle_metadata"]["generator_version"])
        self.assertTrue(report["bundle_metadata"]["synthetic_only"])
        self.assertFalse(report["bundle_metadata"]["private_data_ingested"])
        self.assertFalse(report["bundle_metadata"]["training_run"])
        self.assertFalse(report["bundle_metadata"]["offline_eval_run"])
        self.assertFalse(report["bundle_metadata"]["long_run_gate_run"])
        self.assertEqual("ARTIFACT_HEALTH_CLEAN", report["artifact_health"]["classification"])
        self.assertEqual("MONITORING_REPORT_CLEAN", report["monitoring_report"]["classification"])
        self.assertEqual("COMPANY_DATA_SCHEMA_READY", report["company_data_schema"]["classification"])
        self.assertEqual("clean", report["protected_state"]["artifact_health_report_status"])
        self.assertFalse(report["process_scan"]["train_process_detected"])
        self.assertEqual("OPS_BUNDLE_CLEAN", report["decision"]["classification"])

    def test_artifact_health_drift_blocks_bundle(self) -> None:
        report = bundle.build_ops_bundle_report(
            artifact_health=_artifact_health("ARTIFACT_HEALTH_ACTIVE_REGISTRY_MISMATCH"),
            monitoring_report=_monitoring_report("MONITORING_REPORT_PRODUCTION_RISK_FOUND"),
            company_data_schema=_company_schema_report("COMPANY_DATA_SCHEMA_BLOCKED"),
            process_entries=[],
        )

        self.assertEqual("OPS_BUNDLE_PROTECTED_STATE_DRIFT", report["decision"]["classification"])
        self.assertTrue(any("artifact-health" in reason for reason in report["decision"]["reasons"]))

    def test_monitoring_production_risk_escalates_after_clean_artifact_health(self) -> None:
        report = bundle.build_ops_bundle_report(
            artifact_health=_artifact_health(),
            monitoring_report=_monitoring_report("MONITORING_REPORT_PRODUCTION_RISK_FOUND"),
            company_data_schema=_company_schema_report("COMPANY_DATA_SCHEMA_BLOCKED"),
            process_entries=[],
        )

        self.assertEqual("OPS_BUNDLE_PRODUCTION_RISK_FOUND", report["decision"]["classification"])

    def test_company_schema_blocked_escalates_when_prior_inputs_are_clean(self) -> None:
        report = bundle.build_ops_bundle_report(
            artifact_health=_artifact_health(),
            monitoring_report=_monitoring_report(),
            company_data_schema=_company_schema_report("COMPANY_DATA_SCHEMA_BLOCKED"),
            process_entries=[],
        )

        self.assertEqual("OPS_BUNDLE_DATA_SCHEMA_BLOCKED", report["decision"]["classification"])

    def test_warning_or_investigation_inputs_are_warnings_only(self) -> None:
        for monitoring_classification, company_classification in (
            ("MONITORING_REPORT_WARNINGS_ONLY", "COMPANY_DATA_SCHEMA_READY"),
            ("MONITORING_REPORT_NEEDS_INVESTIGATION", "COMPANY_DATA_SCHEMA_READY"),
            ("MONITORING_REPORT_CLEAN", "COMPANY_DATA_SCHEMA_WARNINGS_ONLY"),
            ("MONITORING_REPORT_CLEAN", "COMPANY_DATA_SCHEMA_NEEDS_FIXES"),
        ):
            with self.subTest(
                monitoring_classification=monitoring_classification,
                company_classification=company_classification,
            ):
                report = bundle.build_ops_bundle_report(
                    artifact_health=_artifact_health(),
                    monitoring_report=_monitoring_report(monitoring_classification),
                    company_data_schema=_company_schema_report(company_classification),
                    process_entries=[],
                )

                self.assertEqual("OPS_BUNDLE_WARNINGS_ONLY", report["decision"]["classification"])

    def test_malformed_artifact_health_blocks_as_protected_state_drift(self) -> None:
        report = bundle.build_ops_bundle_report(
            artifact_health={"report_metadata": {}},
            monitoring_report=_monitoring_report("MONITORING_REPORT_PRODUCTION_RISK_FOUND"),
            company_data_schema=_company_schema_report("COMPANY_DATA_SCHEMA_BLOCKED"),
            process_entries=[],
        )

        self.assertEqual("OPS_BUNDLE_PROTECTED_STATE_DRIFT", report["decision"]["classification"])

    def test_malformed_monitoring_report_blocks_as_production_risk(self) -> None:
        report = bundle.build_ops_bundle_report(
            artifact_health=_artifact_health(),
            monitoring_report={"report_metadata": {}},
            company_data_schema=_company_schema_report("COMPANY_DATA_SCHEMA_BLOCKED"),
            process_entries=[],
        )

        self.assertEqual("OPS_BUNDLE_PRODUCTION_RISK_FOUND", report["decision"]["classification"])

    def test_malformed_company_schema_report_blocks_as_data_schema_blocked(self) -> None:
        report = bundle.build_ops_bundle_report(
            artifact_health=_artifact_health(),
            monitoring_report=_monitoring_report(),
            company_data_schema={"report_metadata": {}},
            process_entries=[],
        )

        self.assertEqual("OPS_BUNDLE_DATA_SCHEMA_BLOCKED", report["decision"]["classification"])

    def test_process_scan_flags_forbidden_process_patterns_without_terminating(self) -> None:
        report = bundle.build_ops_bundle_report(
            artifact_health=_artifact_health(),
            monitoring_report=_monitoring_report(),
            company_data_schema=_company_schema_report(),
            process_entries=[
                bundle.ProcessEntry(pid=123, name="python.exe", command_line="python -m src.learn.train_joint_torch"),
                bundle.ProcessEntry(pid=456, name="aws.exe", command_line="aws s3 cp s3://bucket/private_extract.csv"),
            ],
        )

        self.assertTrue(report["process_scan"]["train_process_detected"])
        self.assertTrue(report["process_scan"]["aws_download_process_detected"])
        self.assertTrue(report["process_scan"]["private_data_process_detected"])

    def test_cli_rejects_protected_output_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            protected = root / "models" / "eval" / "ops.json"

            with self.assertRaisesRegex(ValueError, "protected output path"):
                bundle.run_bundle(
                    protected,
                    repo_root=root,
                    command_runner=_fake_command_runner,
                    process_entries=[],
                )

            self.assertFalse(protected.exists())

    def test_cli_writes_only_requested_output_and_removes_temp_intermediates(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "reports" / "ops" / "ops_bundle.json"
            before = _relative_files(root)
            calls: list[list[str]] = []

            report = bundle.run_bundle(
                output,
                repo_root=root,
                command_runner=lambda command: _fake_command_runner(command, calls),
                process_entries=[],
            )

            after = _relative_files(root)
            self.assertEqual(before | {Path("reports/ops/ops_bundle.json")}, after)
            self.assertEqual("OPS_BUNDLE_CLEAN", report["decision"]["classification"])
            self.assertEqual(report, json.loads(output.read_text(encoding="utf-8")))
            self.assertEqual(3, len(calls))
            for command in calls:
                self.assertTrue(any("scripts/" in part.replace("\\", "/") for part in command))
            self.assertFalse(any("artifact_health.json" in str(path) for path in after))
            self.assertFalse(any("monitoring_report.json" in str(path) for path in after))
            self.assertFalse(any("company_schema_report.json" in str(path) for path in after))

    def test_protected_files_are_not_mutated_by_cli_wrapper(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            protected_paths = [
                root / "models" / "registry" / "active_models.json",
                root / "models" / "registry" / "models.jsonl",
                root / "models" / "production" / "manifest.json",
                root / "models" / "baselines" / "baseline.json",
                root / "db" / "state.sqlite",
                root / "models" / "checkpoints" / "checkpoint.pt",
                root / "models" / "eval" / "summary.json",
            ]
            for path in protected_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"protected:{path.name}", encoding="utf-8")
            before = {path: path.read_text(encoding="utf-8") for path in protected_paths}

            bundle.run_bundle(
                root / "reports" / "ops" / "ops_bundle.json",
                repo_root=root,
                command_runner=_fake_command_runner,
                process_entries=[],
            )

            after = {path: path.read_text(encoding="utf-8") for path in protected_paths}
            self.assertEqual(before, after)

    def test_repo_root_is_added_for_direct_script_runtime_imports(self) -> None:
        original_sys_path = list(sys.path)
        try:
            repo_root = bundle._repo_root()
            sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != repo_root]

            bundle._ensure_repo_root_on_path(repo_root)

            self.assertEqual(str(repo_root), sys.path[0])
        finally:
            sys.path = original_sys_path


def _artifact_health(classification: str = "ARTIFACT_HEALTH_CLEAN") -> dict:
    return {
        "final_classification": classification,
        "report_metadata": {
            "generated_at_utc": "2026-06-11T21:16:20+00:00",
            "script": "scripts/production_artifact_health_report.py",
        },
        "active_registry_checks": {"status": "PASS"},
        "production_manifest_checks": {"status": "PASS"},
        "runtime_smoke_result": {"status": "PASS"},
        "dashboard_consistency_checks": {"status": "PASS"},
        "process_scan_result": {"status": "PASS", "matches": []},
        "protected_path_profile_checks": {"status": "PASS"},
    }


def _monitoring_report(classification: str = "MONITORING_REPORT_CLEAN") -> dict:
    return {
        "report_metadata": {
            "report_timestamp_utc": "2026-06-11T21:49:01+00:00",
            "logical_model_id": "joint_torch_v5_prod_hierarchical_v1_1m_20260611",
            "runtime_family": "torch_joint",
            "dqn_architecture": "hierarchical_v1",
            "hierarchical_init_method": "flat_teacher_distillation_v1",
            "synthetic_data": True,
        },
        "decision": {"classification": classification},
        "alerts": [],
        "residual_watches": [{"watch_id": "route_action_32_concentration"}],
        "action_metrics": [{"action_id": 24}, {"action_id": 32}],
        "scenario_metrics": [],
        "protected_state": {"status": "clean"},
    }


def _company_schema_report(classification: str = "COMPANY_DATA_SCHEMA_READY") -> dict:
    return {
        "report_metadata": {
            "report_timestamp_utc": "2026-06-11T22:58:19+00:00",
            "synthetic_data": True,
            "table_count": 7,
        },
        "decision": {"classification": classification},
        "table_summaries": [{"table": "orders"} for _ in range(7)],
        "data_quality_checks": [{"status": "pass"} for _ in range(39)],
        "join_key_checks": [{"status": "pass"} for _ in range(9)],
        "warnings": [],
    }


def _fake_command_runner(command: list[str], calls: list[list[str]] | None = None) -> object:
    if calls is not None:
        calls.append(command)
    output = Path(command[command.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    command_text = " ".join(command).replace("\\", "/")
    if "production_artifact_health_report.py" in command_text:
        output.write_text(json.dumps(_artifact_health()), encoding="utf-8")
    elif "monitoring_report_generator.py" in command_text:
        output.write_text(json.dumps(_monitoring_report()), encoding="utf-8")
    elif "company_data_intake_validator.py" in command_text:
        output.write_text(json.dumps(_company_schema_report()), encoding="utf-8")
    else:
        raise AssertionError(f"unexpected command: {command}")
    return _CompletedProcess()


def _relative_files(root: Path) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
    }


class _CompletedProcess:
    returncode = 0
    stdout = ""
    stderr = ""


if __name__ == "__main__":
    unittest.main()
