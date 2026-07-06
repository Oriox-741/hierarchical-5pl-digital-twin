from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts import monitoring_report_generator as monitor


class MonitoringReportGeneratorTests(unittest.TestCase):
    def test_clean_synthetic_report_returns_clean_classification(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_health = _write_artifact_health(Path(temp_dir) / "artifact_health.json")

            report = monitor.build_monitoring_report(
                artifact_health=monitor.load_artifact_health_report(artifact_health),
                telemetry_rows=monitor.built_in_synthetic_fixture("clean"),
                source_telemetry_kind="built_in:clean",
                synthetic_data=True,
            )

            self.assertEqual(
                {
                    "report_metadata",
                    "protected_state",
                    "scenario_metrics",
                    "action_metrics",
                    "residual_watches",
                    "alerts",
                    "decision",
                },
                set(report),
            )
            self.assertEqual("clean", report["protected_state"]["status"])
            self.assertEqual("MONITORING_REPORT_CLEAN", report["decision"]["classification"])
            self.assertEqual("joint_torch_v5_prod_hierarchical_v1_1m_20260611", report["report_metadata"]["logical_model_id"])
            self.assertEqual("torch_joint", report["report_metadata"]["runtime_family"])
            self.assertEqual("hierarchical_v1", report["report_metadata"]["dqn_architecture"])
            self.assertEqual("flat_teacher_distillation_v1", report["report_metadata"]["hierarchical_init_method"])
            self.assertEqual(73, report["report_metadata"]["observation_dim"])
            self.assertEqual(5, report["report_metadata"]["continuous_action_dim"])
            self.assertEqual(48, report["report_metadata"]["external_discrete_action_count"])
            self.assertEqual(2, report["report_metadata"]["source_telemetry_rows"])
            self.assertEqual(
                {
                    "route_action_32_concentration",
                    "mixed_action_24_concentration",
                    "top_action_concentration",
                    "mixed_success_route_failure",
                    "secondary_fleet_economics_unvalidated",
                    "reorder_none_economics_unvalidated",
                },
                {watch["watch_id"] for watch in report["residual_watches"]},
            )
            action_rows = {(row["scenario_or_regime"], row["action_id"]): row for row in report["action_metrics"]}
            self.assertEqual(
                "dispatch + shortest + secondary_fleet + none",
                action_rows[("mixed_stress", 24)]["decoded_action"],
            )
            self.assertEqual(
                "dispatch + low_congestion + secondary_fleet + none",
                action_rows[("route_disruption_congestion", 32)]["decoded_action"],
            )
            self.assertFalse([alert for alert in report["alerts"] if alert["level"] in {"warning", "critical"}])

    def test_action_24_concentration_warning_is_nonfatal_without_operational_degradation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_health = _write_artifact_health(Path(temp_dir) / "artifact_health.json")

            report = monitor.build_monitoring_report(
                artifact_health=monitor.load_artifact_health_report(artifact_health),
                telemetry_rows=monitor.built_in_synthetic_fixture("warning_action24"),
                source_telemetry_kind="built_in:warning_action24",
                synthetic_data=True,
            )

            self.assertEqual("MONITORING_REPORT_WARNINGS_ONLY", report["decision"]["classification"])
            watch = _watch(report, "mixed_action_24_concentration")
            self.assertEqual("watch", watch["decision"])
            self.assertGreater(watch["current_rate"], 0.580208 + 0.10)
            self.assertTrue(
                any(alert["alert_id"] == "action_24_concentration" and alert["level"] == "watch" for alert in report["alerts"])
            )

    def test_action_32_concentration_warning_is_nonfatal_without_operational_degradation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_health = _write_artifact_health(Path(temp_dir) / "artifact_health.json")

            report = monitor.build_monitoring_report(
                artifact_health=monitor.load_artifact_health_report(artifact_health),
                telemetry_rows=monitor.built_in_synthetic_fixture("warning_action32"),
                source_telemetry_kind="built_in:warning_action32",
                synthetic_data=True,
            )

            self.assertEqual("MONITORING_REPORT_WARNINGS_ONLY", report["decision"]["classification"])
            watch = _watch(report, "route_action_32_concentration")
            self.assertEqual("watch", watch["decision"])
            self.assertGreater(watch["current_rate"], 0.475174 + 0.10)
            self.assertTrue(
                any(alert["alert_id"] == "action_32_concentration" and alert["level"] == "watch" for alert in report["alerts"])
            )

    def test_action_concentration_plus_operational_degradation_needs_investigation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_health = _write_artifact_health(Path(temp_dir) / "artifact_health.json")
            rows = monitor.built_in_synthetic_fixture("warning_action32")
            rows[1]["route_failure"] = 1

            report = monitor.build_monitoring_report(
                artifact_health=monitor.load_artifact_health_report(artifact_health),
                telemetry_rows=rows,
                source_telemetry_kind="fixture:warning_action32_route_failure",
                synthetic_data=True,
            )

            self.assertEqual("MONITORING_REPORT_NEEDS_INVESTIGATION", report["decision"]["classification"])
            self.assertTrue(
                any(alert["alert_id"] == "action_32_concentration" and alert["level"] == "warning" for alert in report["alerts"])
            )

    def test_no_current_no_unassigned_or_failed_noop_on_watched_actions_escalates(self) -> None:
        for field in ("no_current", "no_unassigned", "failed_noop"):
            with self.subTest(field=field):
                with TemporaryDirectory() as temp_dir:
                    artifact_health = _write_artifact_health(Path(temp_dir) / "artifact_health.json")
                    rows = monitor.built_in_synthetic_fixture("clean")
                    rows[0][field] = 1

                    report = monitor.build_monitoring_report(
                        artifact_health=monitor.load_artifact_health_report(artifact_health),
                        telemetry_rows=rows,
                        source_telemetry_kind=f"fixture:{field}",
                        synthetic_data=True,
                    )

                    self.assertEqual("MONITORING_REPORT_PRODUCTION_RISK_FOUND", report["decision"]["classification"])
                    self.assertTrue(
                        any(
                            alert["level"] == "critical"
                            and alert["action_id"] in (24, 32)
                            and alert["metric"] == field
                            for alert in report["alerts"]
                        )
                    )

    def test_protected_state_drift_escalates_even_when_telemetry_is_clean(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_health = _write_artifact_health(
                Path(temp_dir) / "artifact_health.json",
                final_classification="ARTIFACT_HEALTH_RUNTIME_SMOKE_FAILED",
                runtime_status="FAIL",
            )

            report = monitor.build_monitoring_report(
                artifact_health=monitor.load_artifact_health_report(artifact_health),
                telemetry_rows=monitor.built_in_synthetic_fixture("clean"),
                source_telemetry_kind="built_in:clean",
                synthetic_data=True,
            )

            self.assertEqual("drift", report["protected_state"]["status"])
            self.assertEqual("MONITORING_REPORT_PROTECTED_STATE_DRIFT", report["decision"]["classification"])
            self.assertTrue(any(alert["alert_id"] == "protected_state_drift" for alert in report["alerts"]))

    def test_malformed_telemetry_returns_blocked_by_data_quality(self) -> None:
        with TemporaryDirectory() as temp_dir:
            artifact_health = _write_artifact_health(Path(temp_dir) / "artifact_health.json")
            bad_rows = monitor.built_in_synthetic_fixture("clean")
            bad_rows[0]["decision_steps"] = 0

            report = monitor.build_monitoring_report(
                artifact_health=monitor.load_artifact_health_report(artifact_health),
                telemetry_rows=bad_rows,
                source_telemetry_kind="fixture:bad",
                synthetic_data=True,
            )

            self.assertEqual("MONITORING_REPORT_BLOCKED_BY_DATA_QUALITY", report["decision"]["classification"])
            self.assertTrue(any(alert["alert_id"] == "telemetry_data_quality" for alert in report["alerts"]))

    def test_json_and_csv_fixture_inputs_produce_same_clean_classification(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = root / "telemetry.json"
            csv_path = root / "telemetry.csv"
            rows = monitor.built_in_synthetic_fixture("clean")
            json_path.write_text(json.dumps(rows), encoding="utf-8")
            _write_csv(csv_path, rows)

            self.assertEqual(rows, monitor.load_telemetry_json(json_path))
            self.assertEqual(rows, monitor.load_telemetry_csv(csv_path))

    def test_cli_writes_only_requested_output_json_in_fixture_mode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_health = _write_artifact_health(root / "artifact_health.json")
            telemetry_path = root / "telemetry.json"
            telemetry_path.write_text(json.dumps(monitor.built_in_synthetic_fixture("clean")), encoding="utf-8")
            output = root / "report.json"
            before = _relative_files(root)

            exit_code = monitor.main(
                [
                    "--artifact-health",
                    str(artifact_health),
                    "--telemetry-json",
                    str(telemetry_path),
                    "--output",
                    str(output),
                ]
            )

            after = _relative_files(root)
            self.assertEqual(0, exit_code)
            self.assertEqual(before | {Path("report.json")}, after)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("MONITORING_REPORT_CLEAN", report["decision"]["classification"])


def _write_artifact_health(
    path: Path,
    *,
    final_classification: str = "ARTIFACT_HEALTH_CLEAN",
    runtime_status: str = "PASS",
) -> Path:
    artifact_health = {
        "final_classification": final_classification,
        "expected_production_identity": {
            "logical_model_id": "joint_torch_v5_prod_hierarchical_v1_1m_20260611",
            "active_ppo_id": "17ba1d28-0054-4f7c-ae9a-34cd305ebb89",
            "active_dqn_id": "f87e10d6-479f-44fc-99d1-6925bc9cb346",
            "dqn_architecture": "hierarchical_v1",
            "hierarchical_init_method": "flat_teacher_distillation_v1",
            "contract": "physical_reality_v5_route_candidate_visibility",
            "observation_dim": 73,
            "action_count": 48,
        },
        "active_registry_checks": {"status": "PASS", "sha256": "active_hash"},
        "models_jsonl_checks": {"status": "PASS", "sha256": "models_hash"},
        "production_manifest_checks": {"status": "PASS", "sha256": "manifest_hash"},
        "production_file_hash_checks": {
            "status": "PASS",
            "actual_sha256": {
                "joint_torch_latest.pt": "joint_hash",
                "production_manifest.json": "manifest_hash",
            },
        },
        "protected_path_profile_checks": {"status": "PASS"},
        "process_scan_result": {"status": "PASS"},
        "runtime_smoke_result": {
            "status": runtime_status,
            "checkpoint_path": "models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt",
        },
    }
    path.write_text(json.dumps(artifact_health, indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _watch(report: dict, watch_id: str) -> dict:
    for watch in report["residual_watches"]:
        if watch["watch_id"] == watch_id:
            return watch
    raise AssertionError(f"watch not found: {watch_id}")


def _relative_files(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


if __name__ == "__main__":
    unittest.main()
