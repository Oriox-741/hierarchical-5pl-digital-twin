from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any, Sequence


GENERATOR_VERSION = "monitoring_report_generator_v1"

DEFAULT_ARTIFACT_HEALTH_PATH = Path("reports/artifact_health/20260611_hierarchical_v1_artifact_health_report.json")
DEFAULT_LOGICAL_MODEL_ID = "joint_torch_v5_prod_hierarchical_v1_1m_20260611"
DEFAULT_RUNTIME_FAMILY = "torch_joint"
DEFAULT_DQN_ARCHITECTURE = "hierarchical_v1"
DEFAULT_HIERARCHICAL_INIT_METHOD = "flat_teacher_distillation_v1"
DEFAULT_CONTRACT = "physical_reality_v5_route_candidate_visibility"
DEFAULT_OBSERVATION_DIM = 73
DEFAULT_CONTINUOUS_ACTION_DIM = 5
DEFAULT_EXTERNAL_DISCRETE_ACTION_COUNT = 48
DEFAULT_PRODUCTION_CHECKPOINT = (
    "models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt"
)
DEFAULT_ACTIVE_PPO_ID = "17ba1d28-0054-4f7c-ae9a-34cd305ebb89"
DEFAULT_ACTIVE_DQN_ID = "f87e10d6-479f-44fc-99d1-6925bc9cb346"

ACTION_24_BASELINE_RATE = 0.580208
ACTION_32_BASELINE_RATE = 0.475174
MATERIAL_CONCENTRATION_MARGIN = 0.10

CLASSIFICATION_CLEAN = "MONITORING_REPORT_CLEAN"
CLASSIFICATION_WARNINGS = "MONITORING_REPORT_WARNINGS_ONLY"
CLASSIFICATION_INVESTIGATION = "MONITORING_REPORT_NEEDS_INVESTIGATION"
CLASSIFICATION_RISK = "MONITORING_REPORT_PRODUCTION_RISK_FOUND"
CLASSIFICATION_PROTECTED_DRIFT = "MONITORING_REPORT_PROTECTED_STATE_DRIFT"
CLASSIFICATION_DATA_QUALITY = "MONITORING_REPORT_BLOCKED_BY_DATA_QUALITY"

TELEMETRY_FIELDS = (
    "window_start_utc",
    "window_end_utc",
    "scenario_or_regime",
    "decision_steps",
    "orders",
    "served",
    "late_orders",
    "lateness_minutes",
    "lateness_p95",
    "delivered",
    "dispatch_attempts",
    "dispatch_successes",
    "action_id",
    "action_attempts",
    "no_current",
    "no_unassigned",
    "failed_noop",
    "route_failure",
    "no_vehicle",
    "already_assigned",
    "secondary_fleet_attempts",
    "reorder_none_decisions",
)
NUMERIC_FIELDS = tuple(field for field in TELEMETRY_FIELDS if field not in {"window_start_utc", "window_end_utc", "scenario_or_regime"})
INTEGER_FIELDS = (
    "decision_steps",
    "orders",
    "served",
    "late_orders",
    "delivered",
    "dispatch_attempts",
    "dispatch_successes",
    "action_id",
    "action_attempts",
    "no_current",
    "no_unassigned",
    "failed_noop",
    "route_failure",
    "no_vehicle",
    "already_assigned",
    "secondary_fleet_attempts",
    "reorder_none_decisions",
)
POSITIVE_DENOMINATORS = ("decision_steps", "orders", "dispatch_attempts")

WATCHED_ACTIONS = {24, 32}


@dataclass(frozen=True, slots=True)
class ValidationResult:
    rows: list[dict[str, Any]]
    issues: list[str]


def load_artifact_health_report(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_telemetry_json(path: Path | str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("telemetry JSON must be a list of row objects.")
    return [_coerce_telemetry_row(row) for row in payload]


def load_telemetry_csv(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [_coerce_telemetry_row(row) for row in csv.DictReader(handle)]


def built_in_synthetic_fixture(name: str) -> list[dict[str, Any]]:
    clean_rows = [
        _telemetry_row(
            scenario_or_regime="mixed_stress",
            action_id=24,
            decision_steps=100,
            orders=100,
            served=96,
            late_orders=4,
            lateness_minutes=8.0,
            lateness_p95=4.0,
            delivered=96,
            dispatch_attempts=80,
            dispatch_successes=78,
            action_attempts=20,
            secondary_fleet_attempts=20,
            reorder_none_decisions=20,
        ),
        _telemetry_row(
            scenario_or_regime="route_disruption_congestion",
            action_id=32,
            decision_steps=100,
            orders=100,
            served=95,
            late_orders=5,
            lateness_minutes=10.0,
            lateness_p95=5.0,
            delivered=95,
            dispatch_attempts=82,
            dispatch_successes=80,
            action_attempts=20,
            secondary_fleet_attempts=20,
            reorder_none_decisions=20,
        ),
    ]
    if name == "clean":
        return [dict(row) for row in clean_rows]
    if name == "warning_action24":
        rows = [dict(row) for row in clean_rows]
        rows[0]["action_attempts"] = 75
        rows[0]["secondary_fleet_attempts"] = 75
        rows[0]["reorder_none_decisions"] = 75
        return rows
    if name == "warning_action32":
        rows = [dict(row) for row in clean_rows]
        rows[1]["action_attempts"] = 70
        rows[1]["secondary_fleet_attempts"] = 70
        rows[1]["reorder_none_decisions"] = 70
        return rows
    if name == "critical_no_current":
        rows = [dict(row) for row in clean_rows]
        rows[0]["no_current"] = 1
        return rows
    raise ValueError(f"unknown built-in synthetic fixture: {name}")


def build_monitoring_report(
    *,
    artifact_health: dict[str, Any],
    telemetry_rows: list[dict[str, Any]],
    source_telemetry_kind: str,
    synthetic_data: bool,
) -> dict[str, Any]:
    validation = validate_telemetry_rows(telemetry_rows)
    protected_state = build_protected_state(artifact_health)
    if validation.issues:
        return _build_blocked_report(
            artifact_health=artifact_health,
            protected_state=protected_state,
            source_telemetry_kind=source_telemetry_kind,
            synthetic_data=synthetic_data,
            source_telemetry_rows=len(telemetry_rows),
            data_quality_issues=validation.issues,
        )

    rows = validation.rows
    scenario_metrics = build_scenario_metrics(rows)
    action_metrics = build_action_metrics(rows)
    residual_watches = build_residual_watches(scenario_metrics, action_metrics)
    alerts = build_alerts(protected_state, residual_watches, action_metrics)
    decision = classify_monitoring_report(protected_state, alerts)
    return {
        "report_metadata": build_report_metadata(
            artifact_health,
            rows,
            source_telemetry_kind=source_telemetry_kind,
            synthetic_data=synthetic_data,
        ),
        "protected_state": protected_state,
        "scenario_metrics": scenario_metrics,
        "action_metrics": action_metrics,
        "residual_watches": residual_watches,
        "alerts": alerts,
        "decision": decision,
    }


def validate_telemetry_rows(rows: list[dict[str, Any]]) -> ValidationResult:
    issues: list[str] = []
    coerced_rows: list[dict[str, Any]] = []
    if not rows:
        return ValidationResult(rows=[], issues=["telemetry must contain at least one row"])

    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            issues.append(f"row {index} must be a mapping")
            continue
        missing = [field for field in TELEMETRY_FIELDS if field not in raw_row]
        if missing:
            issues.append(f"row {index} missing fields: {missing}")
            continue
        try:
            row = _coerce_telemetry_row(raw_row)
        except Exception as exc:
            issues.append(f"row {index} failed coercion: {exc}")
            continue
        for field in NUMERIC_FIELDS:
            value = row[field]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                issues.append(f"row {index} field {field} must be finite")
            if float(value) < 0:
                issues.append(f"row {index} field {field} must be non-negative")
        for field in POSITIVE_DENOMINATORS:
            if int(row[field]) <= 0:
                issues.append(f"row {index} denominator {field} must be positive")
        if not 0 <= int(row["action_id"]) < DEFAULT_EXTERNAL_DISCRETE_ACTION_COUNT:
            issues.append(f"row {index} action_id must be in 0..{DEFAULT_EXTERNAL_DISCRETE_ACTION_COUNT - 1}")
        try:
            _parse_timestamp(str(row["window_start_utc"]))
            _parse_timestamp(str(row["window_end_utc"]))
        except ValueError as exc:
            issues.append(f"row {index} timestamp invalid: {exc}")
        coerced_rows.append(row)
    return ValidationResult(rows=coerced_rows if not issues else [], issues=issues)


def build_protected_state(artifact_health: dict[str, Any]) -> dict[str, Any]:
    active_status = _nested_status(artifact_health, "active_registry_checks")
    models_status = _nested_status(artifact_health, "models_jsonl_checks")
    manifest_status = _nested_status(artifact_health, "production_manifest_checks")
    production_hash_status = _nested_status(artifact_health, "production_file_hash_checks")
    profile_status = _nested_status(artifact_health, "protected_path_profile_checks")
    process_status = _nested_status(artifact_health, "process_scan_result")
    runtime_status = _nested_status(artifact_health, "runtime_smoke_result")
    classification = str(artifact_health.get("final_classification", ""))
    clean = (
        classification == "ARTIFACT_HEALTH_CLEAN"
        and active_status == "PASS"
        and models_status == "PASS"
        and manifest_status == "PASS"
        and production_hash_status == "PASS"
        and profile_status == "PASS"
        and process_status == "PASS"
        and runtime_status == "PASS"
    )
    production_hashes = artifact_health.get("production_file_hash_checks", {}).get("actual_sha256", {})
    return {
        "artifact_health_classification": classification,
        "active_registry_status": active_status,
        "models_jsonl_status": models_status,
        "production_manifest_status": manifest_status,
        "production_hash_status": production_hash_status,
        "protected_path_profile_status": profile_status,
        "runtime_smoke_status": runtime_status,
        "process_scan_status": process_status,
        "active_models_hash": artifact_health.get("active_registry_checks", {}).get("sha256"),
        "models_jsonl_hash": artifact_health.get("models_jsonl_checks", {}).get("sha256"),
        "production_joint_hash": production_hashes.get("joint_torch_latest.pt"),
        "production_manifest_hash": artifact_health.get("production_manifest_checks", {}).get("sha256"),
        "active_registry_matches_handoff": active_status == "PASS",
        "production_manifest_matches_handoff": manifest_status == "PASS",
        "train_eval_gate_process_found": process_status != "PASS",
        "status": "clean" if clean else "drift",
    }


def build_report_metadata(
    artifact_health: dict[str, Any],
    telemetry_rows: list[dict[str, Any]],
    *,
    source_telemetry_kind: str,
    synthetic_data: bool,
) -> dict[str, Any]:
    identity = artifact_health.get("expected_production_identity", {})
    window_start = min(str(row["window_start_utc"]) for row in telemetry_rows) if telemetry_rows else None
    window_end = max(str(row["window_end_utc"]) for row in telemetry_rows) if telemetry_rows else None
    checkpoint = artifact_health.get("runtime_smoke_result", {}).get("checkpoint_path", DEFAULT_PRODUCTION_CHECKPOINT)
    return {
        "report_timestamp_utc": datetime.now(UTC).isoformat(),
        "monitoring_window_start_utc": window_start,
        "monitoring_window_end_utc": window_end,
        "logical_model_id": identity.get("logical_model_id", DEFAULT_LOGICAL_MODEL_ID),
        "runtime_family": DEFAULT_RUNTIME_FAMILY,
        "dqn_architecture": identity.get("dqn_architecture", DEFAULT_DQN_ARCHITECTURE),
        "hierarchical_init_method": identity.get("hierarchical_init_method", DEFAULT_HIERARCHICAL_INIT_METHOD),
        "contract": identity.get("contract", DEFAULT_CONTRACT),
        "observation_dim": int(identity.get("observation_dim", DEFAULT_OBSERVATION_DIM)),
        "continuous_action_dim": DEFAULT_CONTINUOUS_ACTION_DIM,
        "external_discrete_action_count": int(identity.get("action_count", DEFAULT_EXTERNAL_DISCRETE_ACTION_COUNT)),
        "production_checkpoint": checkpoint,
        "active_ppo_id": identity.get("active_ppo_id", DEFAULT_ACTIVE_PPO_ID),
        "active_dqn_id": identity.get("active_dqn_id", DEFAULT_ACTIVE_DQN_ID),
        "source_telemetry_kind": source_telemetry_kind,
        "source_telemetry_rows": len(telemetry_rows),
        "generator_version": GENERATOR_VERSION,
        "synthetic_data": bool(synthetic_data),
    }


def build_scenario_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario_or_regime"]), []).append(row)
    metrics: list[dict[str, Any]] = []
    for scenario, scenario_rows in sorted(by_scenario.items()):
        decision_steps = sum(int(row["decision_steps"]) for row in scenario_rows)
        orders = sum(int(row["orders"]) for row in scenario_rows)
        served = sum(int(row["served"]) for row in scenario_rows)
        late_orders = sum(int(row["late_orders"]) for row in scenario_rows)
        lateness_minutes = sum(float(row["lateness_minutes"]) for row in scenario_rows)
        delivered = sum(int(row["delivered"]) for row in scenario_rows)
        dispatch_attempts = sum(int(row["dispatch_attempts"]) for row in scenario_rows)
        dispatch_successes = sum(int(row["dispatch_successes"]) for row in scenario_rows)
        route_failure = sum(int(row["route_failure"]) for row in scenario_rows)
        no_vehicle = sum(int(row["no_vehicle"]) for row in scenario_rows)
        already_assigned = sum(int(row["already_assigned"]) for row in scenario_rows)
        no_work = sum(int(row["no_current"]) + int(row["no_unassigned"]) + int(row["failed_noop"]) for row in scenario_rows)
        top_row = max(scenario_rows, key=lambda row: int(row["action_attempts"]))
        top_attempts = int(top_row["action_attempts"])
        top_share = _safe_rate(top_attempts, decision_steps)
        verdict = _scenario_verdict(
            service_level=_safe_rate(served, orders),
            route_failure=route_failure,
            no_vehicle=no_vehicle,
            no_work=no_work,
        )
        metrics.append(
            {
                "scenario_or_regime": scenario,
                "rows": len(scenario_rows),
                "orders": orders,
                "decision_steps": decision_steps,
                "service_level": _safe_rate(served, orders),
                "lateness_mean": _safe_rate(lateness_minutes, late_orders) if late_orders else 0.0,
                "lateness_p95": max(float(row["lateness_p95"]) for row in scenario_rows),
                "dispatch_rate": _safe_rate(dispatch_attempts, decision_steps),
                "dispatch_success": _safe_rate(dispatch_successes, dispatch_attempts),
                "delivered_per_step": _safe_rate(delivered, decision_steps),
                "no_work_per_step": _safe_rate(no_work, decision_steps),
                "route_failure_per_step": _safe_rate(route_failure, decision_steps),
                "no_vehicle_per_step": _safe_rate(no_vehicle, decision_steps),
                "already_assigned_per_step": _safe_rate(already_assigned, decision_steps),
                "top_action_id": int(top_row["action_id"]),
                "top_action_share": top_share,
                "verdict": verdict,
                "notes": [],
            }
        )
    return metrics


def build_action_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    action_rows: list[dict[str, Any]] = []
    for row in rows:
        action_id = int(row["action_id"])
        decision_steps = int(row["decision_steps"])
        dispatch_attempts = int(row["dispatch_attempts"])
        orders = int(row["orders"])
        attempts = int(row["action_attempts"])
        hard_blocked = any(int(row[field]) > 0 for field in ("no_current", "no_unassigned", "failed_noop"))
        concentration_warning = _concentration_warning(str(row["scenario_or_regime"]), action_id, _safe_rate(attempts, decision_steps))
        action_rows.append(
            {
                "action_id": action_id,
                "decoded_action": decode_action(action_id),
                "scenario_or_regime": str(row["scenario_or_regime"]),
                "attempts": attempts,
                "rate_per_step": _safe_rate(attempts, decision_steps),
                "no_current": int(row["no_current"]),
                "no_unassigned": int(row["no_unassigned"]),
                "failed_noop": int(row["failed_noop"]),
                "route_failure": int(row["route_failure"]),
                "no_vehicle": int(row["no_vehicle"]),
                "already_assigned": int(row["already_assigned"]),
                "dispatch_success": _safe_rate(int(row["dispatch_successes"]), dispatch_attempts),
                "service_level": _safe_rate(int(row["served"]), orders),
                "warning_threshold_crossed": concentration_warning,
                "escalation_threshold_crossed": hard_blocked and action_id in WATCHED_ACTIONS,
                "secondary_fleet_rate": _safe_rate(int(row["secondary_fleet_attempts"]), dispatch_attempts),
                "reorder_none_rate": _safe_rate(int(row["reorder_none_decisions"]), decision_steps),
            }
        )
    return sorted(action_rows, key=lambda item: (item["scenario_or_regime"], item["action_id"]))


def build_residual_watches(scenario_metrics: list[dict[str, Any]], action_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    action_lookup = {(row["scenario_or_regime"], row["action_id"]): row for row in action_metrics}
    scenario_lookup = {row["scenario_or_regime"]: row for row in scenario_metrics}
    route_action = action_lookup.get(("route_disruption_congestion", 32), {})
    mixed_action = action_lookup.get(("mixed_stress", 24), {})
    route_scenario = scenario_lookup.get("route_disruption_congestion", {})
    mixed_scenario = scenario_lookup.get("mixed_stress", {})
    watches = [
        _action_concentration_watch(
            watch_id="route_action_32_concentration",
            action_row=route_action,
            scenario_row=route_scenario,
            baseline_context="route_disruption_congestion equal-budget seed42 20ep",
            baseline_rate=ACTION_32_BASELINE_RATE,
        ),
        _action_concentration_watch(
            watch_id="mixed_action_24_concentration",
            action_row=mixed_action,
            scenario_row=mixed_scenario,
            baseline_context="mixed_stress equal-budget seed42 20ep",
            baseline_rate=ACTION_24_BASELINE_RATE,
        ),
        _top_action_watch(scenario_metrics),
        _mixed_success_route_failure_watch(mixed_action, mixed_scenario),
        {
            "watch_id": "secondary_fleet_economics_unvalidated",
            "action_id": None,
            "baseline_context": "company fleet/cost data required",
            "equal_budget_baseline_rate": None,
            "current_rate": _max_rate(action_metrics, "secondary_fleet_rate"),
            "paired_operational_metrics": {},
            "decision": "info",
            "notes": ["Synthetic/public-route data cannot validate secondary_fleet economics."],
        },
        {
            "watch_id": "reorder_none_economics_unvalidated",
            "action_id": None,
            "baseline_context": "company inventory/cost data required",
            "equal_budget_baseline_rate": None,
            "current_rate": _max_rate(action_metrics, "reorder_none_rate"),
            "paired_operational_metrics": {},
            "decision": "info",
            "notes": ["Synthetic/public-route data cannot validate reorder-none economics."],
        },
    ]
    return watches


def build_alerts(
    protected_state: dict[str, Any],
    residual_watches: list[dict[str, Any]],
    action_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    if protected_state["status"] != "clean":
        alerts.append(
            {
                "level": "critical",
                "alert_id": "protected_state_drift",
                "scenario_or_regime": None,
                "action_id": None,
                "metric": "artifact_health",
                "observed_value": protected_state["artifact_health_classification"],
                "threshold": "ARTIFACT_HEALTH_CLEAN",
                "reason": "Artifact-health report is not clean; monitoring interpretation is gated.",
            }
        )
    for action_row in action_metrics:
        action_id = int(action_row["action_id"])
        if action_id not in WATCHED_ACTIONS:
            continue
        for field in ("no_current", "no_unassigned", "failed_noop"):
            if int(action_row[field]) > 0:
                alerts.append(
                    {
                        "level": "critical",
                        "alert_id": f"action_{action_id}_{field}",
                        "scenario_or_regime": action_row["scenario_or_regime"],
                        "action_id": action_id,
                        "metric": field,
                        "observed_value": int(action_row[field]),
                        "threshold": 0,
                        "reason": f"Watched action {action_id} has nonzero {field}.",
                    }
                )
    for watch in residual_watches:
        if watch["decision"] not in {"watch", "warning", "critical"}:
            continue
        action_id = watch.get("action_id")
        alert_id = watch["watch_id"]
        if alert_id == "mixed_action_24_concentration":
            alert_id = "action_24_concentration"
        elif alert_id == "route_action_32_concentration":
            alert_id = "action_32_concentration"
        alerts.append(
            {
                "level": watch["decision"],
                "alert_id": alert_id,
                "scenario_or_regime": watch["paired_operational_metrics"].get("scenario_or_regime"),
                "action_id": action_id,
                "metric": "rate_per_step",
                "observed_value": watch.get("current_rate"),
                "threshold": (
                    None
                    if watch.get("equal_budget_baseline_rate") is None
                    else watch["equal_budget_baseline_rate"] + MATERIAL_CONCENTRATION_MARGIN
                ),
                "reason": f"Residual watch {watch['watch_id']} decision is {watch['decision']}.",
            }
        )
    return alerts


def classify_monitoring_report(protected_state: dict[str, Any], alerts: list[dict[str, Any]]) -> dict[str, str]:
    if any(alert["alert_id"] == "telemetry_data_quality" for alert in alerts):
        return {
            "classification": CLASSIFICATION_DATA_QUALITY,
            "reason": "Telemetry input failed schema or denominator validation.",
            "recommended_action": "Fix fixture/data quality before interpreting monitoring metrics.",
        }
    if protected_state["status"] != "clean":
        return {
            "classification": CLASSIFICATION_PROTECTED_DRIFT,
            "reason": "Artifact-health protected state is not clean.",
            "recommended_action": "Run a read-only artifact-health investigation; do not train automatically.",
        }
    if any(alert["level"] == "critical" for alert in alerts):
        return {
            "classification": CLASSIFICATION_RISK,
            "reason": "A watched action has a critical action-quality blocker.",
            "recommended_action": "Open read-only production-risk investigation; do not train automatically.",
        }
    if any(alert["level"] == "warning" for alert in alerts):
        return {
            "classification": CLASSIFICATION_INVESTIGATION,
            "reason": "Residual watch coincides with operational degradation.",
            "recommended_action": "Investigate telemetry and business impact before any model work.",
        }
    if any(alert["level"] == "watch" for alert in alerts):
        return {
            "classification": CLASSIFICATION_WARNINGS,
            "reason": "Residual watch threshold crossed without critical operational degradation.",
            "recommended_action": "Track the next synthetic or approved monitoring window.",
        }
    return {
        "classification": CLASSIFICATION_CLEAN,
        "reason": "No warning or critical thresholds crossed.",
        "recommended_action": "Record report; no model action authorized.",
    }


def decode_action(action_id: int) -> str:
    if action_id == 24:
        return "dispatch + shortest + secondary_fleet + none"
    if action_id == 32:
        return "dispatch + low_congestion + secondary_fleet + none"
    return f"unknown_action_{action_id}"


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    artifact_health = load_artifact_health_report(args.artifact_health)
    telemetry_rows, source_kind, synthetic_data = _load_telemetry_from_args(args)
    report = build_monitoring_report(
        artifact_health=artifact_health,
        telemetry_rows=telemetry_rows,
        source_telemetry_kind=source_kind,
        synthetic_data=synthetic_data,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["decision"]["classification"])
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic/local-fixture hierarchical v1 monitoring report.")
    parser.add_argument("--artifact-health", type=Path, default=DEFAULT_ARTIFACT_HEALTH_PATH)
    parser.add_argument("--telemetry-json", type=Path)
    parser.add_argument("--telemetry-csv", type=Path)
    parser.add_argument(
        "--use-built-in-synthetic-fixture",
        choices=("clean", "warning_action24", "warning_action32", "critical_no_current"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_telemetry_from_args(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str, bool]:
    selected = [
        args.telemetry_json is not None,
        args.telemetry_csv is not None,
        args.use_built_in_synthetic_fixture is not None,
    ]
    if sum(selected) != 1:
        raise ValueError("exactly one telemetry source is required.")
    if args.telemetry_json is not None:
        return load_telemetry_json(args.telemetry_json), f"json:{args.telemetry_json}", True
    if args.telemetry_csv is not None:
        return load_telemetry_csv(args.telemetry_csv), f"csv:{args.telemetry_csv}", True
    return (
        built_in_synthetic_fixture(args.use_built_in_synthetic_fixture),
        f"built_in:{args.use_built_in_synthetic_fixture}",
        True,
    )


def _build_blocked_report(
    *,
    artifact_health: dict[str, Any],
    protected_state: dict[str, Any],
    source_telemetry_kind: str,
    synthetic_data: bool,
    source_telemetry_rows: int,
    data_quality_issues: list[str],
) -> dict[str, Any]:
    alerts = [
        {
            "level": "critical",
            "alert_id": "telemetry_data_quality",
            "scenario_or_regime": None,
            "action_id": None,
            "metric": "telemetry_schema",
            "observed_value": data_quality_issues,
            "threshold": "valid telemetry schema and denominators",
            "reason": "Telemetry input failed validation.",
        }
    ]
    if protected_state["status"] != "clean":
        alerts.append(
            {
                "level": "critical",
                "alert_id": "protected_state_drift",
                "scenario_or_regime": None,
                "action_id": None,
                "metric": "artifact_health",
                "observed_value": protected_state["artifact_health_classification"],
                "threshold": "ARTIFACT_HEALTH_CLEAN",
                "reason": "Artifact-health report is not clean.",
            }
        )
    return {
        "report_metadata": {
            **build_report_metadata(
                artifact_health,
                [],
                source_telemetry_kind=source_telemetry_kind,
                synthetic_data=synthetic_data,
            ),
            "source_telemetry_rows": source_telemetry_rows,
        },
        "protected_state": protected_state,
        "scenario_metrics": [],
        "action_metrics": [],
        "residual_watches": [],
        "alerts": alerts,
        "decision": classify_monitoring_report(protected_state, alerts),
    }


def _coerce_telemetry_row(raw_row: dict[str, Any]) -> dict[str, Any]:
    row = dict(raw_row)
    for field in INTEGER_FIELDS:
        row[field] = int(row[field])
    for field in ("lateness_minutes", "lateness_p95"):
        row[field] = float(row[field])
    row["window_start_utc"] = str(row["window_start_utc"])
    row["window_end_utc"] = str(row["window_end_utc"])
    row["scenario_or_regime"] = str(row["scenario_or_regime"])
    return row


def _telemetry_row(
    *,
    scenario_or_regime: str,
    action_id: int,
    decision_steps: int,
    orders: int,
    served: int,
    late_orders: int,
    lateness_minutes: float,
    lateness_p95: float,
    delivered: int,
    dispatch_attempts: int,
    dispatch_successes: int,
    action_attempts: int,
    secondary_fleet_attempts: int,
    reorder_none_decisions: int,
) -> dict[str, Any]:
    return {
        "window_start_utc": "2026-06-11T00:00:00Z",
        "window_end_utc": "2026-06-11T01:00:00Z",
        "scenario_or_regime": scenario_or_regime,
        "decision_steps": decision_steps,
        "orders": orders,
        "served": served,
        "late_orders": late_orders,
        "lateness_minutes": lateness_minutes,
        "lateness_p95": lateness_p95,
        "delivered": delivered,
        "dispatch_attempts": dispatch_attempts,
        "dispatch_successes": dispatch_successes,
        "action_id": action_id,
        "action_attempts": action_attempts,
        "no_current": 0,
        "no_unassigned": 0,
        "failed_noop": 0,
        "route_failure": 0,
        "no_vehicle": 0,
        "already_assigned": 0,
        "secondary_fleet_attempts": secondary_fleet_attempts,
        "reorder_none_decisions": reorder_none_decisions,
    }


def _action_concentration_watch(
    *,
    watch_id: str,
    action_row: dict[str, Any],
    scenario_row: dict[str, Any],
    baseline_context: str,
    baseline_rate: float,
) -> dict[str, Any]:
    current_rate = float(action_row.get("rate_per_step", 0.0))
    hard_blocked = any(int(action_row.get(field, 0)) > 0 for field in ("no_current", "no_unassigned", "failed_noop"))
    operational_degradation = _operational_degradation(action_row, scenario_row)
    if hard_blocked:
        decision = "critical"
    elif current_rate > baseline_rate + MATERIAL_CONCENTRATION_MARGIN and operational_degradation:
        decision = "warning"
    elif current_rate > baseline_rate + MATERIAL_CONCENTRATION_MARGIN:
        decision = "watch"
    else:
        decision = "info"
    return {
        "watch_id": watch_id,
        "action_id": action_row.get("action_id"),
        "baseline_context": baseline_context,
        "equal_budget_baseline_rate": baseline_rate,
        "current_rate": current_rate,
        "paired_operational_metrics": {
            "scenario_or_regime": action_row.get("scenario_or_regime"),
            "service_level": action_row.get("service_level", scenario_row.get("service_level", 0.0)),
            "lateness": scenario_row.get("lateness_mean", 0.0),
            "route_failure": action_row.get("route_failure", 0),
            "no_vehicle": action_row.get("no_vehicle", 0),
            "already_assigned": action_row.get("already_assigned", 0),
            "no_current": action_row.get("no_current", 0),
            "no_unassigned": action_row.get("no_unassigned", 0),
            "failed_noop": action_row.get("failed_noop", 0),
        },
        "decision": decision,
        "notes": [],
    }


def _top_action_watch(scenario_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    top_row = max(scenario_metrics, key=lambda row: float(row["top_action_share"]), default={})
    current_rate = float(top_row.get("top_action_share", 0.0))
    return {
        "watch_id": "top_action_concentration",
        "action_id": top_row.get("top_action_id"),
        "baseline_context": "material top-action concentration monitor",
        "equal_budget_baseline_rate": None,
        "current_rate": current_rate,
        "paired_operational_metrics": {
            "scenario_or_regime": top_row.get("scenario_or_regime"),
            "service_level": top_row.get("service_level"),
            "lateness": top_row.get("lateness_mean"),
        },
        "decision": "watch" if current_rate >= 0.75 else "info",
        "notes": [],
    }


def _mixed_success_route_failure_watch(action_row: dict[str, Any], scenario_row: dict[str, Any]) -> dict[str, Any]:
    route_failure = int(action_row.get("route_failure", 0))
    service_level = float(scenario_row.get("service_level", 1.0))
    if route_failure and service_level < 0.90:
        decision = "warning"
    elif route_failure:
        decision = "watch"
    else:
        decision = "info"
    return {
        "watch_id": "mixed_success_route_failure",
        "action_id": action_row.get("action_id"),
        "baseline_context": "mixed-success route-failure warning watch",
        "equal_budget_baseline_rate": None,
        "current_rate": None,
        "paired_operational_metrics": {
            "scenario_or_regime": action_row.get("scenario_or_regime", "mixed_stress"),
            "service_level": service_level,
            "route_failure": route_failure,
        },
        "decision": decision,
        "notes": [],
    }


def _operational_degradation(action_row: dict[str, Any], scenario_row: dict[str, Any]) -> bool:
    return (
        float(action_row.get("service_level", scenario_row.get("service_level", 1.0))) < 0.90
        or float(action_row.get("dispatch_success", 1.0)) < 0.90
        or int(action_row.get("route_failure", 0)) > 0
        or int(action_row.get("no_vehicle", 0)) > 0
    )


def _scenario_verdict(*, service_level: float, route_failure: int, no_vehicle: int, no_work: int) -> str:
    if no_work:
        return "critical"
    if service_level < 0.90 and (route_failure or no_vehicle):
        return "warning"
    if route_failure or no_vehicle:
        return "watch"
    return "pass"


def _concentration_warning(scenario: str, action_id: int, rate: float) -> bool:
    if scenario == "mixed_stress" and action_id == 24:
        return rate > ACTION_24_BASELINE_RATE + MATERIAL_CONCENTRATION_MARGIN
    if scenario == "route_disruption_congestion" and action_id == 32:
        return rate > ACTION_32_BASELINE_RATE + MATERIAL_CONCENTRATION_MARGIN
    return False


def _nested_status(payload: dict[str, Any], key: str) -> str | None:
    section = payload.get(key)
    if not isinstance(section, dict):
        return None
    return section.get("status")


def _parse_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _max_rate(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return max(float(row.get(field, 0.0)) for row in rows)


if __name__ == "__main__":
    raise SystemExit(main())
