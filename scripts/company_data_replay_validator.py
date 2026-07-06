"""Synthetic-only company replay validation harness.

This is a readiness harness for future approved company data. It uses synthetic
fixtures or caller-supplied rows and validates whether action 24/32 replay data
can answer fleet, route, inventory, and cost questions. It does not ingest
private data by default and does not write databases or model artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "dispatch_attempts": (
        "dispatch_attempt_id",
        "order_id",
        "action_id",
        "dispatch_status",
        "dispatch_failure_reason",
        "fleet_type",
        "carrier_id",
        "vehicle_id",
        "route_id",
    ),
    "fleet": ("vehicle_id", "carrier_id", "fleet_type", "accepted", "fleet_cost"),
    "inventory": ("order_id", "sku_id", "stock_on_hand_at_decision", "stockout_after_decision", "reorder_type"),
    "costs": ("order_id", "primary_fleet_cost", "secondary_fleet_cost", "stockout_cost", "holding_cost", "lateness_penalty_cost"),
    "routes": ("route_id", "planned_route_type", "route_failure_flag", "actual_travel_time", "planned_travel_time"),
}

PROTECTED_OUTPUT_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("models", "registry"),
    ("models", "production"),
    ("models", "baselines"),
    ("models", "checkpoints"),
    ("models", "eval"),
    ("db",),
)


def built_in_replay_fixture(name: str) -> dict[str, list[dict[str, str]]]:
    base = {
        "dispatch_attempts": [
            {
                "dispatch_attempt_id": "d1",
                "order_id": "o1",
                "action_id": "24",
                "dispatch_status": "success",
                "dispatch_failure_reason": "none",
                "fleet_type": "secondary",
                "carrier_id": "c1",
                "vehicle_id": "v1",
                "route_id": "r1",
            },
            {
                "dispatch_attempt_id": "d2",
                "order_id": "o2",
                "action_id": "32",
                "dispatch_status": "success",
                "dispatch_failure_reason": "none",
                "fleet_type": "secondary",
                "carrier_id": "c2",
                "vehicle_id": "v2",
                "route_id": "r2",
            },
        ],
        "fleet": [
            {"vehicle_id": "v1", "carrier_id": "c1", "fleet_type": "secondary", "accepted": "1", "fleet_cost": "42"},
            {"vehicle_id": "v2", "carrier_id": "c2", "fleet_type": "secondary", "accepted": "1", "fleet_cost": "50"},
        ],
        "inventory": [
            {"order_id": "o1", "sku_id": "sku1", "stock_on_hand_at_decision": "10", "stockout_after_decision": "0", "reorder_type": "none"},
            {"order_id": "o2", "sku_id": "sku2", "stock_on_hand_at_decision": "8", "stockout_after_decision": "0", "reorder_type": "none"},
        ],
        "costs": [
            {
                "order_id": "o1",
                "primary_fleet_cost": "28",
                "secondary_fleet_cost": "42",
                "stockout_cost": "100",
                "holding_cost": "2",
                "lateness_penalty_cost": "80",
            },
            {
                "order_id": "o2",
                "primary_fleet_cost": "36",
                "secondary_fleet_cost": "50",
                "stockout_cost": "100",
                "holding_cost": "2",
                "lateness_penalty_cost": "120",
            },
        ],
        "routes": [
            {"route_id": "r1", "planned_route_type": "shortest", "route_failure_flag": "0", "actual_travel_time": "58", "planned_travel_time": "55"},
            {"route_id": "r2", "planned_route_type": "low_congestion", "route_failure_flag": "0", "actual_travel_time": "70", "planned_travel_time": "68"},
        ],
    }
    if name == "valid_minimal":
        return deepcopy(base)
    if name == "missing_secondary_cost":
        modified = deepcopy(base)
        for row in modified["costs"]:
            row.pop("secondary_fleet_cost", None)
        return modified
    if name == "no_reorder_stockout":
        modified = deepcopy(base)
        modified["inventory"][1]["stockout_after_decision"] = "1"
        return modified
    raise ValueError(f"unknown replay fixture: {name}")


def build_replay_validation_report(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    synthetic_only: bool = True,
    input_mode: str = "built_in_synthetic_fixture",
) -> dict[str, Any]:
    normalized = {
        table: [{str(key): "" if value is None else str(value) for key, value in row.items()} for row in rows]
        for table, rows in tables.items()
    }
    missing_fields = _missing_fields(normalized)
    action_summary = _action_watch_summary(normalized.get("dispatch_attempts", []))
    fleet_economics = _fleet_economics(normalized.get("costs", []))
    inventory_reorder = _inventory_reorder(normalized.get("inventory", []))
    route_reliability = _route_reliability(normalized.get("routes", []))
    warnings = []
    if inventory_reorder["none_reorder_stockout_rate"] and inventory_reorder["none_reorder_stockout_rate"] > 0:
        warnings.append("stockout_after_no_reorder")
    failures = [table for table, fields in missing_fields.items() if fields]
    if failures:
        classification = "COMPANY_REPLAY_SCHEMA_NEEDS_FIXES"
        reason = "Required replay validation fields are missing."
    elif warnings:
        classification = "COMPANY_REPLAY_SCHEMA_WARNINGS_ONLY"
        reason = "Replay schema is usable but synthetic warning conditions are present."
    else:
        classification = "COMPANY_REPLAY_SCHEMA_READY"
        reason = "Synthetic replay fixture can answer action 24/32 fleet, route, inventory, and cost questions."
    return {
        "report_metadata": {
            "validator_version": "company_data_replay_validator_v1",
            "report_timestamp_utc": datetime.now(UTC).isoformat(),
            "synthetic_only": bool(synthetic_only),
            "input_mode": input_mode,
        },
        "missing_fields_by_table": missing_fields,
        "action_watch_summary": action_summary,
        "fleet_economics": fleet_economics,
        "inventory_reorder": inventory_reorder,
        "route_reliability": route_reliability,
        "warnings": warnings,
        "decision": {"classification": classification, "reason": reason},
    }


def load_fixture_root(root: Path) -> dict[str, list[dict[str, str]]]:
    tables: dict[str, list[dict[str, str]]] = {}
    for table in REQUIRED_FIELDS:
        path = Path(root) / f"{table}.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            tables[table] = [dict(row) for row in csv.DictReader(handle)]
    return tables


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path = Path(path)
    _reject_protected_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite replay validation report: {path}")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _missing_fields(tables: Mapping[str, Sequence[Mapping[str, str]]]) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for table, required in REQUIRED_FIELDS.items():
        columns = {key for row in tables.get(table, []) for key in row}
        missing[table] = [field for field in required if field not in columns]
    return missing


def _action_watch_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    counts = {"24": 0, "32": 0}
    failure_counts = {"24": 0, "32": 0}
    for row in rows:
        action = str(row.get("action_id", ""))
        if action in counts:
            counts[action] += 1
            if str(row.get("dispatch_status", "")).lower() != "success":
                failure_counts[action] += 1
    total = len(rows)
    return {
        "dispatch_attempt_rows": total,
        "action_24_count": counts["24"],
        "action_32_count": counts["32"],
        "action_24_rate": counts["24"] / total if total else 0.0,
        "action_32_rate": counts["32"] / total if total else 0.0,
        "action_24_failure_count": failure_counts["24"],
        "action_32_failure_count": failure_counts["32"],
    }


def _fleet_economics(cost_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    premiums = []
    for row in cost_rows:
        primary = _finite_float(row.get("primary_fleet_cost"))
        secondary = _finite_float(row.get("secondary_fleet_cost"))
        if primary is not None and secondary is not None:
            premiums.append(secondary - primary)
    return {"secondary_cost_premium_mean": mean(premiums) if premiums else None, "compared_rows": len(premiums)}


def _inventory_reorder(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    none_rows = [row for row in rows if str(row.get("reorder_type", "")).lower() == "none"]
    stockouts = sum(1 for row in none_rows if _truthy(row.get("stockout_after_decision")))
    return {
        "none_reorder_rows": len(none_rows),
        "none_reorder_stockout_count": stockouts,
        "none_reorder_stockout_rate": stockouts / len(none_rows) if none_rows else 0.0,
    }


def _route_reliability(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    failures = sum(1 for row in rows if _truthy(row.get("route_failure_flag")))
    ratios = []
    for row in rows:
        planned = _finite_float(row.get("planned_travel_time"))
        actual = _finite_float(row.get("actual_travel_time"))
        if planned and planned > 0 and actual is not None:
            ratios.append(actual / planned)
    return {
        "route_rows": len(rows),
        "route_failure_rate": failures / len(rows) if rows else 0.0,
        "actual_to_planned_time_ratio_mean": mean(ratios) if ratios else None,
    }


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _reject_protected_path(path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    candidate = Path(path).resolve(strict=False)
    for prefix in PROTECTED_OUTPUT_PREFIXES:
        protected = repo_root.joinpath(*prefix).resolve(strict=False)
        if candidate == protected or protected in candidate.parents:
            raise ValueError(f"refusing to write under protected path: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic company replay validation harness.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--use-built-in-fixture", choices=("valid_minimal", "missing_secondary_cost", "no_reorder_stockout"))
    source.add_argument("--fixture-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.use_built_in_fixture:
        tables = built_in_replay_fixture(args.use_built_in_fixture)
        report = build_replay_validation_report(tables, synthetic_only=True, input_mode="built_in_synthetic_fixture")
    else:
        tables = load_fixture_root(args.fixture_root)
        report = build_replay_validation_report(tables, synthetic_only=False, input_mode="caller_supplied_fixture_root")
    write_report(args.output, report)
    print(report["decision"]["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
