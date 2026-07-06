from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any, Sequence


VALIDATOR_VERSION = "company_data_intake_validator_v1"

CLASSIFICATION_READY = "COMPANY_DATA_SCHEMA_READY"
CLASSIFICATION_WARNINGS = "COMPANY_DATA_SCHEMA_WARNINGS_ONLY"
CLASSIFICATION_NEEDS_FIXES = "COMPANY_DATA_SCHEMA_NEEDS_FIXES"
CLASSIFICATION_BLOCKED = "COMPANY_DATA_SCHEMA_BLOCKED"

TABLE_ORDER = (
    "orders",
    "dispatch_attempts",
    "deliveries",
    "routes",
    "fleet",
    "inventory",
    "costs",
)

TABLE_SCHEMAS: dict[str, dict[str, list[str]]] = {
    "orders": {
        "required": [
            "order_id",
            "created_at",
            "promised_window_start",
            "promised_window_end",
            "order_priority",
            "order_size",
            "zone_id",
        ],
        "optional": ["customer_id", "customer_location_id", "latitude", "longitude", "sku_id", "site_id"],
    },
    "dispatch_attempts": {
        "required": [
            "dispatch_attempt_id",
            "order_id",
            "decision_timestamp",
            "dispatch_attempt_timestamp",
            "dispatch_status",
            "dispatch_failure_reason",
            "action_id",
            "vehicle_id",
            "carrier_id",
            "fleet_type",
            "route_id",
        ],
        "optional": [
            "no_vehicle_flag",
            "already_assigned_flag",
            "route_failure_flag",
            "carrier_rejected_flag",
            "duplicate_assignment_flag",
            "policy_version",
        ],
    },
    "deliveries": {
        "required": [
            "order_id",
            "dispatch_attempt_id",
            "pickup_timestamp",
            "delivery_timestamp",
            "delivery_status",
            "delivered_on_time",
        ],
        "optional": ["failed_attempt_reason", "lateness_minutes", "actual_sequence_rank"],
    },
    "routes": {
        "required": [
            "route_id",
            "order_id",
            "planned_route_type",
            "planned_distance",
            "planned_travel_time",
            "route_failure_flag",
        ],
        "optional": [
            "actual_route_id",
            "actual_distance",
            "actual_travel_time",
            "planned_sequence_rank",
            "actual_sequence_rank",
            "reroute_flag",
            "congestion_delay_flag",
            "route_provider",
        ],
    },
    "fleet": {
        "required": [
            "vehicle_id",
            "carrier_id",
            "fleet_type",
            "vehicle_capacity",
            "available_at_decision",
            "assigned_at",
            "fleet_cost",
        ],
        "optional": ["accepted_at", "cancelled_at", "fleet_reliability_score"],
    },
    "inventory": {
        "required": [
            "sku_id",
            "site_id",
            "order_id",
            "stock_on_hand_at_decision",
            "backlog_quantity",
            "stockout_flag",
            "supplier_lead_time",
        ],
        "optional": ["allocated_quantity", "reorder_event_id", "reorder_type", "supplier_id"],
    },
    "costs": {
        "required": [
            "order_id",
            "primary_fleet_cost",
            "secondary_fleet_cost",
            "lateness_penalty_cost",
            "stockout_cost",
            "holding_cost",
        ],
        "optional": ["route_cost", "emergency_reorder_cost", "cancellation_cost"],
    },
}

PRIMARY_KEYS: dict[str, list[str]] = {
    "orders": ["order_id"],
    "dispatch_attempts": ["dispatch_attempt_id"],
    "deliveries": ["dispatch_attempt_id"],
    "routes": ["route_id"],
    "fleet": ["vehicle_id"],
    "inventory": ["sku_id", "site_id", "order_id"],
    "costs": ["order_id"],
}

TIMESTAMP_FIELDS: dict[str, list[str]] = {
    "orders": ["created_at", "promised_window_start", "promised_window_end"],
    "dispatch_attempts": ["decision_timestamp", "dispatch_attempt_timestamp"],
    "deliveries": ["pickup_timestamp", "delivery_timestamp"],
    "fleet": ["available_at_decision", "assigned_at", "accepted_at", "cancelled_at"],
}

CONTROLLED_LABELS: dict[str, dict[str, set[str]]] = {
    "fleet": {"fleet_type": {"primary", "secondary"}},
    "dispatch_attempts": {"fleet_type": {"primary", "secondary"}},
    "inventory": {"reorder_type": {"none", "conservative", "aggressive", "emergency"}},
}

NON_NEGATIVE_FIELDS: dict[str, list[str]] = {
    "orders": ["order_size"],
    "routes": ["planned_distance", "planned_travel_time", "actual_distance", "actual_travel_time"],
    "fleet": ["vehicle_capacity", "fleet_cost", "fleet_reliability_score"],
    "inventory": [
        "stock_on_hand_at_decision",
        "backlog_quantity",
        "supplier_lead_time",
        "allocated_quantity",
    ],
    "costs": [
        "primary_fleet_cost",
        "secondary_fleet_cost",
        "lateness_penalty_cost",
        "stockout_cost",
        "holding_cost",
        "route_cost",
        "emergency_reorder_cost",
        "cancellation_cost",
    ],
    "deliveries": ["lateness_minutes", "actual_sequence_rank"],
    "dispatch_attempts": ["action_id"],
}

JOIN_RULES = (
    {
        "check_id": "dispatch_attempts_order_id_join",
        "left_table": "dispatch_attempts",
        "left_key": "order_id",
        "right_table": "orders",
        "right_key": "order_id",
        "missing_status": "fail",
    },
    {
        "check_id": "deliveries_order_id_join",
        "left_table": "deliveries",
        "left_key": "order_id",
        "right_table": "orders",
        "right_key": "order_id",
        "missing_status": "fail",
    },
    {
        "check_id": "deliveries_dispatch_attempt_id_join",
        "left_table": "deliveries",
        "left_key": "dispatch_attempt_id",
        "right_table": "dispatch_attempts",
        "right_key": "dispatch_attempt_id",
        "missing_status": "fail",
    },
    {
        "check_id": "routes_order_id_join",
        "left_table": "routes",
        "left_key": "order_id",
        "right_table": "orders",
        "right_key": "order_id",
        "missing_status": "fail",
    },
    {
        "check_id": "dispatch_attempts_route_id_join",
        "left_table": "dispatch_attempts",
        "left_key": "route_id",
        "right_table": "routes",
        "right_key": "route_id",
        "missing_status": "fail",
    },
    {
        "check_id": "dispatch_attempts_vehicle_id_join",
        "left_table": "dispatch_attempts",
        "left_key": "vehicle_id",
        "right_table": "fleet",
        "right_key": "vehicle_id",
        "missing_status": "fail",
    },
    {
        "check_id": "dispatch_attempts_carrier_id_join",
        "left_table": "dispatch_attempts",
        "left_key": "carrier_id",
        "right_table": "fleet",
        "right_key": "carrier_id",
        "missing_status": "warning",
    },
    {
        "check_id": "inventory_order_id_join",
        "left_table": "inventory",
        "left_key": "order_id",
        "right_table": "orders",
        "right_key": "order_id",
        "missing_status": "fail",
    },
    {
        "check_id": "costs_order_id_join",
        "left_table": "costs",
        "left_key": "order_id",
        "right_table": "orders",
        "right_key": "order_id",
        "missing_status": "fail",
    },
)

BUILT_IN_FIXTURES = (
    "valid_minimal",
    "missing_required_columns",
    "bad_timestamps",
    "duplicate_keys",
    "invalid_labels",
)


def built_in_synthetic_fixture(name: str) -> dict[str, list[dict[str, str]]]:
    tables = _valid_minimal_tables()
    if name == "valid_minimal":
        return tables
    if name == "missing_required_columns":
        tables["orders"][0].pop("promised_window_end")
        tables["dispatch_attempts"][0].pop("dispatch_attempt_id")
        return tables
    if name == "bad_timestamps":
        tables["orders"][0]["created_at"] = "not-a-timestamp"
        tables["orders"][0]["promised_window_start"] = "2026-06-11T12:00:00Z"
        tables["orders"][0]["promised_window_end"] = "2026-06-11T11:00:00Z"
        return tables
    if name == "duplicate_keys":
        tables["orders"].append(dict(tables["orders"][0]))
        tables["dispatch_attempts"].append(dict(tables["dispatch_attempts"][0]))
        return tables
    if name == "invalid_labels":
        tables["fleet"][0]["fleet_type"] = "overflow_unknown"
        tables["dispatch_attempts"][0]["fleet_type"] = "overflow_unknown"
        tables["inventory"][0]["reorder_type"] = "delay_forever"
        return tables
    raise ValueError(f"unknown built-in synthetic fixture: {name}")


def load_fixture_root(path: Path | str) -> dict[str, list[dict[str, str]]]:
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"synthetic fixture root is not a directory: {root}")
    tables: dict[str, list[dict[str, str]]] = {}
    for table_name in TABLE_ORDER:
        csv_path = root / f"{table_name}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            tables[table_name] = [dict(row) for row in csv.DictReader(handle)]
    return tables


def build_company_data_schema_report(
    tables: dict[str, list[dict[str, Any]]],
    *,
    source_kind: str,
    synthetic_data: bool,
) -> dict[str, Any]:
    normalized_tables = _normalize_tables(tables)
    table_summaries = _build_table_summaries(normalized_tables)
    data_quality_checks: list[dict[str, Any]] = []
    data_quality_checks.extend(validate_required_columns(normalized_tables))
    data_quality_checks.extend(validate_required_values(normalized_tables))
    data_quality_checks.extend(validate_duplicate_primary_keys(normalized_tables))
    data_quality_checks.extend(validate_timestamps(normalized_tables))
    data_quality_checks.extend(validate_controlled_labels(normalized_tables))
    data_quality_checks.extend(validate_non_negative_fields(normalized_tables))
    data_quality_checks.extend(validate_action_ids(normalized_tables))
    data_quality_checks.extend(validate_ordering_rules(normalized_tables))
    join_key_checks = validate_join_keys(normalized_tables)
    warnings = [check for check in data_quality_checks + join_key_checks if check["status"] == "warning"]
    decision = classify_schema_report(data_quality_checks, join_key_checks)
    _attach_table_issues(table_summaries, data_quality_checks, join_key_checks)
    return {
        "report_metadata": {
            "report_timestamp_utc": datetime.now(UTC).isoformat(),
            "validator_version": VALIDATOR_VERSION,
            "synthetic_data": bool(synthetic_data),
            "source_kind": source_kind,
            "table_count": len(TABLE_ORDER),
            "tables_present": sorted(normalized_tables),
        },
        "table_summaries": table_summaries,
        "join_key_checks": join_key_checks,
        "data_quality_checks": data_quality_checks,
        "warnings": warnings,
        "decision": decision,
    }


def validate_required_columns(tables: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for table_name in TABLE_ORDER:
        rows = tables.get(table_name)
        if rows is None:
            checks.append(
                {
                    "check_id": f"{table_name}_table_present",
                    "table": table_name,
                    "status": "blocked",
                    "missing_columns": TABLE_SCHEMAS[table_name]["required"],
                    "message": f"Required table {table_name}.csv is missing.",
                }
            )
            continue
        columns = _columns_for_rows(rows)
        missing = [column for column in TABLE_SCHEMAS[table_name]["required"] if column not in columns]
        checks.append(
            {
                "check_id": f"{table_name}_required_columns",
                "table": table_name,
                "status": "fail" if missing else "pass",
                "missing_columns": missing,
                "message": "Required columns missing." if missing else "Required columns present.",
            }
        )
    return checks


def validate_required_values(tables: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for table_name in TABLE_ORDER:
        blank_values: list[dict[str, Any]] = []
        blank_fields: set[str] = set()
        for index, row in enumerate(tables.get(table_name, [])):
            for field in TABLE_SCHEMAS[table_name]["required"]:
                if str(row.get(field, "")).strip() == "":
                    blank_values.append({"row": index, "field": field})
                    blank_fields.add(field)
        checks.append(
            {
                "check_id": f"{table_name}_required_values",
                "table": table_name,
                "status": "fail" if blank_values else "pass",
                "blank_fields": sorted(blank_fields),
                "blank_values": blank_values,
                "message": "Required fields must be non-empty." if blank_values else "Required field values are non-empty.",
            }
        )
    return checks


def validate_duplicate_primary_keys(tables: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for table_name in TABLE_ORDER:
        rows = tables.get(table_name, [])
        primary_key = PRIMARY_KEYS[table_name]
        duplicates = _duplicate_keys(rows, primary_key)
        checks.append(
            {
                "check_id": f"{table_name}_duplicate_primary_key",
                "table": table_name,
                "primary_key": primary_key,
                "status": "fail" if duplicates else "pass",
                "duplicate_primary_keys": duplicates,
                "message": "Duplicate primary keys found." if duplicates else "Primary keys are unique.",
            }
        )
    return checks


def validate_timestamps(tables: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for table_name, fields in TIMESTAMP_FIELDS.items():
        invalid: list[dict[str, Any]] = []
        naive: list[dict[str, Any]] = []
        for index, row in enumerate(tables.get(table_name, [])):
            for field in fields:
                if field not in row or row.get(field, "") == "":
                    continue
                parsed = _parse_timestamp(str(row[field]))
                if parsed is None:
                    invalid.append({"row": index, "field": field, "value": row[field]})
                elif parsed.tzinfo is None:
                    naive.append({"row": index, "field": field, "value": row[field]})
        status = "fail" if invalid else "warning" if naive else "pass"
        checks.append(
            {
                "check_id": f"{table_name}_timestamp_parse",
                "table": table_name,
                "status": status,
                "invalid_values": invalid,
                "timezone_naive_values": naive,
                "message": _status_message(status, "Required timestamps parse with timezone semantics."),
            }
        )
    return checks


def validate_controlled_labels(tables: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for table_name, field_map in CONTROLLED_LABELS.items():
        for field, allowed in field_map.items():
            invalid: list[dict[str, Any]] = []
            for index, row in enumerate(tables.get(table_name, [])):
                value = str(row.get(field, "")).strip()
                if value and value not in allowed:
                    invalid.append({"row": index, "field": field, "value": value, "allowed": sorted(allowed)})
            checks.append(
                {
                    "check_id": f"{table_name}_{field}_controlled_label",
                    "table": table_name,
                    "field": field,
                    "status": "fail" if invalid else "pass",
                    "invalid_values": invalid,
                    "message": "Controlled label violation." if invalid else "Controlled labels valid.",
                }
            )
    return checks


def validate_non_negative_fields(tables: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for table_name, fields in NON_NEGATIVE_FIELDS.items():
        invalid: list[dict[str, Any]] = []
        for index, row in enumerate(tables.get(table_name, [])):
            for field in fields:
                if field not in row or row.get(field, "") == "":
                    continue
                value = _parse_number(row[field])
                if value is None or value < 0:
                    invalid.append({"row": index, "field": field, "value": row[field]})
        checks.append(
            {
                "check_id": f"{table_name}_non_negative_values",
                "table": table_name,
                "status": "fail" if invalid else "pass",
                "invalid_values": invalid,
                "message": "Negative or non-numeric values found." if invalid else "Non-negative values valid.",
            }
        )
    return checks


def validate_action_ids(tables: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    for index, row in enumerate(tables.get("dispatch_attempts", [])):
        raw_value = row.get("action_id")
        try:
            action_id = int(str(raw_value))
        except Exception:
            invalid.append({"row": index, "field": "action_id", "value": raw_value})
            continue
        if not 0 <= action_id < 48:
            invalid.append({"row": index, "field": "action_id", "value": raw_value})
    return [
        {
            "check_id": "dispatch_attempts_action_id_range",
            "table": "dispatch_attempts",
            "status": "fail" if invalid else "pass",
            "invalid_values": invalid,
            "message": "action_id must be integer 0..47." if invalid else "action_id values are valid.",
        }
    ]


def validate_ordering_rules(tables: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    promised_failures = []
    for index, row in enumerate(tables.get("orders", [])):
        start = _parse_timestamp(row.get("promised_window_start", ""))
        end = _parse_timestamp(row.get("promised_window_end", ""))
        if start is not None and end is not None and start > end:
            promised_failures.append({"row": index, "start": row.get("promised_window_start"), "end": row.get("promised_window_end")})
    checks.append(
        {
            "check_id": "orders_promised_window_order",
            "table": "orders",
            "status": "fail" if promised_failures else "pass",
            "invalid_values": promised_failures,
            "message": "promised_window_start must be <= promised_window_end."
            if promised_failures
            else "Promised windows are ordered.",
        }
    )

    dispatch_by_id = {row.get("dispatch_attempt_id"): row for row in tables.get("dispatch_attempts", [])}
    sequence_failures = []
    dispatch_pickup_failures = []
    for index, delivery in enumerate(tables.get("deliveries", [])):
        dispatch = dispatch_by_id.get(delivery.get("dispatch_attempt_id"), {})
        decision_ts = _parse_timestamp(dispatch.get("decision_timestamp", ""))
        dispatch_ts = _parse_timestamp(dispatch.get("dispatch_attempt_timestamp", ""))
        pickup_ts = _parse_timestamp(delivery.get("pickup_timestamp", ""))
        delivery_ts = _parse_timestamp(delivery.get("delivery_timestamp", ""))
        if decision_ts is not None and pickup_ts is not None and delivery_ts is not None:
            if not decision_ts <= pickup_ts <= delivery_ts:
                sequence_failures.append({"row": index, "dispatch_attempt_id": delivery.get("dispatch_attempt_id")})
        if dispatch_ts is not None and pickup_ts is not None and dispatch_ts > pickup_ts:
            dispatch_pickup_failures.append({"row": index, "dispatch_attempt_id": delivery.get("dispatch_attempt_id")})
    checks.append(
        {
            "check_id": "deliveries_decision_pickup_delivery_order",
            "table": "deliveries",
            "status": "fail" if sequence_failures else "pass",
            "invalid_values": sequence_failures,
            "message": "decision_timestamp <= pickup_timestamp <= delivery_timestamp violated."
            if sequence_failures
            else "Decision, pickup, and delivery timestamps are ordered.",
        }
    )
    checks.append(
        {
            "check_id": "deliveries_dispatch_pickup_order",
            "table": "deliveries",
            "status": "fail" if dispatch_pickup_failures else "pass",
            "invalid_values": dispatch_pickup_failures,
            "message": "dispatch_attempt_timestamp <= pickup_timestamp violated."
            if dispatch_pickup_failures
            else "Dispatch and pickup timestamps are ordered.",
        }
    )
    return checks


def validate_join_keys(tables: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for rule in JOIN_RULES:
        left_rows = tables.get(str(rule["left_table"]), [])
        right_rows = tables.get(str(rule["right_table"]), [])
        left_key = str(rule["left_key"])
        right_key = str(rule["right_key"])
        right_values = {row.get(right_key) for row in right_rows if row.get(right_key)}
        missing_values = sorted({row.get(left_key, "") for row in left_rows if row.get(left_key, "") not in right_values})
        status = "pass"
        if missing_values:
            status = str(rule["missing_status"])
        checks.append(
            {
                "check_id": rule["check_id"],
                "left_table": rule["left_table"],
                "left_key": left_key,
                "right_table": rule["right_table"],
                "right_key": right_key,
                "status": status,
                "missing_values": missing_values,
                "message": _status_message(status, "Join keys are present."),
            }
        )
    return checks


def classify_schema_report(
    data_quality_checks: list[dict[str, Any]],
    join_key_checks: list[dict[str, Any]],
) -> dict[str, str]:
    checks = data_quality_checks + join_key_checks
    if any(check["status"] == "blocked" for check in checks):
        return {
            "classification": CLASSIFICATION_BLOCKED,
            "reason": "A required synthetic table or fixture input is missing.",
            "recommended_action": "Fix synthetic fixture availability before interpreting schema readiness.",
        }
    if any(check["status"] == "fail" for check in checks):
        return {
            "classification": CLASSIFICATION_NEEDS_FIXES,
            "reason": "Required schema, join-key, timestamp, label, duplicate-key, or value checks failed.",
            "recommended_action": "Fix the synthetic schema contract before using the validator against approved extracts.",
        }
    if any(check["status"] == "warning" for check in checks):
        return {
            "classification": CLASSIFICATION_WARNINGS,
            "reason": "Schema is usable but warning-level join or timestamp semantics need review.",
            "recommended_action": "Review warnings before any private-data intake approval.",
        }
    return {
        "classification": CLASSIFICATION_READY,
        "reason": "Synthetic schema fixture passed required table, join, timestamp, label, duplicate-key, and value checks.",
        "recommended_action": "Record readiness; no private-data intake or model action is authorized.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.use_built_in_synthetic_fixture is not None:
        tables = built_in_synthetic_fixture(args.use_built_in_synthetic_fixture)
        source_kind = f"built_in:{args.use_built_in_synthetic_fixture}"
    else:
        tables = load_fixture_root(args.synthetic_fixture_root)
        source_kind = f"synthetic_fixture_root:{args.synthetic_fixture_root}"
    report = build_company_data_schema_report(tables, source_kind=source_kind, synthetic_data=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["decision"]["classification"])
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate synthetic company-data intake schemas and join keys without private data ingestion."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--use-built-in-synthetic-fixture", choices=BUILT_IN_FIXTURES)
    source.add_argument("--synthetic-fixture-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _valid_minimal_tables() -> dict[str, list[dict[str, str]]]:
    return deepcopy(
        {
            "orders": [
                {
                    "order_id": "order_001",
                    "created_at": "2026-06-11T08:00:00Z",
                    "promised_window_start": "2026-06-11T10:00:00Z",
                    "promised_window_end": "2026-06-11T12:00:00Z",
                    "order_priority": "premium",
                    "order_size": "3",
                    "zone_id": "zone_a",
                    "customer_id": "customer_hash_001",
                    "customer_location_id": "location_hash_001",
                    "latitude": "41.0000",
                    "longitude": "29.0000",
                    "sku_id": "sku_001",
                    "site_id": "site_001",
                }
            ],
            "dispatch_attempts": [
                {
                    "dispatch_attempt_id": "dispatch_001",
                    "order_id": "order_001",
                    "decision_timestamp": "2026-06-11T08:05:00Z",
                    "dispatch_attempt_timestamp": "2026-06-11T08:06:00Z",
                    "dispatch_status": "success",
                    "dispatch_failure_reason": "none",
                    "action_id": "24",
                    "vehicle_id": "vehicle_001",
                    "carrier_id": "carrier_001",
                    "fleet_type": "secondary",
                    "route_id": "route_001",
                    "no_vehicle_flag": "0",
                    "already_assigned_flag": "0",
                    "route_failure_flag": "0",
                    "carrier_rejected_flag": "0",
                    "duplicate_assignment_flag": "0",
                    "policy_version": "synthetic_hierarchical_v1",
                }
            ],
            "deliveries": [
                {
                    "order_id": "order_001",
                    "dispatch_attempt_id": "dispatch_001",
                    "pickup_timestamp": "2026-06-11T08:30:00Z",
                    "delivery_timestamp": "2026-06-11T10:30:00Z",
                    "delivery_status": "delivered",
                    "delivered_on_time": "true",
                    "failed_attempt_reason": "",
                    "lateness_minutes": "0",
                    "actual_sequence_rank": "1",
                }
            ],
            "routes": [
                {
                    "route_id": "route_001",
                    "order_id": "order_001",
                    "planned_route_type": "shortest",
                    "planned_distance": "12.5",
                    "planned_travel_time": "55.0",
                    "route_failure_flag": "0",
                    "actual_route_id": "route_actual_001",
                    "actual_distance": "12.8",
                    "actual_travel_time": "58.0",
                    "planned_sequence_rank": "1",
                    "actual_sequence_rank": "1",
                    "reroute_flag": "0",
                    "congestion_delay_flag": "0",
                    "route_provider": "synthetic_provider",
                }
            ],
            "fleet": [
                {
                    "vehicle_id": "vehicle_001",
                    "carrier_id": "carrier_001",
                    "fleet_type": "secondary",
                    "vehicle_capacity": "12",
                    "available_at_decision": "2026-06-11T08:04:00Z",
                    "assigned_at": "2026-06-11T08:06:00Z",
                    "fleet_cost": "42.50",
                    "accepted_at": "2026-06-11T08:07:00Z",
                    "cancelled_at": "",
                    "fleet_reliability_score": "0.98",
                }
            ],
            "inventory": [
                {
                    "sku_id": "sku_001",
                    "site_id": "site_001",
                    "order_id": "order_001",
                    "stock_on_hand_at_decision": "20",
                    "backlog_quantity": "0",
                    "stockout_flag": "0",
                    "supplier_lead_time": "2",
                    "allocated_quantity": "3",
                    "reorder_event_id": "",
                    "reorder_type": "none",
                    "supplier_id": "supplier_001",
                }
            ],
            "costs": [
                {
                    "order_id": "order_001",
                    "primary_fleet_cost": "28.00",
                    "secondary_fleet_cost": "42.50",
                    "lateness_penalty_cost": "120.00",
                    "stockout_cost": "75.00",
                    "holding_cost": "3.50",
                    "route_cost": "8.00",
                    "emergency_reorder_cost": "0.00",
                    "cancellation_cost": "0.00",
                }
            ],
        }
    )


def _normalize_tables(tables: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, str]]]:
    normalized: dict[str, list[dict[str, str]]] = {}
    for table_name, rows in tables.items():
        normalized[table_name] = [{str(key): "" if value is None else str(value) for key, value in row.items()} for row in rows]
    return normalized


def _build_table_summaries(tables: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for table_name in TABLE_ORDER:
        rows = tables.get(table_name, [])
        columns = _columns_for_rows(rows)
        duplicate_keys = _duplicate_keys(rows, PRIMARY_KEYS[table_name])
        summaries[table_name] = {
            "rows": len(rows),
            "required_columns_present": all(column in columns for column in TABLE_SCHEMAS[table_name]["required"]),
            "optional_columns_present": [column for column in TABLE_SCHEMAS[table_name]["optional"] if column in columns],
            "duplicate_primary_keys": len(duplicate_keys),
            "issues": [],
        }
    return summaries


def _attach_table_issues(
    table_summaries: dict[str, dict[str, Any]],
    data_quality_checks: list[dict[str, Any]],
    join_key_checks: list[dict[str, Any]],
) -> None:
    for check in data_quality_checks:
        table = check.get("table")
        if table in table_summaries and check["status"] != "pass":
            table_summaries[table]["issues"].append(check["check_id"])
    for check in join_key_checks:
        left_table = check.get("left_table")
        if left_table in table_summaries and check["status"] != "pass":
            table_summaries[left_table]["issues"].append(check["check_id"])


def _columns_for_rows(rows: list[dict[str, str]]) -> set[str]:
    columns: set[str] = set()
    for row in rows:
        columns.update(row)
    return columns


def _duplicate_keys(rows: list[dict[str, str]], primary_key: list[str]) -> list[str]:
    seen: set[tuple[str, ...]] = set()
    duplicates: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in primary_key)
        if key in seen:
            duplicates.add(key)
        else:
            seen.add(key)
    return ["|".join(key) for key in sorted(duplicates)]


def _parse_timestamp(value: str) -> datetime | None:
    if value == "":
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_number(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _status_message(status: str, pass_message: str) -> str:
    if status == "pass":
        return pass_message
    if status == "warning":
        return "Warning-level schema issue found."
    if status == "blocked":
        return "Required fixture input is blocked."
    return "Required schema check failed."


if __name__ == "__main__":
    raise SystemExit(main())
