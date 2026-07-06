"""Public-data benchmark helpers for the 2026-06-14 expansion sprint.

The module is deliberately lightweight: it parses caller-provided public data
files, computes proxy metrics, and writes fresh reports only. It never touches
model artifacts or protected production paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


PROTECTED_OUTPUT_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("models", "registry"),
    ("models", "production"),
    ("models", "baselines"),
    ("models", "checkpoints"),
    ("models", "eval"),
    ("db",),
)


def summarize_lade_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    package_ids = _distinct(rows, ("package_id", "pkg_id", "task_id", "waybill_id", "order_id"))
    courier_ids = _distinct(rows, ("courier_id", "courier", "courier_idx", "worker_id"))
    cities = _distinct(rows, ("city", "city_id", "region", "region_id"))
    latencies = _latencies(
        rows,
        start_fields=("accept_time", "accept_time_utc", "accept_ts", "accept_timestamp", "accept_gps_time"),
        end_fields=("finish_time", "finish_time_utc", "finish_ts", "finish_timestamp", "delivery_time"),
    )
    return {
        "package_count": len(package_ids) if package_ids else len(rows),
        "row_count": len(rows),
        "courier_count": len(courier_ids),
        "city_count": len(cities),
        "cities": sorted(cities)[:25],
        "accept_to_finish_latency_seconds": _numeric_summary(latencies),
        "field_summary": _field_summary(rows),
    }


def summarize_planned_actual_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deviations: list[float] = []
    absolute_rank_delta: list[float] = []
    route_ids = set()
    driver_ids = set()
    location_ids = set()
    for row in rows:
        route_id = _first_value(row, ("route_id", "planned_route_id", "RouteID", "route"))
        if route_id is not None:
            route_ids.add(str(route_id))
        driver = _first_value(row, ("driver_id", "courier_id", "vehicle_id", "DriverID"))
        if driver is not None:
            driver_ids.add(str(driver))
        planned = _sequence_from_row(row, ("planned_sequence", "planned_stop_sequence", "planned_route", "PlannedSequence"))
        actual = _sequence_from_row(row, ("actual_sequence", "actual_stop_sequence", "driven_route", "ActualSequence"))
        location_ids.update(planned)
        location_ids.update(actual)
        if planned and actual:
            actual_pos = {stop: idx for idx, stop in enumerate(actual)}
            common = [stop for stop in planned if stop in actual_pos]
            if len(common) > 1:
                rank_delta = sum(abs(idx - actual_pos[stop]) for idx, stop in enumerate(planned) if stop in actual_pos)
                deviations.append(rank_delta / (len(common) * (len(common) - 1)))
            else:
                deviations.append(0.0 if planned == actual else 1.0)
            for idx, stop in enumerate(planned):
                if stop in actual_pos:
                    absolute_rank_delta.append(abs(idx - actual_pos[stop]))
    return {
        "route_pair_count": len(rows),
        "route_id_count": len(route_ids),
        "driver_count": len(driver_ids),
        "location_count": len(location_ids),
        "sequence_deviation": {
            "mean_position_mismatch_rate": mean(deviations) if deviations else None,
            "max_position_mismatch_rate": max(deviations) if deviations else None,
            "routes_with_deviation": sum(1 for value in deviations if value > 0.0),
        },
        "absolute_rank_delta": _numeric_summary(absolute_rank_delta),
        "field_summary": _field_summary(rows),
    }


def summarize_olist_rows(
    orders: Sequence[Mapping[str, Any]],
    order_items: Sequence[Mapping[str, Any]] | None = None,
    customers: Sequence[Mapping[str, Any]] | None = None,
    sellers: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    delivered = 0
    late = 0
    delay_days: list[float] = []
    status_counts: Counter[str] = Counter()
    for row in orders:
        status = str(_first_value(row, ("order_status",)) or "").strip()
        if status:
            status_counts[status] += 1
        delivered_at = _parse_datetime(_first_value(row, ("order_delivered_customer_date", "delivered_at")))
        estimated_at = _parse_datetime(_first_value(row, ("order_estimated_delivery_date", "estimated_delivery_at")))
        if delivered_at is None or estimated_at is None:
            continue
        delivered += 1
        delta = (delivered_at - estimated_at).total_seconds() / 86400.0
        delay_days.append(delta)
        if delta > 0:
            late += 1
    freight_values = [
        value
        for value in (_finite_float(_first_value(row, ("freight_value", "freight"))) for row in (order_items or []))
        if value is not None
    ]
    customer_states = _distinct(customers or (), ("customer_state", "state"))
    seller_states = _distinct(sellers or (), ("seller_state", "state"))
    return {
        "order_count": len(orders),
        "delivered_with_estimate_count": delivered,
        "late_delivery_rate": late / delivered if delivered else None,
        "delivery_delay_days": _numeric_summary(delay_days),
        "freight_value": _numeric_summary(freight_values),
        "status_distribution": dict(status_counts),
        "customer_state_count": len(customer_states),
        "seller_state_count": len(seller_states),
        "field_summary": {
            "orders": _field_summary(orders),
            "order_items": _field_summary(order_items or []),
            "customers": _field_summary(customers or []),
            "sellers": _field_summary(sellers or []),
        },
    }


def summarize_demand_reorder_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reorder_values = [
        value
        for value in (_finite_float(_first_value(row, ("reordered", "reorder", "is_reorder"))) for row in rows)
        if value is not None
    ]
    demand_by_key: defaultdict[str, int] = defaultdict(int)
    gaps: list[float] = []
    for row in rows:
        sku = _first_value(row, ("product_id", "item_id", "sku_id", "id"))
        store = _first_value(row, ("store_id", "site_id", "state_id", "dept_id"))
        day = _first_value(row, ("d", "day", "order_dow", "date"))
        qty = _finite_float(_first_value(row, ("sales", "demand", "quantity", "units")))
        if sku is not None and store is not None and day is not None:
            demand_by_key[f"{store}|{sku}|{day}"] += int(qty if qty is not None else 1)
        gap = _finite_float(_first_value(row, ("days_since_prior_order", "inter_order_gap", "lead_time")))
        if gap is not None:
            gaps.append(gap)
    demand_values = list(demand_by_key.values())
    return {
        "row_count": len(rows),
        "reorder_rate": (sum(reorder_values) / len(reorder_values)) if reorder_values else None,
        "inter_order_gap": _numeric_summary(gaps),
        "sku_store_day_demand": _numeric_summary(demand_values),
        "sku_store_day_count": len(demand_by_key),
        "field_summary": _field_summary(rows),
    }


def simulate_zone_dispatch_proxy(
    trips: Sequence[Mapping[str, Any]],
    *,
    vehicles_per_zone: int = 2,
    service_minutes: float = 30.0,
) -> dict[str, Any]:
    parsed = []
    for row in trips:
        pickup_time = _parse_datetime(_first_value(row, ("pickup_datetime", "tpep_pickup_datetime", "pickup_datetime_utc")))
        pickup_zone = _first_value(row, ("pickup_zone", "PULocationID", "pulocationid", "pickup_location_id"))
        dropoff_zone = _first_value(row, ("dropoff_zone", "DOLocationID", "dolocationid", "dropoff_location_id"))
        if pickup_time is None or pickup_zone is None or dropoff_zone is None:
            continue
        parsed.append((pickup_time, str(pickup_zone), str(dropoff_zone)))
    parsed.sort(key=lambda item: item[0])
    zones = sorted({zone for _, pickup, dropoff in parsed for zone in (pickup, dropoff)})
    idle_until: dict[str, list[datetime]] = {zone: [datetime.min] * int(vehicles_per_zone) for zone in zones}
    served = 0
    waits: list[float] = []
    empty_moves = 0
    service_delta = timedelta(minutes=float(service_minutes))
    for pickup_time, pickup_zone, dropoff_zone in parsed:
        fleet = idle_until.setdefault(pickup_zone, [datetime.min] * int(vehicles_per_zone))
        available_index = next((idx for idx, ready in enumerate(fleet) if ready <= pickup_time), None)
        if available_index is None:
            future_waits = [(ready - pickup_time).total_seconds() for ready in fleet if ready > pickup_time]
            if future_waits:
                waits.append(min(future_waits))
            continue
        served += 1
        fleet.pop(available_index)
        idle_until.setdefault(dropoff_zone, [])[0:0] = [pickup_time + service_delta]
        if dropoff_zone != pickup_zone:
            empty_moves += 1
    return {
        "trip_count": len(parsed),
        "zone_count": len(zones),
        "vehicles_per_zone": int(vehicles_per_zone),
        "served_demand_proxy": served,
        "served_demand_proxy_rate": served / len(parsed) if parsed else None,
        "unserved_demand_proxy": len(parsed) - served,
        "wait_proxy_seconds": _numeric_summary(waits),
        "empty_movement_proxy_rate": empty_moves / served if served else None,
    }


def summarize_svrpbench_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    customer_counts: list[float] = []
    stochastic_fields = set()
    time_window_fields = set()
    candidate_stochastic = ("congestion", "delay", "accident", "stochastic", "prob", "sigma", "variance")
    candidate_tw = ("time_window", "time_windows", "tw", "ready_time", "due_time")
    for row in records:
        for key in row:
            lower = str(key).lower()
            if any(token in lower for token in candidate_stochastic):
                stochastic_fields.add(str(key))
            if any(token in lower for token in candidate_tw):
                time_window_fields.add(str(key))
        count = _finite_float(_first_value(row, ("num_customers", "customer_count", "n_customers", "n")))
        if count is None:
            customers = _first_value(row, ("customers", "coords", "locations"))
            if _is_sequence_like(customers):
                count = float(len(customers))
        if count is not None:
            customer_counts.append(count)
    return {
        "instance_count": len(records),
        "customer_count": _numeric_summary(customer_counts),
        "stochastic_fields_present": sorted(stochastic_fields),
        "time_window_fields_present": sorted(time_window_fields),
        "field_summary": _field_summary(records),
    }


def read_csv_rows(path: Path, *, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def read_json_records(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        if all(isinstance(value, Mapping) for value in payload.values()):
            records = [dict(value, _record_id=key) for key, value in payload.items()]
        else:
            records = [payload]
    elif isinstance(payload, list):
        records = [dict(item) for item in payload if isinstance(item, Mapping)]
    else:
        records = []
    return records[:limit] if limit is not None else records


def write_report_fresh(path: Path, report: Mapping[str, Any]) -> None:
    path = Path(path)
    _reject_protected_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {path}")
    path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _latencies(
    rows: Sequence[Mapping[str, Any]],
    *,
    start_fields: Sequence[str],
    end_fields: Sequence[str],
) -> list[float]:
    values: list[float] = []
    for row in rows:
        start = _parse_datetime(_first_value(row, start_fields))
        end = _parse_datetime(_first_value(row, end_fields))
        if start is None or end is None:
            continue
        seconds = (end - start).total_seconds()
        if math.isfinite(seconds) and seconds >= 0:
            values.append(seconds)
    return values


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {"count": len(finite), "mean": mean(finite), "min": min(finite), "max": max(finite)}


def _field_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    columns = Counter()
    for row in rows:
        columns.update(str(key) for key in row.keys())
    return {"row_count": len(rows), "columns": sorted(columns), "column_presence": dict(columns)}


def _distinct(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> set[str]:
    values = set()
    for row in rows:
        value = _first_value(row, fields)
        if value is not None and str(value).strip() != "":
            values.add(str(value))
    return values


def _first_value(row: Mapping[str, Any], fields: Sequence[str]) -> Any | None:
    lower_map = {str(key).lower(): key for key in row.keys()}
    for field in fields:
        if field in row and _is_present(row[field]):
            return row[field]
        key = lower_map.get(str(field).lower())
        if key is not None and _is_present(row[key]):
            return row[key]
    return None


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if _is_nan(value):
        return False
    return True


def _sequence_from_row(row: Mapping[str, Any], fields: Sequence[str]) -> list[str]:
    value = _first_value(row, fields)
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    for sep in (">", "|", ";", ","):
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text] if text else []


def _is_sequence_like(value: Any) -> bool:
    return hasattr(value, "__len__") and not isinstance(value, (str, bytes, bytearray, Mapping))


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    for candidate in (text, text.replace("Z", "+00:00"), text.replace("/", "-")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        parsed = datetime.strptime(text, "%m-%d %H:%M:%S")
        return parsed.replace(year=2000)
    except ValueError:
        pass
    return None


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _reject_protected_path(path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    candidate = Path(path).resolve(strict=False)
    for prefix in PROTECTED_OUTPUT_PREFIXES:
        protected = repo_root.joinpath(*prefix).resolve(strict=False)
        if candidate == protected or protected in candidate.parents:
            raise ValueError(f"refusing to write under protected path: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Public data expansion benchmark helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    lade = sub.add_parser("lade")
    lade.add_argument("--csv", type=Path, required=True)
    lade.add_argument("--output", type=Path, required=True)
    lade.add_argument("--limit", type=int, default=None)

    planned = sub.add_parser("planned-actual")
    planned.add_argument("--csv", type=Path, required=True)
    planned.add_argument("--output", type=Path, required=True)
    planned.add_argument("--limit", type=int, default=None)

    olist = sub.add_parser("olist")
    olist.add_argument("--orders", type=Path, required=True)
    olist.add_argument("--items", type=Path)
    olist.add_argument("--customers", type=Path)
    olist.add_argument("--sellers", type=Path)
    olist.add_argument("--output", type=Path, required=True)
    olist.add_argument("--limit", type=int, default=None)

    demand = sub.add_parser("demand-reorder")
    demand.add_argument("--csv", type=Path, required=True)
    demand.add_argument("--output", type=Path, required=True)
    demand.add_argument("--limit", type=int, default=None)

    tlc = sub.add_parser("nyc-tlc")
    tlc.add_argument("--csv", type=Path, required=True)
    tlc.add_argument("--output", type=Path, required=True)
    tlc.add_argument("--limit", type=int, default=None)
    tlc.add_argument("--vehicles-per-zone", type=int, default=2)
    tlc.add_argument("--service-minutes", type=float, default=30.0)

    svrp = sub.add_parser("svrpbench")
    svrp.add_argument("--json", type=Path)
    svrp.add_argument("--output", type=Path, required=True)
    svrp.add_argument("--limit", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "lade":
        report = {"decision": "LADE_PUBLIC_DATA_READY", "summary": summarize_lade_rows(read_csv_rows(args.csv, limit=args.limit))}
    elif args.command == "planned-actual":
        report = {
            "decision": "PLANNED_ACTUAL_ROUTE_DATA_READY",
            "summary": summarize_planned_actual_rows(read_csv_rows(args.csv, limit=args.limit)),
        }
    elif args.command == "olist":
        report = {
            "decision": "OLIST_LOGISTICS_DATA_READY",
            "summary": summarize_olist_rows(
                read_csv_rows(args.orders, limit=args.limit),
                read_csv_rows(args.items, limit=args.limit) if args.items else [],
                read_csv_rows(args.customers, limit=args.limit) if args.customers else [],
                read_csv_rows(args.sellers, limit=args.limit) if args.sellers else [],
            ),
        }
    elif args.command == "demand-reorder":
        report = {"decision": "PUBLIC_DEMAND_REORDER_PROXY_READY", "summary": summarize_demand_reorder_rows(read_csv_rows(args.csv, limit=args.limit))}
    elif args.command == "nyc-tlc":
        report = {
            "decision": "NYC_TLC_DISPATCH_PROXY_READY",
            "summary": simulate_zone_dispatch_proxy(
                read_csv_rows(args.csv, limit=args.limit),
                vehicles_per_zone=args.vehicles_per_zone,
                service_minutes=args.service_minutes,
            ),
        }
    elif args.command == "svrpbench":
        if args.json is None:
            raise ValueError("--json is required for svrpbench command")
        report = {"decision": "SVRPBENCH_STOCHASTIC_METADATA_READY", "summary": summarize_svrpbench_records(read_json_records(args.json, limit=args.limit))}
    else:  # pragma: no cover
        raise ValueError(args.command)
    write_report_fresh(args.output, report)
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
