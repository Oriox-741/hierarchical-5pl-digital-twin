"""Fleet/dispatch public benchmark helpers for the 2026-06-14 upgrade sprint.

The helpers consume caller-provided public trip rows and write fresh reports
only. They do not touch model artifacts, registry state, production copies, DB,
checkpoints, or existing eval outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
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

DEFAULT_CAPACITY_VARIANTS: dict[str, tuple[int, float]] = {
    "low_capacity": (1, 30.0),
    "balanced_capacity": (2, 25.0),
    "high_capacity": (4, 20.0),
}


def summarize_hvfhs_dispatch(
    rows: Sequence[Mapping[str, Any]],
    *,
    capacity_variants: Mapping[str, tuple[int, float]] | None = None,
) -> dict[str, Any]:
    parsed = [_parse_trip_row(row, schema="hvfhs") for row in rows]
    parsed = [row for row in parsed if row is not None]
    request_to_pickup = [
        (row["pickup_time"] - row["request_time"]).total_seconds()
        for row in parsed
        if row.get("request_time") is not None and row.get("pickup_time") is not None
    ]
    request_to_onscene = [
        (row["on_scene_time"] - row["request_time"]).total_seconds()
        for row in parsed
        if row.get("request_time") is not None and row.get("on_scene_time") is not None
    ]
    base_waits: defaultdict[str, list[float]] = defaultdict(list)
    for row, wait in zip(parsed, request_to_pickup):
        base = row.get("base") or "unknown"
        base_waits[str(base)].append(wait)
    return {
        "trip_rows": len(parsed),
        "base_count": len({str(row.get("base")) for row in parsed if row.get("base")}),
        "request_to_pickup_wait_seconds": _numeric_summary(request_to_pickup),
        "request_to_on_scene_wait_seconds": _numeric_summary(request_to_onscene),
        "trip_miles": _numeric_summary([row["trip_miles"] for row in parsed if row.get("trip_miles") is not None]),
        "peak_offpeak_counts": _peak_offpeak_counts(row["pickup_time"] for row in parsed if row.get("pickup_time") is not None),
        "zone_hour_demand_pressure": _zone_hour_pressure(parsed),
        "base_wait_seconds": {base: _numeric_summary(values) for base, values in sorted(base_waits.items())},
        "dispatch_capacity_proxy": _capacity_proxy(parsed, capacity_variants or DEFAULT_CAPACITY_VARIANTS),
    }


def summarize_city_trip_dispatch(
    rows: Sequence[Mapping[str, Any]],
    *,
    capacity_variants: Mapping[str, tuple[int, float]] | None = None,
) -> dict[str, Any]:
    parsed = [_parse_trip_row(row, schema="city_trip") for row in rows]
    parsed = [row for row in parsed if row is not None]
    return {
        "trip_rows": len(parsed),
        "trip_seconds": _numeric_summary([row["trip_seconds"] for row in parsed if row.get("trip_seconds") is not None]),
        "trip_miles": _numeric_summary([row["trip_miles"] for row in parsed if row.get("trip_miles") is not None]),
        "fare_proxy": _numeric_summary([row["fare"] for row in parsed if row.get("fare") is not None]),
        "zone_hour_demand_pressure": _zone_hour_pressure(parsed),
        "peak_offpeak_counts": _peak_offpeak_counts(row["pickup_time"] for row in parsed if row.get("pickup_time") is not None),
        "dispatch_capacity_proxy": _capacity_proxy(parsed, capacity_variants or DEFAULT_CAPACITY_VARIANTS),
    }


def summarize_vehicle_fleet_proxy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parsed = [_parse_trip_row(row, schema="vehicle_trip") for row in rows]
    parsed = [row for row in parsed if row is not None]
    by_vehicle: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        vehicle = row.get("vehicle_id") or "unknown"
        by_vehicle[str(vehicle)].append(row)
    reposition_changes = 0
    idle_gaps: list[float] = []
    active_seconds: list[float] = []
    for vehicle_rows in by_vehicle.values():
        vehicle_rows.sort(key=lambda row: row["pickup_time"] or datetime.min)
        for row in vehicle_rows:
            if row.get("trip_seconds") is not None:
                active_seconds.append(row["trip_seconds"])
        for previous, current in zip(vehicle_rows, vehicle_rows[1:]):
            if previous.get("dropoff_zone") and current.get("pickup_zone") and previous["dropoff_zone"] != current["pickup_zone"]:
                reposition_changes += 1
            if previous.get("dropoff_time") and current.get("pickup_time"):
                gap = (current["pickup_time"] - previous["dropoff_time"]).total_seconds()
                if math.isfinite(gap) and gap >= 0:
                    idle_gaps.append(gap)
    return {
        "trip_rows": len(parsed),
        "vehicle_count": len(by_vehicle),
        "trips_per_vehicle": _numeric_summary([len(values) for values in by_vehicle.values()]),
        "trip_seconds": _numeric_summary([row["trip_seconds"] for row in parsed if row.get("trip_seconds") is not None]),
        "trip_miles": _numeric_summary([row["trip_miles"] for row in parsed if row.get("trip_miles") is not None]),
        "active_seconds_per_trip": _numeric_summary(active_seconds),
        "idle_gap_seconds": _numeric_summary(idle_gaps),
        "reposition_zone_change_proxy_count": reposition_changes,
        "reposition_zone_change_proxy_rate": reposition_changes / max(1, sum(max(0, len(v) - 1) for v in by_vehicle.values())),
    }


def summarize_taxi_trajectories(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    distances: list[float] = []
    taxi_ids = set()
    timestamps = []
    for row in rows:
        taxi = _first_value(row, ("taxi_id", "taxi", "cab_id", "vehicle_id"))
        if taxi is None:
            trajectory = _first_value(row, ("trajectory", "trajectory_id"))
            if trajectory is not None:
                taxi = str(trajectory).strip().split("_", 1)[0]
        if taxi is not None:
            taxi_ids.add(str(taxi))
        timestamp = _parse_datetime(_first_value(row, ("timestamp", "time", "trip_start_timestamp")))
        if timestamp is not None:
            timestamps.append(timestamp)
        source = _parse_point(_first_value(row, ("source_point", "start_point", "pickup_point")))
        target = _parse_point(_first_value(row, ("target_point", "end_point", "dropoff_point")))
        if source is not None and target is not None:
            distances.append(_haversine_km(source[1], source[0], target[1], target[0]))
    return {
        "trajectory_count": len(rows),
        "taxi_count": len(taxi_ids),
        "distance_km_proxy": _numeric_summary(distances),
        "time_span": {
            "start": min(timestamps).isoformat() if timestamps else None,
            "end": max(timestamps).isoformat() if timestamps else None,
        },
    }


def summarize_dynamic_routing_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dynamic_tokens = ("appear", "arrival", "request_time", "release", "dynamic")
    stochastic_tokens = ("stochastic", "delay", "accident", "prob", "variance", "sigma", "congestion")
    dynamic_fields = set()
    stochastic_fields = set()
    customer_counts: list[float] = []
    for record in records:
        for key in record:
            lower = str(key).lower()
            if any(token in lower for token in dynamic_tokens):
                dynamic_fields.add(str(key))
            if any(token in lower for token in stochastic_tokens):
                stochastic_fields.add(str(key))
        count = _finite_float(_first_value(record, ("num_customers", "customer_count", "n_customers", "n")))
        if count is None:
            locations = _first_value(record, ("locations", "coords", "customers"))
            if _is_sequence_like(locations):
                count = float(len(locations))
        if count is not None:
            customer_counts.append(count)
    return {
        "instance_count": len(records),
        "customer_count": _numeric_summary(customer_counts),
        "dynamic_fields_present": sorted(dynamic_fields),
        "stochastic_fields_present": sorted(stochastic_fields),
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


def write_report_fresh(path: Path, report: Mapping[str, Any]) -> None:
    path = Path(path)
    _reject_protected_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {path}")
    path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_trip_row(row: Mapping[str, Any], *, schema: str) -> dict[str, Any] | None:
    if schema == "hvfhs":
        pickup_time = _parse_datetime(_first_value(row, ("pickup_datetime", "pickup_time")))
        request_time = _parse_datetime(_first_value(row, ("request_datetime", "request_time")))
        on_scene_time = _parse_datetime(_first_value(row, ("on_scene_datetime", "on_scene_time")))
        dropoff_time = _parse_datetime(_first_value(row, ("dropoff_datetime", "dropoff_time")))
        pickup_zone = _first_value(row, ("PULocationID", "pulocationid", "pickup_zone", "pickup_location_id"))
        dropoff_zone = _first_value(row, ("DOLocationID", "dolocationid", "dropoff_zone", "dropoff_location_id"))
        base = _first_value(row, ("dispatching_base_num", "hvfhs_license_num", "base"))
    elif schema in {"city_trip", "vehicle_trip"}:
        pickup_time = _parse_datetime(_first_value(row, ("trip_start_timestamp", "pickup_datetime", "start_time")))
        dropoff_time = _parse_datetime(_first_value(row, ("trip_end_timestamp", "dropoff_datetime", "end_time")))
        request_time = pickup_time
        on_scene_time = None
        pickup_zone = _first_value(row, ("pickup_community_area", "pickup_census_tract", "pickup_zone", "pickup_location_id"))
        dropoff_zone = _first_value(row, ("dropoff_community_area", "dropoff_census_tract", "dropoff_zone", "dropoff_location_id"))
        base = _first_value(row, ("company", "dispatching_base_num", "shared_trip_authorized"))
    else:
        raise ValueError(schema)
    if pickup_time is None or pickup_zone is None or dropoff_zone is None:
        return None
    trip_seconds = _finite_float(_first_value(row, ("trip_seconds", "trip_time", "trip_time_seconds")))
    if trip_seconds is None and dropoff_time is not None:
        trip_seconds = (dropoff_time - pickup_time).total_seconds()
    return {
        "request_time": request_time,
        "on_scene_time": on_scene_time,
        "pickup_time": pickup_time,
        "dropoff_time": dropoff_time,
        "pickup_zone": str(pickup_zone),
        "dropoff_zone": str(dropoff_zone),
        "base": str(base) if base is not None else None,
        "vehicle_id": str(_first_value(row, ("taxi_id", "vehicle_id", "cab_id"))) if _first_value(row, ("taxi_id", "vehicle_id", "cab_id")) is not None else None,
        "trip_seconds": trip_seconds,
        "trip_miles": _finite_float(_first_value(row, ("trip_miles", "miles", "distance_miles"))),
        "fare": _finite_float(_first_value(row, ("trip_total", "fare", "fare_amount"))),
    }


def _capacity_proxy(
    parsed_rows: Sequence[Mapping[str, Any]],
    variants: Mapping[str, tuple[int, float]],
) -> dict[str, Any]:
    proxy_rows = [
        {
            "pickup_datetime": row["pickup_time"],
            "pickup_zone": row["pickup_zone"],
            "dropoff_zone": row["dropoff_zone"],
        }
        for row in parsed_rows
        if row.get("pickup_time") is not None and row.get("pickup_zone") and row.get("dropoff_zone")
    ]
    return {
        name: _simulate_zone_dispatch(proxy_rows, vehicles_per_zone=vehicles, service_minutes=service_minutes)
        for name, (vehicles, service_minutes) in variants.items()
    }


def _simulate_zone_dispatch(
    trips: Sequence[Mapping[str, Any]],
    *,
    vehicles_per_zone: int,
    service_minutes: float,
) -> dict[str, Any]:
    parsed = []
    for row in trips:
        pickup_time = _parse_datetime(_first_value(row, ("pickup_datetime", "pickup_time")))
        pickup_zone = _first_value(row, ("pickup_zone", "PULocationID", "pulocationid"))
        dropoff_zone = _first_value(row, ("dropoff_zone", "DOLocationID", "dolocationid"))
        if pickup_time is not None and pickup_zone is not None and dropoff_zone is not None:
            parsed.append((pickup_time, str(pickup_zone), str(dropoff_zone)))
    parsed.sort(key=lambda item: item[0])
    zones = sorted({zone for _, pickup, dropoff in parsed for zone in (pickup, dropoff)})
    idle_until: dict[str, list[datetime]] = {zone: [datetime.min] * int(vehicles_per_zone) for zone in zones}
    service_delta = timedelta(minutes=float(service_minutes))
    served = 0
    wait_seconds: list[float] = []
    empty_moves = 0
    for pickup_time, pickup_zone, dropoff_zone in parsed:
        fleet = idle_until.setdefault(pickup_zone, [datetime.min] * int(vehicles_per_zone))
        available_index = next((idx for idx, ready in enumerate(fleet) if ready <= pickup_time), None)
        if available_index is None:
            future_waits = [(ready - pickup_time).total_seconds() for ready in fleet if ready > pickup_time]
            if future_waits:
                wait_seconds.append(min(future_waits))
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
        "wait_proxy_seconds": _numeric_summary(wait_seconds),
        "empty_movement_proxy_rate": empty_moves / served if served else None,
    }


def _zone_hour_pressure(parsed_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in parsed_rows:
        pickup_time = row.get("pickup_time")
        pickup_zone = row.get("pickup_zone")
        if pickup_time is None or pickup_zone is None:
            continue
        bucket = pickup_time.replace(minute=0, second=0, microsecond=0).isoformat()
        counts[f"{pickup_zone}|{bucket}"] += 1
    return _numeric_summary(list(counts.values())) | {"bucket_count": len(counts)}


def _peak_offpeak_counts(times: Iterable[datetime]) -> dict[str, int]:
    peak = 0
    offpeak = 0
    for timestamp in times:
        if timestamp.hour in {7, 8, 9, 16, 17, 18, 19}:
            peak += 1
        else:
            offpeak += 1
    return {"peak": peak, "offpeak": offpeak}


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {"count": len(finite), "mean": mean(finite), "min": min(finite), "max": max(finite)}


def _field_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    columns = Counter()
    for row in rows:
        columns.update(str(key) for key in row.keys())
    return {"row_count": len(rows), "columns": sorted(columns), "column_presence": dict(columns)}


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
        return value.strip() != ""
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    for candidate in (text, text.replace("Z", "+00:00"), text.replace("/", "-")):
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed.replace(tzinfo=None)
        except ValueError:
            pass
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_point(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    match = re.search(r"POINT\s*\(\s*([-+0-9.]+)\s+([-+0-9.]+)\s*\)", str(value), flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _is_sequence_like(value: Any) -> bool:
    return hasattr(value, "__len__") and not isinstance(value, (str, bytes, bytearray, Mapping))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return value


def _reject_protected_path(path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    candidate = Path(path).resolve(strict=False)
    for prefix in PROTECTED_OUTPUT_PREFIXES:
        protected = repo_root.joinpath(*prefix).resolve(strict=False)
        if candidate == protected or protected in candidate.parents:
            raise ValueError(f"refusing to write under protected path: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fleet dispatch public benchmark helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    hvfhs = sub.add_parser("hvfhs")
    hvfhs.add_argument("--csv", type=Path, required=True)
    hvfhs.add_argument("--output", type=Path, required=True)
    hvfhs.add_argument("--limit", type=int)

    city = sub.add_parser("city-trip")
    city.add_argument("--csv", type=Path, required=True)
    city.add_argument("--output", type=Path, required=True)
    city.add_argument("--limit", type=int)

    vehicle = sub.add_parser("vehicle-fleet")
    vehicle.add_argument("--csv", type=Path, required=True)
    vehicle.add_argument("--output", type=Path, required=True)
    vehicle.add_argument("--limit", type=int)

    args = parser.parse_args(argv)
    if args.command == "hvfhs":
        report = {"decision": "NYC_HVFHS_DISPATCH_PROXY_READY", "summary": summarize_hvfhs_dispatch(read_csv_rows(args.csv, limit=args.limit))}
    elif args.command == "city-trip":
        report = {"decision": "CITY_TRIP_DISPATCH_PROXY_READY", "summary": summarize_city_trip_dispatch(read_csv_rows(args.csv, limit=args.limit))}
    elif args.command == "vehicle-fleet":
        report = {"decision": "VEHICLE_FLEET_PROXY_READY", "summary": summarize_vehicle_fleet_proxy(read_csv_rows(args.csv, limit=args.limit))}
    else:  # pragma: no cover
        raise ValueError(args.command)
    write_report_fresh(args.output, report)
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
