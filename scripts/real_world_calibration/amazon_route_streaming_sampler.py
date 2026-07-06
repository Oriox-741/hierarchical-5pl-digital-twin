"""Streaming Amazon Last Mile route-proxy analyzer.

The full training travel-time file is large, so this script streams top-level
route objects instead of eager-loading the dataset.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ijson

from scripts.real_world_calibration.route_proxy_metrics import (
    greedy_nearest_neighbor_sequence,
    route_efficiency_ratio,
    route_sequence_from_rank_map,
    sequence_travel_time,
    travel_time_asymmetry,
)
from src.eval.benchmark_statistics import bootstrap_mean_ci, finite_values, summary_stats


EXPECTED_BUILD_FILES = {
    "actual_sequences": "actual_sequences.json",
    "invalid_sequence_scores": "invalid_sequence_scores.json",
    "package_data": "package_data.json",
    "route_data": "route_data.json",
    "travel_times": "travel_times.json",
}
DEFAULT_SMALL_SAMPLE_REPORT = Path("reports/benchmarks/amazon_route_proxy_20260613/amazon_route_proxy_summary.json")


def stream_top_level_json_items(path: Path) -> list[tuple[str, Any]] | Any:
    """Yield top-level ``(key, value)`` JSON object items from ``path``."""

    with Path(path).open("rb") as handle:
        yield from ijson.kvitems(_NanToNullReader(handle), "", use_float=True)


class _NanToNullReader:
    """Binary reader that normalizes Amazon's non-standard NaN JSON tokens."""

    _TAIL_BYTES = 2

    def __init__(self, raw: Any) -> None:
        self._raw = raw
        self._tail = b""
        self._buffer = b""
        self._eof = False

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size is None or size < 0:
            data = self._buffer + self._tail + self._raw.read()
            self._buffer = b""
            self._tail = b""
            self._eof = True
            return data.replace(b"NaN", b"null")

        while len(self._buffer) < size and not self._eof:
            chunk = self._raw.read(max(size, 65536))
            if not chunk:
                self._eof = True
                self._buffer += self._tail.replace(b"NaN", b"null")
                self._tail = b""
                break
            data = self._tail + chunk
            normalized = data.replace(b"NaN", b"null")
            if len(normalized) <= self._TAIL_BYTES:
                self._tail = normalized
                continue
            self._buffer += normalized[: -self._TAIL_BYTES]
            self._tail = normalized[-self._TAIL_BYTES :]

        result = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return result


def analyze_amazon_full_route_proxy(
    *,
    data_root: Path,
    output_dir: Path,
    previous_sample_report_path: Path | None = DEFAULT_SMALL_SAMPLE_REPORT,
) -> dict[str, Any]:
    paths = find_required_files(data_root)
    missing = sorted(name for name, path in paths.items() if path is None)
    if missing:
        raise FileNotFoundError(f"missing required Amazon files: {', '.join(missing)}")
    concrete_paths = {name: path for name, path in paths.items() if path is not None}

    _prepare_output_dir(output_dir)
    actual_sequences = _load_json(concrete_paths["actual_sequences"])
    invalid_scores = _load_json(concrete_paths["invalid_sequence_scores"])
    route_metadata = _stream_route_metadata(concrete_paths["route_data"])
    package_summaries = _stream_package_summaries(concrete_paths["package_data"])

    per_route_path = output_dir / "per_route_metrics.jsonl"
    route_rows: list[dict[str, Any]] = []
    with per_route_path.open("w", encoding="utf-8") as handle:
        for route_id, travel_times in stream_top_level_json_items(concrete_paths["travel_times"]):
            if route_id not in actual_sequences or route_id not in route_metadata:
                continue
            row = _analyze_route(
                route_id=str(route_id),
                route_metadata=route_metadata[str(route_id)],
                package_summary=package_summaries.get(str(route_id), _empty_package_summary()),
                actual_payload=actual_sequences[str(route_id)],
                invalid_sequence_score=invalid_scores.get(str(route_id))
                if isinstance(invalid_scores, Mapping)
                else None,
                travel_times=travel_times,
            )
            if row is None:
                continue
            route_rows.append(row)
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")

    report = _build_report(
        data_root=data_root,
        input_files=concrete_paths,
        route_rows=route_rows,
        route_metadata=route_metadata,
        package_summaries=package_summaries,
        previous_sample_report_path=previous_sample_report_path,
    )
    _write_json(output_dir / "amazon_full_route_proxy_report.json", report)
    _write_route_summary_csv(output_dir / "amazon_full_route_proxy_summary.csv", route_rows)
    return report


def find_required_files(data_root: Path) -> dict[str, Path | None]:
    root = Path(data_root)
    build_dir = root / "almrrc2021-data-training" / "model_build_inputs"
    paths: dict[str, Path | None] = {}
    for logical_name, filename in EXPECTED_BUILD_FILES.items():
        direct = build_dir / filename
        if direct.exists():
            paths[logical_name] = direct
            continue
        root_direct = root / filename
        if root_direct.exists():
            paths[logical_name] = root_direct
            continue
        matches = list(root.rglob(filename)) if root.exists() else []
        paths[logical_name] = matches[0] if matches else None
    return paths


def _stream_route_metadata(route_data_path: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for route_id, route_payload in stream_top_level_json_items(route_data_path):
        if not isinstance(route_payload, Mapping):
            continue
        stops = route_payload.get("stops", {})
        stop_count = len(stops) if isinstance(stops, Mapping) else 0
        station_stop_id = None
        if isinstance(stops, Mapping):
            for stop_id, stop_payload in stops.items():
                if isinstance(stop_payload, Mapping) and stop_payload.get("type") == "Station":
                    station_stop_id = str(stop_id)
                    break
        metadata[str(route_id)] = {
            "station_code": route_payload.get("station_code"),
            "date_YYYY_MM_DD": route_payload.get("date_YYYY_MM_DD"),
            "departure_time_utc": route_payload.get("departure_time_utc"),
            "route_score": route_payload.get("route_score"),
            "stop_count": stop_count,
            "station_stop_id": station_stop_id,
        }
    return metadata


def _stream_package_summaries(package_data_path: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for route_id, route_package_data in stream_top_level_json_items(package_data_path):
        summaries[str(route_id)] = _package_summary(route_package_data)
    return summaries


def _package_summary(route_package_data: Any) -> dict[str, Any]:
    package_count = 0
    window_lengths: list[float] = []
    bucket_counts = Counter({"missing": 0, "le_2h": 0, "le_4h": 0, "le_8h": 0, "gt_8h": 0})
    if not isinstance(route_package_data, Mapping):
        return _empty_package_summary()

    for packages_by_id in route_package_data.values():
        if not isinstance(packages_by_id, Mapping):
            continue
        for package in packages_by_id.values():
            package_count += 1
            package_map = package if isinstance(package, Mapping) else {}
            time_window = package_map.get("time_window", {})
            if not isinstance(time_window, Mapping):
                bucket_counts["missing"] += 1
                continue
            start = _parse_amazon_datetime(time_window.get("start_time_utc"))
            end = _parse_amazon_datetime(time_window.get("end_time_utc"))
            if start is None or end is None:
                bucket_counts["missing"] += 1
                continue
            seconds = (end - start).total_seconds()
            if not math.isfinite(seconds) or seconds <= 0.0:
                bucket_counts["missing"] += 1
                continue
            window_lengths.append(float(seconds))
            if seconds <= 2 * 60 * 60:
                bucket_counts["le_2h"] += 1
            elif seconds <= 4 * 60 * 60:
                bucket_counts["le_4h"] += 1
            elif seconds <= 8 * 60 * 60:
                bucket_counts["le_8h"] += 1
            else:
                bucket_counts["gt_8h"] += 1

    return {
        "package_count": package_count,
        "windowed_package_count": len(window_lengths),
        "tight_window_count_le_2h": bucket_counts["le_2h"],
        "time_window_bucket_counts": dict(bucket_counts),
        "min_window_seconds": min(window_lengths) if window_lengths else None,
        "mean_window_seconds": sum(window_lengths) / len(window_lengths) if window_lengths else None,
    }


def _analyze_route(
    *,
    route_id: str,
    route_metadata: Mapping[str, Any],
    package_summary: Mapping[str, Any],
    actual_payload: Any,
    invalid_sequence_score: Any,
    travel_times: Any,
) -> dict[str, Any] | None:
    if not isinstance(travel_times, Mapping):
        return None
    sequence_map = actual_payload.get("actual", {}) if isinstance(actual_payload, Mapping) else {}
    if not isinstance(sequence_map, Mapping):
        return None
    actual_sequence = route_sequence_from_rank_map(sequence_map)
    station_stop_id = route_metadata.get("station_stop_id")
    if station_stop_id is None or len(actual_sequence) < 2:
        return None

    try:
        greedy_sequence = greedy_nearest_neighbor_sequence(travel_times, start_stop_id=str(station_stop_id))
        actual_time = sequence_travel_time(actual_sequence, travel_times)
        greedy_time = sequence_travel_time(greedy_sequence, travel_times)
        ratio = route_efficiency_ratio(
            actual_sequence=actual_sequence,
            reference_sequence=greedy_sequence,
            travel_times=travel_times,
        )
    except (KeyError, ValueError):
        return None

    asymmetry = travel_time_asymmetry(travel_times)
    return {
        "route_id": route_id,
        "station_code": route_metadata.get("station_code"),
        "date_YYYY_MM_DD": route_metadata.get("date_YYYY_MM_DD"),
        "route_score": route_metadata.get("route_score"),
        "stop_count": route_metadata.get("stop_count"),
        "package_count": package_summary.get("package_count", 0),
        "windowed_package_count": package_summary.get("windowed_package_count", 0),
        "tight_window_count_le_2h": package_summary.get("tight_window_count_le_2h", 0),
        "time_window_bucket_counts": package_summary.get("time_window_bucket_counts", {}),
        "actual_sequence_travel_time_seconds": actual_time,
        "greedy_travel_time_seconds": greedy_time,
        "actual_to_greedy_travel_time_ratio": ratio,
        "travel_time_asymmetry_mean_pair": asymmetry.mean_pair_asymmetry,
        "travel_time_asymmetry_max_pair": asymmetry.max_pair_asymmetry,
        "travel_time_asymmetry_compared_pairs": asymmetry.compared_pairs,
        "invalid_sequence_score": _finite_or_raw(invalid_sequence_score),
    }


def _build_report(
    *,
    data_root: Path,
    input_files: Mapping[str, Path],
    route_rows: Sequence[Mapping[str, Any]],
    route_metadata: Mapping[str, Mapping[str, Any]],
    package_summaries: Mapping[str, Mapping[str, Any]],
    previous_sample_report_path: Path | None,
) -> dict[str, Any]:
    route_score_distribution = Counter(str(row.get("route_score")) for row in route_rows if row.get("route_score"))
    station_distribution = Counter(str(row.get("station_code")) for row in route_rows if row.get("station_code"))
    bucket_totals: Counter[str] = Counter()
    for row in route_rows:
        buckets = row.get("time_window_bucket_counts", {})
        if isinstance(buckets, Mapping):
            for key, value in buckets.items():
                bucket_totals[str(key)] += int(value)

    ratios = [row.get("actual_to_greedy_travel_time_ratio") for row in route_rows]
    asymmetry_mean = [row.get("travel_time_asymmetry_mean_pair") for row in route_rows]
    asymmetry_max = [row.get("travel_time_asymmetry_max_pair") for row in route_rows]
    invalid_scores = [row.get("invalid_sequence_score") for row in route_rows]
    package_count = sum(int(row.get("package_count", 0) or 0) for row in route_rows)
    windowed_count = sum(int(row.get("windowed_package_count", 0) or 0) for row in route_rows)
    tight_count = sum(int(row.get("tight_window_count_le_2h", 0) or 0) for row in route_rows)

    report = {
        "decision": "AMAZON_FULL_ROUTE_PROXY_READY" if route_rows else "AMAZON_FULL_ROUTE_PROXY_BLOCKED",
        "data_root": str(data_root),
        "input_files": _file_inventory(input_files),
        "coverage": {
            "routes_in_route_data": len(route_metadata),
            "routes_in_package_data": len(package_summaries),
            "routes_processed": len(route_rows),
            "package_count_processed": package_count,
            "windowed_package_count": windowed_count,
            "tight_window_count_le_2h": tight_count,
            "tight_window_share_le_2h": tight_count / windowed_count if windowed_count else 0.0,
        },
        "route_score_distribution": dict(route_score_distribution),
        "station_distribution": dict(station_distribution),
        "station_concentration": _concentration(station_distribution),
        "time_window_bucket_counts": dict(bucket_totals),
        "actual_to_greedy_travel_time_ratio": _stats_with_ci(ratios),
        "travel_time_asymmetry_mean_pair": _stats_with_ci(asymmetry_mean),
        "travel_time_asymmetry_max_pair": _stats_with_ci(asymmetry_max),
        "invalid_sequence_score": _stats_with_ci(invalid_scores),
        "stop_count": _stats_with_ci([row.get("stop_count") for row in route_rows]),
        "comparison_to_13_route_sample": _compare_to_previous_sample(previous_sample_report_path, route_rows),
        "route_side_interpretation": {
            "action_24_shortest_component": (
                "Full Amazon training routes compare actual driver sequences to a greedy "
                "fastest/nearest-neighbor proxy. Ratios near 1 support shortest-like route "
                "preferences as directionally plausible, but not universally optimal."
            ),
            "action_32_low_congestion_component": (
                "Directed travel-time asymmetry across full training routes supports a "
                "reliability/low-congestion proxy under disruption. No live congestion labels "
                "are present, so this remains route-side proxy evidence."
            ),
            "strict_limits": (
                "Amazon public data does not validate secondary_fleet, reorder_none, company "
                "carrier economics, inventory economics, or private dispatch failure reasons."
            ),
        },
    }
    return report


def _stats_with_ci(values: Sequence[Any]) -> dict[str, Any]:
    sample = finite_values(values)
    if not sample:
        return {"count": 0}
    stats = summary_stats(sample)
    stats["bootstrap_mean_ci"] = bootstrap_mean_ci(sample, iterations=1000, seed=42)
    return stats


def _compare_to_previous_sample(path: Path | None, route_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False}
    previous = json.loads(path.read_text(encoding="utf-8"))
    full_ratio = _stats_with_ci([row.get("actual_to_greedy_travel_time_ratio") for row in route_rows])
    full_asymmetry = _stats_with_ci([row.get("travel_time_asymmetry_mean_pair") for row in route_rows])
    return {
        "available": True,
        "previous_route_count": previous.get("route_count"),
        "full_route_count": len(route_rows),
        "previous_ratio_mean": _nested(previous, "actual_to_greedy_travel_time_ratio", "mean"),
        "full_ratio_mean": full_ratio.get("mean"),
        "previous_asymmetry_mean_pair_mean": _nested(previous, "travel_time_asymmetry_mean_pair", "mean"),
        "full_asymmetry_mean_pair_mean": full_asymmetry.get("mean"),
        "interpretation": "Full model-build data supersedes the old 13-route bounded sample for route-side proxy coverage.",
    }


def _file_inventory(paths: Mapping[str, Path]) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for name, path in sorted(paths.items()):
        inventory[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return inventory


def _write_route_summary_csv(path: Path, route_rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "route_id",
        "station_code",
        "route_score",
        "stop_count",
        "package_count",
        "windowed_package_count",
        "tight_window_count_le_2h",
        "actual_sequence_travel_time_seconds",
        "greedy_travel_time_seconds",
        "actual_to_greedy_travel_time_ratio",
        "travel_time_asymmetry_mean_pair",
        "travel_time_asymmetry_max_pair",
        "invalid_sequence_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in route_rows:
            writer.writerow({field: row.get(field) for field in fields})


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "amazon_full_route_proxy_report.json",
        "amazon_full_route_proxy_summary.csv",
        "per_route_metrics.jsonl",
    ):
        path = output_dir / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing Amazon route proxy output: {path}")


def _load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _parse_amazon_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() == "nan":
        return None
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None


def _empty_package_summary() -> dict[str, Any]:
    return {
        "package_count": 0,
        "windowed_package_count": 0,
        "tight_window_count_le_2h": 0,
        "time_window_bucket_counts": {},
        "min_window_seconds": None,
        "mean_window_seconds": None,
    }


def _concentration(counter: Mapping[str, int]) -> dict[str, Any]:
    total = sum(int(value) for value in counter.values())
    if total <= 0:
        return {"top_label": None, "top_count": 0, "top_share": 0.0, "total": 0}
    top_label, top_count = max(counter.items(), key=lambda item: (int(item[1]), item[0]))
    return {"top_label": top_label, "top_count": int(top_count), "top_share": int(top_count) / total, "total": total}


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _finite_or_raw(value: Any) -> Any:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return numeric if math.isfinite(numeric) else value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream full Amazon Last Mile route-proxy metrics.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-sample-report", type=Path, default=DEFAULT_SMALL_SAMPLE_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = analyze_amazon_full_route_proxy(
        data_root=args.data_root,
        output_dir=args.output_dir,
        previous_sample_report_path=args.previous_sample_report,
    )
    print(report["decision"])
    print(json.dumps(report["coverage"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
