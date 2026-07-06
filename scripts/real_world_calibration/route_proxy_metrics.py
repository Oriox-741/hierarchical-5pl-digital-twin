"""Pure route-choice proxy metrics for public logistics datasets.

The helpers in this module are intentionally import-safe: they do not download
data, touch model artifacts, or scan the filesystem at import time. CLI usage is
limited to user-provided local data roots.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Mapping, Sequence


EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class AsymmetryMetric:
    compared_pairs: int
    mean_pair_asymmetry: float
    max_pair_asymmetry: float


@dataclass(frozen=True, slots=True)
class ConcentrationMetric:
    top_label: str | None
    top_count: int
    top_share: float
    total: int


@dataclass(frozen=True, slots=True)
class TimeWindowStress:
    package_count: int
    windowed_package_count: int
    tight_package_count: int
    tight_window_share: float
    min_window_seconds: float | None
    mean_window_seconds: float | None


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    sample_size: int
    mean: float
    lower: float
    upper: float
    iterations: int


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def route_sequence_from_rank_map(rank_map: Mapping[str, Any]) -> list[str]:
    """Return stop IDs ordered by the rank values in an Amazon sequence map."""

    ranked: list[tuple[float, str]] = []
    for stop_id, rank in rank_map.items():
        finite_rank = _finite_float(rank)
        if finite_rank is None:
            continue
        ranked.append((finite_rank, str(stop_id)))
    return [stop_id for _, stop_id in sorted(ranked)]


def sequence_travel_time(
    sequence: Sequence[str],
    travel_times: Mapping[str, Mapping[str, Any]],
) -> float:
    """Sum directed travel time along a stop sequence."""

    if len(sequence) < 2:
        return 0.0

    total = 0.0
    for origin, destination in zip(sequence, sequence[1:]):
        origin_times = travel_times[str(origin)]
        edge_time = _finite_float(origin_times[str(destination)])
        if edge_time is None:
            raise ValueError(f"non-finite travel time for edge {origin!r}->{destination!r}")
        total += edge_time
    return total


def greedy_nearest_neighbor_sequence(
    travel_times: Mapping[str, Mapping[str, Any]],
    *,
    start_stop_id: str,
) -> list[str]:
    """Build a simple fastest-like route proxy from directed travel times."""

    unvisited = {str(stop_id) for stop_id in travel_times}
    current = str(start_stop_id)
    if current not in unvisited:
        raise ValueError(f"start_stop_id {start_stop_id!r} is not present in travel_times")

    sequence = [current]
    unvisited.remove(current)
    while unvisited:
        current_times = travel_times.get(current, {})
        next_stop = min(
            unvisited,
            key=lambda stop_id: (
                _finite_float(current_times.get(stop_id))
                if _finite_float(current_times.get(stop_id)) is not None
                else math.inf,
                stop_id,
            ),
        )
        edge_time = _finite_float(current_times.get(next_stop))
        if edge_time is None:
            raise ValueError(f"missing finite travel time from {current!r} to {next_stop!r}")
        sequence.append(next_stop)
        unvisited.remove(next_stop)
        current = next_stop
    return sequence


def route_efficiency_ratio(
    *,
    actual_sequence: Sequence[str],
    reference_sequence: Sequence[str],
    travel_times: Mapping[str, Mapping[str, Any]],
) -> float:
    """Compare actual sequence travel time with a reference proxy sequence."""

    reference_time = sequence_travel_time(reference_sequence, travel_times)
    if reference_time <= 0.0:
        raise ValueError("reference sequence travel time must be positive")
    return sequence_travel_time(actual_sequence, travel_times) / reference_time


def travel_time_asymmetry(
    travel_times: Mapping[str, Mapping[str, Any]],
) -> AsymmetryMetric:
    """Compute normalized pairwise asymmetry from a directed travel-time matrix."""

    stop_ids = sorted(str(stop_id) for stop_id in travel_times)
    asymmetries: list[float] = []
    for index, origin in enumerate(stop_ids):
        for destination in stop_ids[index + 1 :]:
            forward = _finite_float(travel_times.get(origin, {}).get(destination))
            reverse = _finite_float(travel_times.get(destination, {}).get(origin))
            if forward is None or reverse is None:
                continue
            denominator = max((forward + reverse) / 2.0, EPSILON)
            asymmetries.append(abs(forward - reverse) / denominator)

    if not asymmetries:
        return AsymmetryMetric(compared_pairs=0, mean_pair_asymmetry=0.0, max_pair_asymmetry=0.0)

    return AsymmetryMetric(
        compared_pairs=len(asymmetries),
        mean_pair_asymmetry=sum(asymmetries) / len(asymmetries),
        max_pair_asymmetry=max(asymmetries),
    )


def route_concentration(counts_by_label: Mapping[str, Any]) -> ConcentrationMetric:
    """Return top-label concentration for a route/action family distribution."""

    counts = {str(label): max(int(count), 0) for label, count in counts_by_label.items()}
    total = sum(counts.values())
    if total <= 0:
        return ConcentrationMetric(top_label=None, top_count=0, top_share=0.0, total=0)
    top_label, top_count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return ConcentrationMetric(
        top_label=top_label,
        top_count=top_count,
        top_share=top_count / total,
        total=total,
    )


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


def tight_time_window_stress(
    route_package_data: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    tight_window_seconds: float = 2 * 60 * 60,
) -> TimeWindowStress:
    """Summarize how much package data is under tight delivery windows."""

    package_count = 0
    window_lengths: list[float] = []

    for packages_by_id in route_package_data.values():
        if not isinstance(packages_by_id, Mapping):
            continue
        for package in packages_by_id.values():
            package_count += 1
            package_map = package if isinstance(package, Mapping) else {}
            time_window = package_map.get("time_window", {})
            if not isinstance(time_window, Mapping):
                continue
            start = _parse_amazon_datetime(time_window.get("start_time_utc"))
            end = _parse_amazon_datetime(time_window.get("end_time_utc"))
            if start is None or end is None:
                continue
            window_seconds = (end - start).total_seconds()
            if window_seconds > 0 and math.isfinite(window_seconds):
                window_lengths.append(float(window_seconds))

    windowed_count = len(window_lengths)
    tight_count = sum(1 for seconds in window_lengths if seconds <= tight_window_seconds)
    tight_share = tight_count / windowed_count if windowed_count else 0.0
    return TimeWindowStress(
        package_count=package_count,
        windowed_package_count=windowed_count,
        tight_package_count=tight_count,
        tight_window_share=tight_share,
        min_window_seconds=min(window_lengths) if window_lengths else None,
        mean_window_seconds=sum(window_lengths) / windowed_count if windowed_count else None,
    )


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    iterations: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> BootstrapInterval:
    """Return a deterministic percentile bootstrap interval for a sample mean."""

    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        raise ValueError("bootstrap_mean_interval requires at least one finite value")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    rng = random.Random(seed)
    sample_size = len(finite_values)
    boot_means = []
    for _ in range(iterations):
        sample = [finite_values[rng.randrange(sample_size)] for _ in range(sample_size)]
        boot_means.append(mean(sample))
    boot_means.sort()

    lower_index = max(0, min(iterations - 1, int((alpha / 2.0) * iterations)))
    upper_index = max(0, min(iterations - 1, int((1.0 - alpha / 2.0) * iterations) - 1))
    return BootstrapInterval(
        sample_size=sample_size,
        mean=mean(finite_values),
        lower=boot_means[lower_index],
        upper=boot_means[upper_index],
        iterations=iterations,
    )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _first_existing(root: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    for candidate in root.rglob("*.json"):
        if candidate.name in names:
            return candidate
    return None


def summarize_amazon_route_root(data_root: Path, *, route_limit: int = 25) -> dict[str, Any]:
    """Summarize route-proxy metrics when Amazon Last Mile JSON files are local."""

    actual_path = _first_existing(data_root, ("actual_sequences.json", "new_actual_sequences.json"))
    package_path = _first_existing(data_root, ("package_data.json", "new_package_data.json"))
    travel_path = _first_existing(data_root, ("travel_times.json", "new_travel_times.json"))
    route_path = _first_existing(data_root, ("route_data.json", "new_route_data.json"))
    invalid_scores_path = _first_existing(
        data_root, ("invalid_sequence_scores.json", "new_invalid_sequence_scores.json")
    )

    present = {
        "actual_sequences": str(actual_path) if actual_path else None,
        "package_data": str(package_path) if package_path else None,
        "travel_times": str(travel_path) if travel_path else None,
        "route_data": str(route_path) if route_path else None,
        "invalid_sequence_scores": str(invalid_scores_path) if invalid_scores_path else None,
    }
    if actual_path is None or travel_path is None:
        return {"data_root": str(data_root), "present_files": present, "route_summaries": []}

    actual_sequences = _load_json(actual_path)
    travel_times = _load_json(travel_path)
    package_data = _load_json(package_path) if package_path else {}
    route_data = _load_json(route_path) if route_path else {}
    invalid_scores = _load_json(invalid_scores_path) if invalid_scores_path else {}

    route_summaries: list[dict[str, Any]] = []
    for route_id, sequence_payload in list(actual_sequences.items())[:route_limit]:
        sequence_map = sequence_payload.get("actual", {}) if isinstance(sequence_payload, Mapping) else {}
        if not isinstance(sequence_map, Mapping):
            continue
        sequence = route_sequence_from_rank_map(sequence_map)
        route_travel_times = travel_times.get(route_id, {})
        if not isinstance(route_travel_times, Mapping):
            continue
        summary: dict[str, Any] = {
            "route_id": route_id,
            "stop_count": len(sequence),
            "actual_sequence_travel_time_seconds": None,
            "travel_time_asymmetry": asdict(travel_time_asymmetry(route_travel_times)),
        }
        if len(sequence) >= 2:
            try:
                summary["actual_sequence_travel_time_seconds"] = sequence_travel_time(
                    sequence, route_travel_times
                )
            except (KeyError, ValueError):
                summary["actual_sequence_travel_time_seconds"] = None
            if isinstance(route_data, Mapping) and isinstance(route_data.get(route_id), Mapping):
                route_payload = route_data[route_id]
                stops = route_payload.get("stops", {})
                station_stop_id = None
                if isinstance(stops, Mapping):
                    for stop_id, stop_payload in stops.items():
                        if isinstance(stop_payload, Mapping) and stop_payload.get("type") == "Station":
                            station_stop_id = str(stop_id)
                            break
                if station_stop_id is not None:
                    try:
                        greedy_sequence = greedy_nearest_neighbor_sequence(
                            route_travel_times,
                            start_stop_id=station_stop_id,
                        )
                        summary["greedy_travel_time_seconds"] = sequence_travel_time(
                            greedy_sequence, route_travel_times
                        )
                        summary["actual_to_greedy_travel_time_ratio"] = route_efficiency_ratio(
                            actual_sequence=sequence,
                            reference_sequence=greedy_sequence,
                            travel_times=route_travel_times,
                        )
                    except (KeyError, ValueError):
                        summary["greedy_travel_time_seconds"] = None
                        summary["actual_to_greedy_travel_time_ratio"] = None
        if isinstance(package_data, Mapping) and isinstance(package_data.get(route_id), Mapping):
            summary["tight_time_window_stress"] = tight_time_window_stress(
                package_data[route_id]
            )
            summary["tight_time_window_stress"] = asdict(summary["tight_time_window_stress"])
        if isinstance(route_data, Mapping) and isinstance(route_data.get(route_id), Mapping):
            summary["route_score"] = route_data[route_id].get("route_score")
            summary["station_code"] = route_data[route_id].get("station_code")
        if isinstance(invalid_scores, Mapping) and route_id in invalid_scores:
            summary["invalid_sequence_score"] = invalid_scores.get(route_id)
        route_summaries.append(summary)

    return {"data_root": str(data_root), "present_files": present, "route_summaries": route_summaries}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize local Amazon Last Mile route-proxy metrics without downloading data."
        )
    )
    parser.add_argument("data_root", type=Path, help="Local Amazon Last Mile data root.")
    parser.add_argument("--route-limit", type=int, default=25, help="Maximum routes to summarize.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = summarize_amazon_route_root(args.data_root, route_limit=max(args.route_limit, 0))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
