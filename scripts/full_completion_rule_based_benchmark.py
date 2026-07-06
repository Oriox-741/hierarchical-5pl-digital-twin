"""Resumable full rule-based benchmark runner for the completion sprint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.benchmark_statistics import (
    bootstrap_mean_ci,
    exact_sign_test_two_sided,
    finite_values,
    summary_stats,
)
from src.eval.real_world_scenario_arena import load_scenario_configs
from src.eval.rule_based_baseline_arena import (
    _episode_seed,
    _reject_protected_output_dir,
    _rate_from_distribution,
    run_rule_baseline_episode,
)
from src.eval.rule_based_baselines import build_rule_based_baselines
from src.eval.scenario_metrics import summarize_episodes
from src.learn.train_joint_torch import MDP_CONTRACT_VERSION


DEFAULT_HIERARCHICAL_SUMMARY = Path(
    "models/eval/equal_budget_hierarchical_v1_20ep_seed42_20260611/scenario_summary.json"
)
DEFAULT_HIERARCHICAL_EPISODES = Path(
    "models/eval/equal_budget_hierarchical_v1_20ep_seed42_20260611/episode_metrics.jsonl"
)


@dataclass(frozen=True, slots=True)
class BenchmarkChunk:
    baseline_name: str
    scenario_id: str


def build_chunk_plan(scenario_ids: Sequence[str], baseline_names: Sequence[str]) -> list[BenchmarkChunk]:
    return [
        BenchmarkChunk(baseline_name=baseline_name, scenario_id=scenario_id)
        for baseline_name in sorted(str(item) for item in baseline_names)
        for scenario_id in sorted(str(item) for item in scenario_ids)
    ]


def next_pending_chunk(output_dir: Path, plan: Sequence[BenchmarkChunk]) -> BenchmarkChunk | None:
    for chunk in plan:
        if not chunk_complete(output_dir, chunk):
            return chunk
    return None


def chunk_complete(output_dir: Path, chunk: BenchmarkChunk) -> bool:
    directory = chunk_dir(output_dir, chunk)
    marker = directory / "chunk_complete.json"
    summary = directory / "summary.json"
    episodes = directory / "episode_metrics.jsonl"
    return marker.exists() and summary.exists() and episodes.exists()


def chunk_dir(output_dir: Path, chunk: BenchmarkChunk) -> Path:
    return Path(output_dir) / "chunks" / _slug(chunk.baseline_name) / _slug(chunk.scenario_id)


def write_chunk_artifacts(
    *,
    output_dir: Path,
    chunk: BenchmarkChunk,
    summary: Mapping[str, Any],
    episode_rows: Sequence[Mapping[str, Any]],
) -> None:
    target = chunk_dir(output_dir, chunk)
    tmp = target.with_name(target.name + ".__tmp__")
    _remove_owned_tmp_dir(output_dir, tmp)
    tmp.mkdir(parents=True, exist_ok=False)
    (tmp / "summary.json").write_text(
        json.dumps(_jsonable(dict(summary)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (tmp / "episode_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in episode_rows:
            handle.write(json.dumps(_jsonable(dict(row)), sort_keys=True) + "\n")
    (tmp / "chunk_complete.json").write_text(
        json.dumps(
            {
                "baseline_name": chunk.baseline_name,
                "scenario_id": chunk.scenario_id,
                "episode_rows": len(episode_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if target.exists():
        raise FileExistsError(f"refusing to replace completed chunk directory: {target}")
    tmp.rename(target)


def run_missing_chunks(
    *,
    scenario_dir: Path,
    output_dir: Path,
    episodes: int,
    seed: int,
    baseline_names: Sequence[str] | None,
    max_steps: int | None,
    max_chunks: int,
) -> dict[str, Any]:
    _ensure_safe_existing_output_dir(output_dir)
    scenarios = load_scenario_configs(scenario_dir)
    baselines = build_rule_based_baselines()
    selected_baselines = list(baseline_names) if baseline_names else sorted(baselines)
    unknown = sorted(set(selected_baselines) - set(baselines))
    if unknown:
        raise ValueError(f"unknown rule baseline names: {', '.join(unknown)}")

    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    plan = build_chunk_plan(list(scenario_by_id), selected_baselines)
    completed_before = sum(1 for chunk in plan if chunk_complete(output_dir, chunk))
    ran: list[dict[str, str]] = []
    for chunk in plan:
        if len(ran) >= max_chunks:
            break
        if chunk_complete(output_dir, chunk):
            continue
        baseline = baselines[chunk.baseline_name]
        scenario = scenario_by_id[chunk.scenario_id]
        metrics = []
        rows: list[dict[str, Any]] = []
        for episode_index in range(int(episodes)):
            episode_seed = _episode_seed(seed, scenario, episode_index)
            metric = run_rule_baseline_episode(
                scenario,
                baseline,
                seed=episode_seed,
                episode_index=episode_index,
                max_steps=max_steps,
            )
            metrics.append(metric)
            rows.append(
                {
                    "baseline_name": baseline.name,
                    "baseline_description": baseline.description,
                    **metric.to_dict(),
                }
            )
        summary = summarize_episodes(scenario.scenario_id, metrics)
        summary["baseline_name"] = baseline.name
        summary["description"] = baseline.description
        summary["episodes"] = int(episodes)
        summary["secondary_fleet_rate"] = _rate_from_distribution(
            summary,
            "fleet_distribution",
            "secondary_fleet",
        )
        summary["reorder_none_rate"] = _rate_from_distribution(
            summary,
            "reorder_mode_distribution",
            "none",
        )
        write_chunk_artifacts(
            output_dir=output_dir,
            chunk=chunk,
            summary=summary,
            episode_rows=rows,
        )
        ran.append({"baseline_name": chunk.baseline_name, "scenario_id": chunk.scenario_id})

    completed_after = sum(1 for chunk in plan if chunk_complete(output_dir, chunk))
    return {
        "chunks_total": len(plan),
        "chunks_completed_before": completed_before,
        "chunks_completed_after": completed_after,
        "chunks_run": ran,
        "complete": completed_after == len(plan),
    }


def aggregate_rule_chunks(
    *,
    output_dir: Path,
    plan: Sequence[BenchmarkChunk],
    episodes: int,
    seed: int,
    scenario_dir: Path,
    hierarchical_summary_path: Path | None,
    hierarchical_episode_path: Path | None,
) -> dict[str, Any]:
    _ensure_safe_existing_output_dir(output_dir)
    _refuse_existing_final_outputs(output_dir)

    missing = [chunk for chunk in plan if not chunk_complete(output_dir, chunk)]
    if missing:
        raise RuntimeError(f"cannot aggregate incomplete benchmark; missing chunks: {missing[:5]}")

    summaries: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for chunk in plan:
        directory = chunk_dir(output_dir, chunk)
        summaries.append(json.loads((directory / "summary.json").read_text(encoding="utf-8")))
        for line in (directory / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                episode_rows.append(json.loads(line))

    expected_rows = len(plan) * int(episodes)
    if len(episode_rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} episode rows, found {len(episode_rows)}")

    comparison = _compare_to_hierarchical(
        summaries,
        episode_rows,
        hierarchical_summary_path=hierarchical_summary_path,
        hierarchical_episode_path=hierarchical_episode_path,
    )
    report = {
        "benchmark_metadata": {
            "benchmark": "rule_based_full_20ep",
            "scenario_dir": str(scenario_dir),
            "episodes_per_scenario": int(episodes),
            "seed": int(seed),
            "baseline_count": len({chunk.baseline_name for chunk in plan}),
            "scenario_count": len({chunk.scenario_id for chunk in plan}),
            "chunk_count": len(plan),
            "episode_rows": len(episode_rows),
            "contract": MDP_CONTRACT_VERSION,
            "obs_dim": 73,
            "continuous_action_dim": 5,
            "discrete_action_count": 48,
        },
        "decision": _benchmark_decision(summaries),
        "baseline_rollups": _baseline_rollups(summaries),
        "baseline_summaries": summaries,
        "comparison_to_hierarchical_v1": comparison,
    }
    _write_json(output_dir / "rule_based_full_report.json", report)
    _write_summary_csv(output_dir / "rule_based_full_summary.csv", summaries)
    _write_episode_jsonl(output_dir / "episode_metrics.jsonl", episode_rows)
    _write_markdown(output_dir / "rule_based_full_report.md", report)
    return report


def build_plan_from_repo(scenario_dir: Path, baseline_names: Sequence[str] | None = None) -> list[BenchmarkChunk]:
    scenarios = load_scenario_configs(scenario_dir)
    baselines = build_rule_based_baselines()
    selected = list(baseline_names) if baseline_names else sorted(baselines)
    return build_chunk_plan([scenario.scenario_id for scenario in scenarios], selected)


def _compare_to_hierarchical(
    summaries: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    hierarchical_summary_path: Path | None,
    hierarchical_episode_path: Path | None,
) -> list[dict[str, Any]]:
    reference_summary = _load_hierarchical_summaries(hierarchical_summary_path)
    reference_episodes = _load_hierarchical_episodes(hierarchical_episode_path)
    comparisons: list[dict[str, Any]] = []
    rows_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in episode_rows:
        rows_by_key.setdefault((str(row.get("baseline_name")), str(row.get("scenario_id"))), []).append(row)

    for summary in summaries:
        baseline_name = str(summary.get("baseline_name"))
        scenario_id = str(summary.get("scenario_id"))
        reference = reference_summary.get(scenario_id)
        if reference is None:
            continue
        row: dict[str, Any] = {
            "baseline_name": baseline_name,
            "scenario_id": scenario_id,
            "service_delta_vs_hierarchical": _delta(summary, reference, "service_level"),
            "lateness_delta_vs_hierarchical": _delta(summary, reference, "mean_lateness"),
            "dispatch_success_delta_vs_hierarchical": _delta(
                summary,
                reference,
                "dispatch_success_per_attempt",
            ),
        }
        deltas = _paired_metric_deltas(
            rows_by_key.get((baseline_name, scenario_id), []),
            reference_episodes.get(scenario_id, []),
            "service_level",
        )
        if deltas:
            row["paired_service_delta_stats"] = summary_stats(deltas)
            row["paired_service_delta_bootstrap_ci"] = bootstrap_mean_ci(deltas, iterations=1000, seed=42)
            row["paired_service_delta_sign_test"] = exact_sign_test_two_sided(deltas)
        comparisons.append(row)
    return comparisons


def _paired_metric_deltas(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> list[float]:
    right_by_episode = {int(row.get("episode_index", -1)): row for row in right_rows}
    deltas: list[float] = []
    for row in left_rows:
        episode_index = int(row.get("episode_index", -1))
        reference = right_by_episode.get(episode_index)
        if reference is None:
            continue
        left_value = _finite(row.get(metric))
        right_value = _finite(reference.get(metric))
        if left_value is not None and right_value is not None:
            deltas.append(left_value - right_value)
    return deltas


def _load_hierarchical_summaries(path: Path | None) -> dict[str, Mapping[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("scenarios", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, Sequence):
        return {}
    return {
        str(row.get("scenario_id")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("scenario_id") is not None
    }


def _load_hierarchical_episodes(path: Path | None) -> dict[str, list[Mapping[str, Any]]]:
    rows_by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    if path is None or not path.exists():
        return rows_by_scenario
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, Mapping) and row.get("scenario_id") is not None:
            rows_by_scenario.setdefault(str(row["scenario_id"]), []).append(row)
    return rows_by_scenario


def _baseline_rollups(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_baseline: dict[str, list[Mapping[str, Any]]] = {}
    for summary in summaries:
        by_baseline.setdefault(str(summary.get("baseline_name")), []).append(summary)
    rollups = []
    for baseline_name, rows in sorted(by_baseline.items()):
        service_values = finite_values([row.get("service_level") for row in rows])
        lateness_values = finite_values([row.get("mean_lateness") for row in rows])
        success_values = finite_values([row.get("dispatch_success_per_attempt") for row in rows])
        rollups.append(
            {
                "baseline_name": baseline_name,
                "scenario_count": len(rows),
                "mean_service_level": sum(service_values) / len(service_values) if service_values else None,
                "worst_service_level": min(service_values) if service_values else None,
                "mean_lateness": sum(lateness_values) / len(lateness_values) if lateness_values else None,
                "mean_dispatch_success_per_attempt": (
                    sum(success_values) / len(success_values) if success_values else None
                ),
                "hard_blocker_total": _hard_blocker_total(rows),
            }
        )
    return rollups


def _benchmark_decision(summaries: Sequence[Mapping[str, Any]]) -> str:
    if not summaries:
        return "RULE_BASED_FULL_BENCHMARK_BLOCKED"
    if _hard_blocker_total(summaries) > 0:
        return "RULE_BASED_FULL_BENCHMARK_NEEDS_INVESTIGATION"
    return "RULE_BASED_FULL_BENCHMARK_READY"


def _hard_blocker_total(rows: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for row in rows:
        blockers = row.get("hard_blockers", {})
        if isinstance(blockers, Mapping):
            for value in blockers.values():
                finite = _finite(value)
                if finite is not None:
                    total += int(finite)
    return total


def _write_summary_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "baseline_name",
        "scenario_id",
        "episodes",
        "steps",
        "service_level",
        "mean_lateness",
        "dispatch_rate",
        "dispatch_success_per_attempt",
        "no_vehicle_available",
        "route_failures",
        "action_24_percentage",
        "action_25_percentage",
        "secondary_fleet_rate",
        "reorder_none_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fields})


def _write_episode_jsonl(path: Path, episode_rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in episode_rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    metadata = report.get("benchmark_metadata", {})
    rollups = report.get("baseline_rollups", [])
    lines = [
        "# Full Rule-Based 20-Episode Benchmark",
        "",
        f"- Decision: {report.get('decision')}",
        f"- Scenario dir: {metadata.get('scenario_dir') if isinstance(metadata, Mapping) else ''}",
        f"- Episodes per scenario: {metadata.get('episodes_per_scenario') if isinstance(metadata, Mapping) else ''}",
        f"- Episode rows: {metadata.get('episode_rows') if isinstance(metadata, Mapping) else ''}",
        "",
        "## Baseline Rollups",
        "",
        "| Baseline | Scenarios | Mean service | Worst service | Mean lateness | Mean dispatch success | Hard blockers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if isinstance(rollups, Sequence):
        for row in rollups:
            if isinstance(row, Mapping):
                lines.append(
                    "| {baseline} | {scenarios} | {service} | {worst} | {lateness} | {success} | {blockers} |".format(
                        baseline=row.get("baseline_name", ""),
                        scenarios=row.get("scenario_count", ""),
                        service=_fmt(row.get("mean_service_level")),
                        worst=_fmt(row.get("worst_service_level")),
                        lateness=_fmt(row.get("mean_lateness")),
                        success=_fmt(row.get("mean_dispatch_success_per_attempt")),
                        blockers=row.get("hard_blocker_total", ""),
                    )
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_safe_existing_output_dir(output_dir: Path) -> None:
    _reject_protected_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _refuse_existing_final_outputs(output_dir: Path) -> None:
    for name in (
        "rule_based_full_report.json",
        "rule_based_full_summary.csv",
        "episode_metrics.jsonl",
        "rule_based_full_report.md",
    ):
        path = output_dir / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing final output: {path}")


def _remove_owned_tmp_dir(output_dir: Path, tmp: Path) -> None:
    resolved_output = Path(output_dir).resolve(strict=False)
    resolved_tmp = tmp.resolve(strict=False)
    if resolved_output not in resolved_tmp.parents:
        raise ValueError(f"temporary chunk dir is outside output root: {tmp}")
    if tmp.exists():
        shutil.rmtree(tmp)


def _delta(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> float | None:
    left_value = _finite(left.get(key))
    right_value = _finite(right.get(key))
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


def _fmt(value: Any) -> str:
    finite = _finite(value)
    return "" if finite is None else f"{finite:.4f}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or aggregate resumable full rule-based benchmarks.")
    parser.add_argument("--scenario-dir", type=Path, default=Path("configs/eval_scenarios"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument("--baselines", nargs="*", default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--hierarchical-summary", type=Path, default=DEFAULT_HIERARCHICAL_SUMMARY)
    parser.add_argument("--hierarchical-episodes", type=Path, default=DEFAULT_HIERARCHICAL_EPISODES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    plan = build_plan_from_repo(args.scenario_dir, args.baselines)
    if args.aggregate_only:
        report = aggregate_rule_chunks(
            output_dir=args.output_dir,
            plan=plan,
            episodes=args.episodes,
            seed=args.seed,
            scenario_dir=args.scenario_dir,
            hierarchical_summary_path=args.hierarchical_summary,
            hierarchical_episode_path=args.hierarchical_episodes,
        )
        print(report["decision"])
        return 0

    result = run_missing_chunks(
        scenario_dir=args.scenario_dir,
        output_dir=args.output_dir,
        episodes=args.episodes,
        seed=args.seed,
        baseline_names=args.baselines,
        max_steps=args.max_steps,
        max_chunks=max(1, int(args.max_chunks)),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
