"""Resumable continuous-aware rule-baseline benchmark runner."""

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
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.observation_builder import OBSERVATION_DIM
from src.eval.real_world_scenario_arena import load_scenario_configs
from src.eval.rule_based_baseline_arena import (
    ContinuousBaselineMode,
    _episode_seed,
    _reject_protected_output_dir,
    _rate_from_distribution,
    continuous_mode_metadata,
    run_rule_baseline_episode,
)
from src.eval.rule_based_baselines import build_rule_based_baselines
from src.eval.scenario_metrics import summarize_episodes
from src.learn.train_joint_torch import MDP_CONTRACT_VERSION
from src.orchestration.torch_joint_runtime import load_torch_joint_policy


DEFAULT_PRODUCTION_CHECKPOINT = Path(
    "models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt"
)
DEFAULT_MODES = tuple(mode.value for mode in ContinuousBaselineMode)
PPOPolicyLoader = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class ContinuousBenchmarkChunk:
    continuous_mode: str
    baseline_name: str
    scenario_id: str


def build_chunk_plan(
    scenario_ids: Sequence[str],
    baseline_names: Sequence[str],
    continuous_modes: Sequence[str],
) -> list[ContinuousBenchmarkChunk]:
    return [
        ContinuousBenchmarkChunk(
            continuous_mode=str(mode),
            baseline_name=str(baseline_name),
            scenario_id=str(scenario_id),
        )
        for mode in sorted(str(item) for item in continuous_modes)
        for baseline_name in sorted(str(item) for item in baseline_names)
        for scenario_id in sorted(str(item) for item in scenario_ids)
    ]


def build_ppo_continuous_provider(
    *,
    checkpoint_path: Path,
    device: str = "cpu",
    policy_loader: PPOPolicyLoader = load_torch_joint_policy,
) -> Callable[[Any], np.ndarray]:
    policy = policy_loader(checkpoint_path, device=device)

    def provider(observation: Any) -> np.ndarray:
        if observation is None:
            raise ValueError("ppo_assisted_continuous requires the current observation.")
        observation_array = np.asarray(observation, dtype=np.float32).reshape(-1)
        if observation_array.shape != (OBSERVATION_DIM,):
            raise ValueError(
                f"ppo_assisted_continuous observation must have shape ({OBSERVATION_DIM},), "
                f"got {observation_array.shape}."
            )
        action = policy.predict_joint(observation_array, deterministic=True)
        return np.asarray(action.continuous, dtype=np.float32)

    return provider


def chunk_dir(output_dir: Path, chunk: ContinuousBenchmarkChunk) -> Path:
    return (
        Path(output_dir)
        / "chunks"
        / _slug(chunk.continuous_mode)
        / _slug(chunk.baseline_name)
        / _slug(chunk.scenario_id)
    )


def chunk_complete(output_dir: Path, chunk: ContinuousBenchmarkChunk) -> bool:
    directory = chunk_dir(output_dir, chunk)
    return (
        (directory / "chunk_complete.json").exists()
        and (directory / "summary.json").exists()
        and (directory / "episode_metrics.jsonl").exists()
    )


def next_pending_chunk(
    output_dir: Path,
    plan: Sequence[ContinuousBenchmarkChunk],
) -> ContinuousBenchmarkChunk | None:
    for chunk in plan:
        if not chunk_complete(output_dir, chunk):
            return chunk
    return None


def write_chunk_artifacts(
    *,
    output_dir: Path,
    chunk: ContinuousBenchmarkChunk,
    summary: Mapping[str, Any],
    episode_rows: Sequence[Mapping[str, Any]],
) -> None:
    target = chunk_dir(output_dir, chunk)
    tmp = target.with_name(target.name + ".__tmp__")
    _remove_owned_tmp_dir(output_dir, tmp)
    tmp.mkdir(parents=True, exist_ok=False)
    _write_json(tmp / "summary.json", dict(summary))
    with (tmp / "episode_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in episode_rows:
            handle.write(json.dumps(_jsonable(dict(row)), sort_keys=True) + "\n")
    _write_json(
        tmp / "chunk_complete.json",
        {
            "continuous_mode": chunk.continuous_mode,
            "baseline_name": chunk.baseline_name,
            "scenario_id": chunk.scenario_id,
            "episode_rows": len(episode_rows),
        },
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
    continuous_modes: Sequence[str],
    max_steps: int | None,
    max_chunks: int,
    ppo_checkpoint: Path,
    device: str,
) -> dict[str, Any]:
    _ensure_safe_existing_output_dir(output_dir)
    scenarios = load_scenario_configs(scenario_dir)
    baselines = build_rule_based_baselines()
    selected_baselines = list(baseline_names) if baseline_names else sorted(baselines)
    unknown = sorted(set(selected_baselines) - set(baselines))
    if unknown:
        raise ValueError(f"unknown rule baseline names: {', '.join(unknown)}")
    selected_modes = [ContinuousBaselineMode(str(mode)).value for mode in continuous_modes]

    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    plan = build_chunk_plan(list(scenario_by_id), selected_baselines, selected_modes)
    providers: dict[str, Callable[[Any], np.ndarray] | None] = {mode: None for mode in selected_modes}
    if ContinuousBaselineMode.PPO_ASSISTED.value in selected_modes:
        providers[ContinuousBaselineMode.PPO_ASSISTED.value] = build_ppo_continuous_provider(
            checkpoint_path=ppo_checkpoint,
            device=device,
        )

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
                continuous_mode=chunk.continuous_mode,
                continuous_provider=providers.get(chunk.continuous_mode),
                allow_oracle_diagnostic=chunk.continuous_mode
                == ContinuousBaselineMode.ORACLE_DIAGNOSTIC.value,
            )
            metrics.append(metric)
            rows.append(
                {
                    "continuous_mode": chunk.continuous_mode,
                    "baseline_name": baseline.name,
                    "baseline_description": baseline.description,
                    **metric.to_dict(),
                }
            )
        summary = summarize_episodes(scenario.scenario_id, metrics)
        summary["continuous_mode"] = chunk.continuous_mode
        summary["continuous_mode_metadata"] = continuous_mode_metadata(chunk.continuous_mode)
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
        ran.append(
            {
                "continuous_mode": chunk.continuous_mode,
                "baseline_name": chunk.baseline_name,
                "scenario_id": chunk.scenario_id,
            }
        )

    completed_after = sum(1 for chunk in plan if chunk_complete(output_dir, chunk))
    progress = {
        "chunks_total": len(plan),
        "chunks_completed_before": completed_before,
        "chunks_completed_after": completed_after,
        "chunks_run": ran,
        "complete": completed_after == len(plan),
    }
    _write_json(output_dir / "progress_state.json", progress)
    return progress


def aggregate_continuous_chunks(
    *,
    output_dir: Path,
    plan: Sequence[ContinuousBenchmarkChunk],
    episodes: int,
    seed: int,
    scenario_dir: Path,
) -> dict[str, Any]:
    _ensure_safe_existing_output_dir(output_dir)
    _refuse_existing_final_outputs(output_dir)
    missing = [chunk for chunk in plan if not chunk_complete(output_dir, chunk)]
    if missing:
        raise RuntimeError(f"cannot aggregate incomplete continuous benchmark; missing chunks: {missing[:5]}")

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

    report = {
        "benchmark_metadata": {
            "benchmark": "continuous_aware_rule_baselines",
            "scenario_dir": str(scenario_dir),
            "episodes_per_scenario": int(episodes),
            "seed": int(seed),
            "continuous_modes": sorted({chunk.continuous_mode for chunk in plan}),
            "continuous_mode_count": len({chunk.continuous_mode for chunk in plan}),
            "baseline_count": len({chunk.baseline_name for chunk in plan}),
            "scenario_count": len({chunk.scenario_id for chunk in plan}),
            "chunk_count": len(plan),
            "episode_rows": len(episode_rows),
            "contract": MDP_CONTRACT_VERSION,
            "obs_dim": OBSERVATION_DIM,
            "continuous_action_dim": CONTINUOUS_ACTION_DIM,
            "discrete_action_count": DISCRETE_ACTION_COUNT,
        },
        "decision": _benchmark_decision(summaries),
        "continuous_mode_rollups": _continuous_mode_rollups(summaries),
        "baseline_mode_rollups": _baseline_mode_rollups(summaries),
        "baseline_summaries": summaries,
        "fairness_caveats": _fairness_caveats(),
    }
    _write_json(output_dir / "continuous_aware_rule_baseline_report.json", report)
    _write_summary_csv(output_dir / "continuous_aware_rule_baseline_summary.csv", summaries)
    _write_episode_jsonl(output_dir / "episode_metrics.jsonl", episode_rows)
    _write_markdown(output_dir / "continuous_aware_rule_baseline_report.md", report)
    return report


def build_plan_from_repo(
    scenario_dir: Path,
    baseline_names: Sequence[str] | None = None,
    continuous_modes: Sequence[str] = DEFAULT_MODES,
) -> list[ContinuousBenchmarkChunk]:
    scenarios = load_scenario_configs(scenario_dir)
    baselines = build_rule_based_baselines()
    selected_baselines = list(baseline_names) if baseline_names else sorted(baselines)
    selected_modes = [ContinuousBaselineMode(str(mode)).value for mode in continuous_modes]
    return build_chunk_plan([scenario.scenario_id for scenario in scenarios], selected_baselines, selected_modes)


def _benchmark_decision(summaries: Sequence[Mapping[str, Any]]) -> str:
    if not summaries:
        return "CONTINUOUS_AWARE_RULE_BASELINE_BENCHMARK_BLOCKED"
    if _hard_blocker_total(summaries) > 0:
        return "CONTINUOUS_AWARE_RULE_BASELINE_NEEDS_INVESTIGATION"
    return "CONTINUOUS_AWARE_RULE_BASELINE_BENCHMARK_READY"


def _continuous_mode_rollups(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_mode: dict[str, list[Mapping[str, Any]]] = {}
    for summary in summaries:
        by_mode.setdefault(str(summary.get("continuous_mode")), []).append(summary)
    return [
        {
            "continuous_mode": mode,
            "scenario_baseline_rows": len(rows),
            **_rollup_metrics(rows),
        }
        for mode, rows in sorted(by_mode.items())
    ]


def _baseline_mode_rollups(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for summary in summaries:
        by_pair.setdefault(
            (str(summary.get("continuous_mode")), str(summary.get("baseline_name"))),
            [],
        ).append(summary)
    return [
        {
            "continuous_mode": mode,
            "baseline_name": baseline,
            "scenario_count": len(rows),
            **_rollup_metrics(rows),
        }
        for (mode, baseline), rows in sorted(by_pair.items())
    ]


def _rollup_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    service_values = _finite_values([row.get("service_level") for row in rows])
    lateness_values = _finite_values([row.get("mean_lateness") for row in rows])
    success_values = _finite_values([row.get("dispatch_success_per_attempt") for row in rows])
    return {
        "mean_service_level": _mean(service_values),
        "worst_service_level": min(service_values) if service_values else None,
        "mean_lateness": _mean(lateness_values),
        "mean_dispatch_success_per_attempt": _mean(success_values),
        "hard_blocker_total": _hard_blocker_total(rows),
    }


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


def _fairness_caveats() -> list[str]:
    return [
        "neutral_continuous preserves historical fixed normalized-zero behavior; it is midpoint physical control, not physical no-op.",
        "heuristic_continuous is a transparent current-state heuristic, not a learned PPO.",
        "ppo_assisted_continuous pairs production learned continuous controls with rule discrete actions and is diagnostic, not a pure rule baseline.",
        "oracle_diagnostic_continuous is non-deployable and must not be used for advisor-facing fair-comparison claims.",
    ]


def _write_summary_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "continuous_mode",
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
    rollups = report.get("continuous_mode_rollups", [])
    lines = [
        "# Continuous-Aware Rule Baseline Benchmark",
        "",
        f"- Decision: {report.get('decision')}",
        f"- Scenario dir: {metadata.get('scenario_dir') if isinstance(metadata, Mapping) else ''}",
        f"- Episodes per scenario: {metadata.get('episodes_per_scenario') if isinstance(metadata, Mapping) else ''}",
        f"- Episode rows: {metadata.get('episode_rows') if isinstance(metadata, Mapping) else ''}",
        "",
        "## Continuous Mode Rollups",
        "",
        "| Mode | Rows | Mean service | Worst service | Mean lateness | Mean dispatch success | Hard blockers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if isinstance(rollups, Sequence):
        for row in rollups:
            if isinstance(row, Mapping):
                lines.append(
                    "| {mode} | {rows} | {service} | {worst} | {lateness} | {success} | {blockers} |".format(
                        mode=row.get("continuous_mode", ""),
                        rows=row.get("scenario_baseline_rows", ""),
                        service=_fmt(row.get("mean_service_level")),
                        worst=_fmt(row.get("worst_service_level")),
                        lateness=_fmt(row.get("mean_lateness")),
                        success=_fmt(row.get("mean_dispatch_success_per_attempt")),
                        blockers=row.get("hard_blocker_total", ""),
                    )
                )
    lines.extend(["", "## Fairness Caveats", ""])
    for caveat in report.get("fairness_caveats", []):
        lines.append(f"- {caveat}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_safe_existing_output_dir(output_dir: Path) -> None:
    _reject_protected_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _refuse_existing_final_outputs(output_dir: Path) -> None:
    for name in (
        "continuous_aware_rule_baseline_report.json",
        "continuous_aware_rule_baseline_summary.csv",
        "episode_metrics.jsonl",
        "continuous_aware_rule_baseline_report.md",
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


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_values(values: Sequence[Any]) -> list[float]:
    return [finite for value in values if (finite := _finite(value)) is not None]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


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
    parser = argparse.ArgumentParser(description="Run continuous-aware rule-baseline benchmarks.")
    parser.add_argument("--scenario-dir", type=Path, default=Path("configs/eval_scenarios"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument("--baselines", nargs="*", default=None)
    parser.add_argument("--continuous-modes", nargs="*", default=list(DEFAULT_MODES))
    parser.add_argument("--ppo-checkpoint", type=Path, default=DEFAULT_PRODUCTION_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    plan = build_plan_from_repo(args.scenario_dir, args.baselines, args.continuous_modes)
    if args.aggregate_only:
        report = aggregate_continuous_chunks(
            output_dir=args.output_dir,
            plan=plan,
            episodes=args.episodes,
            seed=args.seed,
            scenario_dir=args.scenario_dir,
        )
        print(report["decision"])
        return 0

    progress = run_missing_chunks(
        scenario_dir=args.scenario_dir,
        output_dir=args.output_dir,
        episodes=args.episodes,
        seed=args.seed,
        baseline_names=args.baselines,
        continuous_modes=args.continuous_modes,
        max_steps=args.max_steps,
        max_chunks=max(1, int(args.max_chunks)),
        ppo_checkpoint=args.ppo_checkpoint,
        device=args.device,
    )
    print(json.dumps(progress, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
