"""Repeated runtime latency benchmark wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_runtime_latency import DEFAULT_CHECKPOINT, build_runtime_latency_report
from src.eval.benchmark_statistics import summary_stats


def build_repeated_latency_report(
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    output: Path | None = None,
    runs: int = 5,
    iterations: int = 1000,
    seed_start: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    if runs <= 0:
        raise ValueError("runs must be positive")
    reports = []
    for offset in range(int(runs)):
        reports.append(
            build_runtime_latency_report(
                checkpoint_path=checkpoint_path,
                iterations=iterations,
                seed=seed_start + offset,
                device=device,
            )
        )
    aggregate = aggregate_repeated_latency_reports(reports)
    aggregate["metadata"]["checkpoint_path"] = str(checkpoint_path)
    aggregate["metadata"]["device"] = device
    if output is not None:
        write_repeated_latency_report(output, aggregate)
    return aggregate


def aggregate_repeated_latency_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one run report is required")
    metrics = ("min_ms", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms")
    aggregate_latency = {}
    for metric in metrics:
        aggregate_latency[metric] = summary_stats(
            [report.get("latency_ms", {}).get(metric) for report in reports if isinstance(report.get("latency_ms"), Mapping)]
        )
    load_latency = summary_stats([report.get("load_latency_ms") for report in reports])
    return {
        "decision": "RUNTIME_LATENCY_REPEATED_BENCHMARK_READY",
        "metadata": {
            "benchmark": "runtime_latency_repeated",
            "run_count": len(reports),
            "iterations_per_run": reports[0].get("runtime_metadata", {}).get("iterations")
            if isinstance(reports[0].get("runtime_metadata"), Mapping)
            else None,
            "seeds": [
                report.get("runtime_metadata", {}).get("seed")
                for report in reports
                if isinstance(report.get("runtime_metadata"), Mapping)
            ],
        },
        "run_count": len(reports),
        "per_run_reports": list(reports),
        "load_latency_ms": load_latency,
        "aggregate_latency_ms": aggregate_latency,
    }


def write_repeated_latency_report(output: Path, report: Mapping[str, Any]) -> None:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeated Torch joint runtime latency benchmark.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_repeated_latency_report(
        checkpoint_path=args.checkpoint,
        output=args.output,
        runs=args.runs,
        iterations=args.iterations,
        seed_start=args.seed_start,
        device=args.device,
    )
    print(report["decision"])
    print(json.dumps(report["aggregate_latency_ms"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
