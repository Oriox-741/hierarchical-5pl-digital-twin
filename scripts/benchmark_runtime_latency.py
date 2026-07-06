"""Read-only CPU latency benchmark for the active Torch joint runtime."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.observation_builder import OBSERVATION_DIM
from src.orchestration.torch_joint_runtime import load_torch_joint_policy


DEFAULT_CHECKPOINT = Path(
    "models/production/joint_torch_v5_prod_hierarchical_v1_1m_20260611/joint_torch_latest.pt"
)
PROTECTED_OUTPUT_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("models", "registry"),
    ("models", "production"),
    ("models", "baselines"),
    ("models", "checkpoints"),
    ("models", "eval"),
    ("db",),
)


PolicyLoader = Callable[..., Any]
Timer = Callable[[], float]


def compute_latency_stats_ms(samples_ms: Sequence[float]) -> dict[str, float | int]:
    samples = sorted(float(value) for value in samples_ms)
    if not samples:
        raise ValueError("latency sample list must not be empty.")
    return {
        "count": len(samples),
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": _percentile(samples, 0.50),
        "p95_ms": _percentile(samples, 0.95),
        "p99_ms": _percentile(samples, 0.99),
    }


def validate_prediction(action: Any) -> dict[str, Any]:
    continuous = np.asarray(getattr(action, "continuous"), dtype=np.float32).reshape(-1)
    discrete = int(getattr(action, "discrete"))
    if continuous.shape != (CONTINUOUS_ACTION_DIM,):
        raise ValueError(
            f"continuous action shape mismatch: expected ({CONTINUOUS_ACTION_DIM},), got {continuous.shape}."
        )
    if not np.all(np.isfinite(continuous)):
        raise ValueError("continuous action contains NaN or Inf values.")
    if np.any(continuous < -1.0) or np.any(continuous > 1.0):
        raise ValueError("continuous action must stay within [-1, 1].")
    if not 0 <= discrete < DISCRETE_ACTION_COUNT:
        raise ValueError(f"discrete action must be in [0, {DISCRETE_ACTION_COUNT - 1}], got {discrete}.")
    return {
        "continuous_dim": int(continuous.shape[0]),
        "continuous_min": float(np.min(continuous)),
        "continuous_max": float(np.max(continuous)),
        "discrete": discrete,
    }


def build_runtime_latency_report(
    *,
    checkpoint_path: Path,
    iterations: int,
    seed: int,
    device: str = "cpu",
    policy_loader: PolicyLoader = load_torch_joint_policy,
    timer: Timer = time.perf_counter,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    torch.set_num_threads(min(2, max(1, torch.get_num_threads())))

    load_start = timer()
    policy = policy_loader(checkpoint_path, device=torch.device(device))
    load_ms = (timer() - load_start) * 1000.0

    rng = np.random.default_rng(seed)
    latencies_ms: list[float] = []
    unique_discrete_actions: set[int] = set()
    validation_samples: list[dict[str, Any]] = []
    for index in range(iterations):
        observation = _observation_for_iteration(rng, index)
        start = timer()
        action = policy.predict_joint(observation, deterministic=True)
        elapsed_ms = (timer() - start) * 1000.0
        validation = validate_prediction(action)
        latencies_ms.append(elapsed_ms)
        unique_discrete_actions.add(int(validation["discrete"]))
        if index < 5:
            validation_samples.append(validation)

    checkpoint = getattr(policy, "checkpoint", {})
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    return {
        "decision": "RUNTIME_LATENCY_BENCHMARK_READY",
        "runtime_metadata": {
            "checkpoint_path": str(checkpoint_path),
            "device": str(device),
            "iterations": int(iterations),
            "seed": int(seed),
            "obs_dim": OBSERVATION_DIM,
            "continuous_action_dim": CONTINUOUS_ACTION_DIM,
            "discrete_action_count": DISCRETE_ACTION_COUNT,
            "global_step": checkpoint.get("global_step"),
            "dqn_architecture": checkpoint.get("dqn_architecture"),
            "hierarchical_init_method": checkpoint.get("hierarchical_init_method"),
        },
        "load_latency_ms": load_ms,
        "latency_ms": compute_latency_stats_ms(latencies_ms),
        "prediction_validation": {
            "valid_predictions": int(iterations),
            "unique_discrete_actions": sorted(unique_discrete_actions),
            "sample_predictions": validation_samples,
        },
    }


def write_runtime_latency_report(output: Path, report: dict[str, Any]) -> dict[str, Any]:
    ensure_fresh_output_file(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def ensure_fresh_output_file(output: Path) -> None:
    _reject_protected_output_path(Path(output))
    if Path(output).exists():
        raise FileExistsError(f"output file already exists: {output}")


def _observation_for_iteration(rng: np.random.Generator, index: int) -> np.ndarray:
    if index % 2 == 0:
        return np.zeros(OBSERVATION_DIM, dtype=np.float32)
    return rng.normal(loc=0.0, scale=0.15, size=OBSERVATION_DIM).astype(np.float32)


def _percentile(sorted_samples: Sequence[float], quantile: float) -> float:
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])
    position = (len(sorted_samples) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_samples[lower])
    lower_value = sorted_samples[lower]
    upper_value = sorted_samples[upper]
    return float(lower_value + (upper_value - lower_value) * (position - lower))


def _reject_protected_output_path(output: Path) -> None:
    candidate = Path(output).resolve(strict=False)
    for prefix in PROTECTED_OUTPUT_PREFIXES:
        protected_root = REPO_ROOT.joinpath(*prefix).resolve(strict=False)
        if candidate == protected_root or protected_root in candidate.parents:
            raise ValueError(f"refusing to write benchmark output under protected path: {output}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Torch joint runtime prediction latency.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    report = build_runtime_latency_report(
        checkpoint_path=Path(args.checkpoint),
        iterations=args.iterations,
        seed=args.seed,
        device=args.device,
    )
    write_runtime_latency_report(Path(args.output), report)
    print(report["decision"])
    print(json.dumps(report["latency_ms"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
