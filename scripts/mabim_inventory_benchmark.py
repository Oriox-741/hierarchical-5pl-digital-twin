"""Run a small MABIM/ReplenishmentEnv inventory benchmark.

This script intentionally keeps the adapter narrow: it uses the public
ReplenishmentEnv source tree supplied via --source-root and writes fresh
benchmark summaries without touching project production artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


PolicyFn = Callable[[object], np.ndarray]


@dataclass(frozen=True)
class MabimPolicy:
    name: str
    build_action: PolicyFn


def _sku_count(env: object) -> int:
    return len(env.get_sku_list())


def _coverage(env: object) -> np.ndarray:
    demand_mean = np.asarray(env.get_demand_mean(), dtype=float)
    return (np.asarray(env.get_in_stock(), dtype=float) + np.asarray(env.get_in_transit(), dtype=float)) / (
        demand_mean + 1e-4
    )


def zero_reorder(env: object) -> np.ndarray:
    return np.zeros((env.warehouse_count, _sku_count(env)), dtype=float)


def ss_policy(env: object, *, reorder_point: float, order_up_to: float) -> np.ndarray:
    coverage = _coverage(env)
    action = np.where(coverage < reorder_point, order_up_to - coverage, 0.0)
    return np.clip(action, 0.0, None)


def project_inspired_reorder(env: object) -> np.ndarray:
    coverage = _coverage(env)
    demand = np.asarray(env.get_demand_mean(), dtype=float)
    normalized_demand = demand / (float(np.nanmean(demand)) + 1e-4)
    target = 2.2 + np.clip(normalized_demand - 1.0, 0.0, 1.5)
    action = np.where(coverage < 1.25, target - coverage, 0.0)
    return np.clip(action, 0.0, None)


def benchmark_policies() -> list[MabimPolicy]:
    return [
        MabimPolicy("zero_reorder_stress_floor", zero_reorder),
        MabimPolicy("conservative_ss_builtin_style", lambda env: ss_policy(env, reorder_point=1.0, order_up_to=2.5)),
        MabimPolicy("aggressive_ss_builtin_style", lambda env: ss_policy(env, reorder_point=2.0, order_up_to=4.5)),
        MabimPolicy("project_inspired_continuous_reorder", project_inspired_reorder),
    ]


def run_policy(make_env: Callable[..., object], task_name: str, policy: MabimPolicy, max_steps: int | None) -> dict:
    env = make_env(task_name, wrapper_names=["OracleWrapper"], mode="test")
    env.reset()
    done = False
    step = 0
    total_reward = 0.0
    balances: list[float] = []
    action_means: list[float] = []
    action_maxes: list[float] = []
    stock_means: list[float] = []
    transit_means: list[float] = []

    while not done and (max_steps is None or step < max_steps):
        action = np.asarray(policy.build_action(env), dtype=float)
        if action.shape != (env.warehouse_count, _sku_count(env)):
            raise ValueError(f"{policy.name} returned action shape {action.shape}")
        stock_means.append(float(np.nanmean(env.get_in_stock())))
        transit_means.append(float(np.nanmean(env.get_in_transit())))
        _, reward, done, info = env.step(action)
        total_reward += float(np.asarray(reward, dtype=float).sum())
        balances.append(float(np.asarray(info.get("balance"), dtype=float).sum()))
        action_means.append(float(np.nanmean(action)))
        action_maxes.append(float(np.nanmax(action)))
        step += 1

    return {
        "policy_name": policy.name,
        "steps": step,
        "total_reward": total_reward,
        "final_balance": balances[-1] if balances else None,
        "mean_balance": float(np.mean(balances)) if balances else None,
        "min_balance": float(np.min(balances)) if balances else None,
        "mean_action": float(np.mean(action_means)) if action_means else None,
        "max_action": float(np.max(action_maxes)) if action_maxes else None,
        "mean_in_stock": float(np.mean(stock_means)) if stock_means else None,
        "mean_in_transit": float(np.mean(transit_means)) if transit_means else None,
    }


def write_outputs(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "mabim_benchmark_report.json"
    csv_path = output_dir / "mabim_benchmark_summary.csv"
    if report_path.exists() or csv_path.exists():
        raise FileExistsError("Refusing to overwrite existing MABIM benchmark outputs")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["policy_summaries"][0].keys()))
        writer.writeheader()
        writer.writerows(report["policy_summaries"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="data/public/mabim_or_replenishment_20260613/source/ReplenishmentEnv-main")
    parser.add_argument("--output-dir", default="reports/benchmarks/sota_pathway_20260613/inventory_mabim")
    parser.add_argument("--task-name", default="sku50.single_store.standard")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    source_root = Path(args.source_root)
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    sys.path.insert(0, str(source_root))

    from ReplenishmentEnv import make_env  # pylint: disable=import-outside-toplevel

    summaries = [run_policy(make_env, args.task_name, policy, args.max_steps) for policy in benchmark_policies()]
    report = {
        "benchmark_metadata": {
            "benchmark": "mabim_replenishment_env_small",
            "source_root": str(source_root),
            "task_name": args.task_name,
            "max_steps": args.max_steps,
            "policy_count": len(summaries),
            "route_fleet_caveat": "MABIM validates inventory/reorder behavior only; it does not validate 5PL dispatch or routing.",
        },
        "decision": "MABIM_INVENTORY_BENCHMARK_READY",
        "policy_summaries": summaries,
    }
    write_outputs(Path(args.output_dir), report)
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
