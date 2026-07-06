"""Inventory/reorder benchmark adapter using Gymnasium inventory environments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


POLICY_NAMES = (
    "base_stock_order_up_to",
    "conservative_reorder",
    "aggressive_reorder",
    "project_heuristic_continuous_reorder",
)


def policy_action(policy_name: str, observation: Any, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    if low.shape != high.shape:
        raise ValueError("action_low and action_high shape mismatch.")
    if policy_name == "base_stock_order_up_to":
        target = 0.45 * high
        action = np.maximum(target - _inventory_proxy(obs, high.shape[0]), 0.0)
    elif policy_name == "conservative_reorder":
        action = 0.20 * high
    elif policy_name == "aggressive_reorder":
        action = 0.65 * high
    elif policy_name == "project_heuristic_continuous_reorder":
        inventory = _inventory_proxy(obs, high.shape[0])
        coverage = np.clip(inventory / np.maximum(0.45 * high, 1.0), 0.0, 1.0)
        pressure = 1.0 - float(np.mean(coverage))
        reorder_fraction = np.clip(0.15 + (0.70 * pressure), 0.05, 0.85)
        action = reorder_fraction * high
    else:
        raise ValueError(f"unknown inventory policy: {policy_name}")
    return np.clip(action, low, high).astype(np.float64)


def summarize_policy_run(
    *,
    policy_name: str,
    rewards: Sequence[float],
    inventories: Sequence[float],
    backlogs: Sequence[float],
    order_quantities: Sequence[float],
) -> dict[str, Any]:
    reward_values = [float(value) for value in rewards if math.isfinite(float(value))]
    inventory_values = [float(value) for value in inventories if math.isfinite(float(value))]
    backlog_values = [float(value) for value in backlogs if math.isfinite(float(value))]
    order_values = [float(value) for value in order_quantities if math.isfinite(float(value))]
    return {
        "policy_name": policy_name,
        "steps": len(reward_values),
        "total_reward": sum(reward_values),
        "total_cost": -sum(reward_values),
        "mean_step_cost": (-sum(reward_values) / len(reward_values)) if reward_values else None,
        "mean_inventory": _mean(inventory_values),
        "mean_backlog": _mean(backlog_values),
        "max_backlog": max(backlog_values) if backlog_values else None,
        "no_backlog_step_rate": (
            sum(1 for value in backlog_values if value <= 1e-9) / len(backlog_values) if backlog_values else None
        ),
        "mean_order_quantity": _mean(order_values),
    }


def run_inventory_benchmark(
    *,
    env_id: str = "GymInvMgmt/Serial-v0",
    episodes: int = 10,
    max_steps: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    try:
        import gymnasium as gym
        import gym_invmgmt  # noqa: F401  # registers environments
    except Exception as exc:  # pragma: no cover - exercised only when dependency missing
        return _blocked_report(env_id=env_id, reason=f"dependency_import_failed: {exc}")

    policy_summaries = []
    for policy_name in POLICY_NAMES:
        rewards: list[float] = []
        inventories: list[float] = []
        backlogs: list[float] = []
        order_quantities: list[float] = []
        for episode in range(int(episodes)):
            env = gym.make(env_id)
            try:
                observation, _info = env.reset(seed=seed + episode)
                low = np.asarray(env.action_space.low, dtype=np.float64)
                high = np.asarray(env.action_space.high, dtype=np.float64)
                terminated = False
                truncated = False
                steps = 0
                while not (terminated or truncated) and steps < int(max_steps):
                    action = policy_action(policy_name, observation, low, high)
                    observation, reward, terminated, truncated, info = env.step(action)
                    rewards.append(float(reward))
                    inventories.append(float(info.get("total_inventory", np.nan)))
                    backlogs.append(float(info.get("total_backlog", np.nan)))
                    order_quantities.append(float(np.sum(action)))
                    steps += 1
            finally:
                env.close()
        policy_summaries.append(
            summarize_policy_run(
                policy_name=policy_name,
                rewards=rewards,
                inventories=inventories,
                backlogs=backlogs,
                order_quantities=order_quantities,
            )
        )

    return {
        "benchmark_metadata": {
            "benchmark": "inventory_reorder_gym_invmgmt",
            "env_id": env_id,
            "episodes": int(episodes),
            "max_steps": int(max_steps),
            "seed": int(seed),
            "source": "gym-invmgmt",
            "or_gym_status": "or-gym install blocked on gym<=0.19 metadata build under Python 3.12",
        },
        "decision": "INVENTORY_REORDER_BENCHMARK_READY",
        "policy_summaries": policy_summaries,
    }


def write_outputs(output_dir: Path, report: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "inventory_benchmark_report.json"
    csv_path = output_dir / "inventory_benchmark_summary.csv"
    if report_path.exists() or csv_path.exists():
        raise FileExistsError(f"refusing to overwrite inventory benchmark outputs in {output_dir}")
    report_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = report.get("policy_summaries", [])
    fields = [
        "policy_name",
        "steps",
        "total_reward",
        "total_cost",
        "mean_step_cost",
        "mean_inventory",
        "mean_backlog",
        "max_backlog",
        "no_backlog_step_rate",
        "mean_order_quantity",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows if isinstance(rows, Sequence) else []:
            if isinstance(row, Mapping):
                writer.writerow({field: row.get(field) for field in fields})


def _inventory_proxy(observation: np.ndarray, action_dim: int) -> np.ndarray:
    if observation.size >= action_dim:
        return np.clip(observation[:action_dim], 0.0, None)
    return np.zeros(action_dim, dtype=np.float32)


def _blocked_report(*, env_id: str, reason: str) -> dict[str, Any]:
    return {
        "benchmark_metadata": {"benchmark": "inventory_reorder", "env_id": env_id},
        "decision": "INVENTORY_REORDER_BENCHMARK_BLOCKED",
        "blocker": reason,
        "policy_summaries": [],
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run inventory/reorder benchmark using gym-invmgmt.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-id", default="GymInvMgmt/Serial-v0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    report = run_inventory_benchmark(
        env_id=args.env_id,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    write_outputs(args.output_dir, report)
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
