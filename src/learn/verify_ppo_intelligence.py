"""Audit PPO continuous action variance and stress responsiveness in joint mode."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DQN, PPO

from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv


PPO_MODEL_PATH = Path("models/baselines/historical_isolated/ppo_5pl_final.zip")
DQN_MODEL_PATH = Path("models/baselines/historical_isolated/dqn_5pl_finetuned.zip")
CONFIG_PATH = Path("configs/training_joint.json")
START_SEED = 42
EPISODES = 5
MAX_STEPS = int(json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["shared_global_parameters"]["max_steps"])


@dataclass(slots=True)
class TraceArrays:
    in_transit_orders: list[float]
    capacity_utilization: list[float]
    safety_potential: list[float]
    raw_reorder_fraction: list[float]
    raw_dispatch_intensity: list[float]
    raw_speed_head: list[float]
    raw_safety_stock_head: list[float]
    raw_capacity_buffer_head: list[float]
    speed_multiplier: list[float]
    capacity_buffer_fraction: list[float]
    safety_stock_multiplier: list[float]
    applied_speed_multiplier: list[float]
    applied_capacity_buffer_fraction: list[float]
    applied_safety_stock_multiplier: list[float]


def main() -> None:
    ppo = PPO.load(PPO_MODEL_PATH)
    dqn = DQN.load(DQN_MODEL_PATH)
    traces = TraceArrays([], [], [], [], [], [], [], [], [], [], [], [], [], [])
    print(f"Loaded PPO: {PPO_MODEL_PATH.resolve()} timesteps={ppo.num_timesteps}")
    print(f"Loaded DQN: {DQN_MODEL_PATH.resolve()} timesteps={dqn.num_timesteps}")

    for episode_index in range(EPISODES):
        seed = START_SEED + episode_index
        env = FivePLDigitalTwinEnv(
            EnvironmentConfig(
                action_mode="joint",
                random_seed=seed,
                max_steps=MAX_STEPS,
            )
        )
        try:
            obs, _ = env.reset(seed=seed)
            terminated = False
            truncated = False
            while not terminated and not truncated:
                snapshot = env.simulation.snapshot()
                safety_context = env._safety_context()

                ppo_raw, _ = ppo.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
                dqn_raw, _ = dqn.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
                ppo_vector = np.asarray(ppo_raw, dtype=np.float32).reshape(-1)
                physical = env.action_projector.project(ppo_vector)

                traces.in_transit_orders.append(float(_order_status_count(snapshot, "in_transit")))
                traces.capacity_utilization.append(float(safety_context["capacity_utilization"]))
                traces.safety_potential.append(float(snapshot["network_safety_potential"]))
                traces.raw_reorder_fraction.append(float(ppo_vector[0]))
                traces.raw_dispatch_intensity.append(float(ppo_vector[1]))
                traces.raw_speed_head.append(float(ppo_vector[2]))
                traces.raw_safety_stock_head.append(float(ppo_vector[3]))
                traces.raw_capacity_buffer_head.append(float(ppo_vector[4]))
                traces.speed_multiplier.append(float(physical.speed_multiplier))
                traces.capacity_buffer_fraction.append(float(physical.capacity_buffer_fraction))
                traces.safety_stock_multiplier.append(float(physical.safety_stock_multiplier))

                obs, _, terminated, truncated, info = env.step(
                    {
                        "continuous": ppo_vector,
                        "discrete": int(np.asarray(dqn_raw).reshape(-1)[0]),
                    }
                )
                applied = _applied_continuous_action(info)
                traces.applied_speed_multiplier.append(float(applied.get("speed_multiplier", 0.0)))
                traces.applied_capacity_buffer_fraction.append(
                    float(applied.get("capacity_buffer_fraction", 0.0))
                )
                traces.applied_safety_stock_multiplier.append(
                    float(applied.get("safety_stock_multiplier", 0.0))
                )
        finally:
            env.close()

    print_report(traces)


def print_report(traces: TraceArrays) -> None:
    print("# PPO Continuous-Control Intelligence Audit")
    print()
    print("| Diagnostic | Value |")
    print("|---|---:|")
    print(f"| Samples | {len(traces.speed_multiplier)} |")
    print(f"| Std Dev Raw: reorder_fraction head | {_std(traces.raw_reorder_fraction):.8f} |")
    print(f"| Std Dev Raw: dispatch_intensity head | {_std(traces.raw_dispatch_intensity):.8f} |")
    print(f"| Std Dev Raw: speed head | {_std(traces.raw_speed_head):.8f} |")
    print(f"| Std Dev Raw: safety_stock head | {_std(traces.raw_safety_stock_head):.8f} |")
    print(f"| Std Dev Raw: capacity_buffer head | {_std(traces.raw_capacity_buffer_head):.8f} |")
    print(f"| Std Dev: speed_multiplier | {_std(traces.speed_multiplier):.8f} |")
    print(f"| Std Dev: capacity_buffer_fraction | {_std(traces.capacity_buffer_fraction):.8f} |")
    print(f"| Std Dev: safety_stock_multiplier | {_std(traces.safety_stock_multiplier):.8f} |")
    print(f"| Std Dev Applied: speed_multiplier | {_std(traces.applied_speed_multiplier):.8f} |")
    print(
        "| Std Dev Applied: capacity_buffer_fraction | "
        f"{_std(traces.applied_capacity_buffer_fraction):.8f} |"
    )
    print(
        "| Std Dev Applied: safety_stock_multiplier | "
        f"{_std(traces.applied_safety_stock_multiplier):.8f} |"
    )
    print(
        "| Corr: capacity_utilization vs capacity_buffer_fraction | "
        f"{_corr(traces.capacity_utilization, traces.capacity_buffer_fraction):.8f} |"
    )
    print(
        "| Corr: safety_potential vs safety_stock_multiplier | "
        f"{_corr(traces.safety_potential, traces.safety_stock_multiplier):.8f} |"
    )
    print(f"| Mean in-transit orders | {_mean(traces.in_transit_orders):.8f} |")
    print(f"| Mean capacity utilization | {_mean(traces.capacity_utilization):.8f} |")
    print(f"| Mean safety potential | {_mean(traces.safety_potential):.8f} |")

    static_threshold = 1e-5
    dynamic = any(
        _std(values) > static_threshold
        for values in (
            traces.raw_speed_head,
            traces.raw_safety_stock_head,
            traces.raw_capacity_buffer_head,
            traces.speed_multiplier,
            traces.capacity_buffer_fraction,
            traces.safety_stock_multiplier,
        )
    )
    print()
    print(f"Verdict: {'DYNAMIC' if dynamic else 'STATIC/DEAD'}")
    if not dynamic:
        print(
            "Diagnosis: PPO loaded successfully, but its deterministic strategic heads are saturated "
            "before the environment receives the action."
        )


def _order_status_count(snapshot: dict[str, Any], status: str) -> int:
    orders = snapshot.get("orders", {})
    if not isinstance(orders, dict):
        return 0
    return sum(
        1
        for order in orders.values()
        if isinstance(order, dict) and str(order.get("status", "")).lower() == status
    )


def _applied_continuous_action(info: dict[str, Any]) -> dict[str, float]:
    projected_action = info.get("projected_action", {})
    if not isinstance(projected_action, dict):
        return {}
    continuous = projected_action.get("continuous", {})
    if not isinstance(continuous, dict):
        return {}
    output: dict[str, float] = {}
    for key, value in continuous.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            continue
        output[str(key)] = float(value)
    return output


def _std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0)) if values else 0.0


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _corr(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(right) < 2 or len(left) != len(right):
        return 0.0
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if np.std(left_array) == 0.0 or np.std(right_array) == 0.0:
        return 0.0
    return float(np.corrcoef(left_array, right_array)[0, 1])


if __name__ == "__main__":
    main()
