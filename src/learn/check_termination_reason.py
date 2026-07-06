"""Pinpoint why the fine-tuned DQN evaluation terminates early."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
from stable_baselines3 import DQN

from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv
from src.learn.test_agent import _apply_seeded_stressors


DEFAULT_MODEL_PATH = Path("models/baselines/historical_isolated/dqn_5pl_finetuned.zip")


def build_env(seed: int) -> FivePLDigitalTwinEnv:
    config = EnvironmentConfig(action_mode="discrete", random_seed=seed, max_steps=288)
    env = FivePLDigitalTwinEnv(config)
    default_factory = env.simulation_factory

    def stress_factory(episode_seed: int | None):
        effective_seed = seed if episode_seed is None else int(episode_seed)
        simulation = default_factory(effective_seed)
        _apply_seeded_stressors(simulation, effective_seed)
        return simulation

    env.simulation_factory = stress_factory
    return env


def run(model_path: Path, *, episodes: int, seed: int) -> None:
    model = DQN.load(model_path)
    env = build_env(seed)
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            env.config.random_seed = episode_seed
            obs, info = env.reset(seed=episode_seed)
            recent: deque[dict[str, Any]] = deque(maxlen=3)

            for step in range(1, env.config.max_steps + 1):
                action, _ = model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
                snapshot = dict(info.get("snapshot", {}))
                recent.append(_step_trace(step, action, reward, terminated, truncated, info, snapshot))

                if terminated or truncated:
                    print(f"\n# Episode {episode + 1} | Seed {episode_seed} | Step {step}")
                    print(f"Derived termination reason: {_termination_reason(env, snapshot, terminated, truncated)}")
                    print("\nLast 3 state-steps:")
                    for item in recent:
                        pprint(item, sort_dicts=False)
                    print("\nInfo keys:")
                    pprint(sorted(info.keys()))
                    print("\nRelevant info payload:")
                    pprint(_relevant_payload(info, snapshot), sort_dicts=False)
                    break
    finally:
        env.close()


def _step_trace(
    step: int,
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    return {
        "step": step,
        "action": int(np.asarray(action).item()),
        "reward": float(reward),
        "terminated": terminated,
        "truncated": truncated,
        "projected_action": info.get("projected_action"),
        "projected": info.get("projected"),
        "blocked": info.get("blocked"),
        "orders": _order_counts(snapshot),
        "service_level": snapshot.get("service_level"),
        "safety_potential": snapshot.get("network_safety_potential"),
        "transport_cost": snapshot.get("total_transport_cost"),
        "disruption_score": snapshot.get("disruption_score"),
    }


def _termination_reason(
    env: FivePLDigitalTwinEnv,
    snapshot: dict[str, Any],
    terminated: bool,
    truncated: bool,
) -> str:
    if truncated:
        return "time_limit_truncated"
    if not terminated:
        return "not_terminal"
    safety_potential = float(snapshot.get("network_safety_potential", 0.0))
    reasons: list[str] = []
    if safety_potential >= 0.995:
        reasons.append("catastrophic_safety_potential>=0.995")
    return " + ".join(reasons) if reasons else "unknown_env_terminal_condition"


def _order_counts(snapshot: dict[str, Any]) -> dict[str, int]:
    orders = snapshot.get("orders", {})
    if not isinstance(orders, dict):
        return {}
    counts = {"created": 0, "in_transit": 0, "delivered": 0, "delayed": 0, "late": 0}
    for order in orders.values():
        if not isinstance(order, dict):
            continue
        status = str(order.get("status", "created"))
        counts[status] = counts.get(status, 0) + 1
        if bool(order.get("is_late", False)):
            counts["late"] += 1
    return counts


def _relevant_payload(info: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "info_flags": {
            "projected": info.get("projected"),
            "blocked": info.get("blocked"),
            "projection_reasons": info.get("projection_reasons"),
            "simulation_time": info.get("simulation_time"),
        },
        "projected_action": info.get("projected_action"),
        "snapshot_core": {
            "orders": _order_counts(snapshot),
            "service_level": snapshot.get("service_level"),
            "network_safety_potential": snapshot.get("network_safety_potential"),
            "total_transport_cost": snapshot.get("total_transport_cost"),
            "disruption_score": snapshot.get("disruption_score"),
        },
        "event_log_tail": snapshot.get("event_log", [])[-5:] if isinstance(snapshot.get("event_log"), list) else [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose exact DQN environment termination causes.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.model_path, episodes=int(args.episodes), seed=int(args.seed))


if __name__ == "__main__":
    main()
