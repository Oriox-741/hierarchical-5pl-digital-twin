"""Evaluate coupled PPO+DQN execution in the joint 5PL environment."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from stable_baselines3 import DQN, PPO

from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv


DEFAULT_PPO_PATH = Path("models/baselines/historical_isolated/ppo_5pl_final.zip")
DEFAULT_DQN_PATH = Path("models/baselines/historical_isolated/dqn_5pl_finetuned.zip")
DEFAULT_CONFIG = Path("configs/training_joint.json")
DEFAULT_EPISODES = 100
DEFAULT_SEED = 42
DEFAULT_MAX_STEPS = int(
    json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))["shared_global_parameters"]["max_steps"]
)


@dataclass(slots=True)
class JointTotals:
    episodes: int = 0
    steps: int = 0
    total_reward: float = 0.0
    global_reward: float = 0.0
    ppo_local_reward: float = 0.0
    dqn_local_reward: float = 0.0
    blocked_steps: int = 0
    projected_steps: int = 0
    terminated_episodes: int = 0
    truncated_episodes: int = 0


def load_models(ppo_path: Path, dqn_path: Path) -> tuple[PPO, DQN]:
    if not ppo_path.exists():
        raise FileNotFoundError(f"PPO checkpoint not found: {ppo_path}")
    if not dqn_path.exists():
        raise FileNotFoundError(f"DQN checkpoint not found: {dqn_path}")
    return PPO.load(ppo_path), DQN.load(dqn_path)


def evaluate(
    *,
    ppo_model: PPO,
    dqn_model: DQN,
    episodes: int,
    seed: int,
    max_steps: int,
) -> JointTotals:
    totals = JointTotals()
    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        env = FivePLDigitalTwinEnv(
            EnvironmentConfig(
                action_mode="joint",
                random_seed=episode_seed,
                max_steps=max_steps,
            )
        )
        try:
            observation, _ = env.reset(seed=episode_seed)
            terminated = False
            truncated = False
            episode_reward = 0.0

            while not terminated and not truncated:
                obs = np.asarray(observation, dtype=np.float32)
                ppo_action, _ = ppo_model.predict(obs, deterministic=True)
                dqn_action, _ = dqn_model.predict(obs, deterministic=True)
                joint_action = {
                    "continuous": np.asarray(ppo_action, dtype=np.float32).reshape(-1),
                    "discrete": int(np.asarray(dqn_action).reshape(-1)[0]),
                }

                observation, reward, terminated, truncated, info = env.step(joint_action)
                reward_components = _reward_components(info)

                totals.steps += 1
                totals.total_reward += float(reward)
                totals.global_reward += reward_components.get("global", 0.0)
                totals.ppo_local_reward += reward_components.get("ppo_local", 0.0)
                totals.dqn_local_reward += reward_components.get("dqn_local", 0.0)
                totals.blocked_steps += int(bool(info.get("blocked", False)))
                totals.projected_steps += int(bool(info.get("projected", False)))
                episode_reward += float(reward)

            totals.episodes += 1
            totals.terminated_episodes += int(terminated)
            totals.truncated_episodes += int(truncated)
            print(
                f"episode={episode_index + 1:02d} seed={episode_seed} "
                f"steps={env._step_count} reward={episode_reward:.6f} "
                f"terminated={terminated} truncated={truncated}"
            )
        finally:
            env.close()
    return totals


def print_summary(totals: JointTotals) -> None:
    denominator = max(totals.steps, 1)
    print("\n# Joint PPO+DQN Baseline Evaluation")
    print()
    print("| Metric | Value |")
    print("|---|---:|")
    print(f"| Episodes | {totals.episodes} |")
    print(f"| Total Steps | {totals.steps} |")
    print(f"| Total Cumulative Reward | {totals.total_reward:.6f} |")
    print(f"| Mean Global Reward / Step | {totals.global_reward / denominator:.6f} |")
    print(f"| Mean PPO Local Reward / Step | {totals.ppo_local_reward / denominator:.6f} |")
    print(f"| Mean DQN Local Reward / Step | {totals.dqn_local_reward / denominator:.6f} |")
    print(f"| Blocked Action Rate | {totals.blocked_steps / denominator:.6f} |")
    print(f"| Projected Action Rate | {totals.projected_steps / denominator:.6f} |")
    print(f"| Terminated Episodes | {totals.terminated_episodes} |")
    print(f"| Truncated Episodes | {totals.truncated_episodes} |")


def _reward_components(info: dict[str, Any]) -> dict[str, float]:
    raw = info.get("reward_components", {})
    if not isinstance(raw, dict):
        return {}
    output: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            continue
        output[str(key)] = float(value)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate coupled PPO+DQN policies in joint 5PL mode.")
    parser.add_argument("--ppo-path", type=Path, default=DEFAULT_PPO_PATH)
    parser.add_argument("--dqn-path", type=Path, default=DEFAULT_DQN_PATH)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be greater than zero.")
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be greater than zero.")

    ppo_model, dqn_model = load_models(cast(Path, args.ppo_path), cast(Path, args.dqn_path))
    totals = evaluate(
        ppo_model=ppo_model,
        dqn_model=dqn_model,
        episodes=int(args.episodes),
        seed=int(args.seed),
        max_steps=int(args.max_steps),
    )
    print_summary(totals)


if __name__ == "__main__":
    main()
