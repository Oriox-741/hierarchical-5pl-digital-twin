"""Evaluate trained PPO/DQN policies against a deterministic heuristic baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from src.act.env_5pl import ActionMode, EnvironmentConfig, FivePLDigitalTwinEnv


class PredictableModel(Protocol):
    def predict(
        self,
        observation: np.ndarray,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        ...


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    episodes: int
    mean_reward: float
    mean_service_level: float
    mean_safety_potential: float
    mean_steps: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "episodes": self.episodes,
            "mean_reward": self.mean_reward,
            "mean_service_level": self.mean_service_level,
            "mean_safety_potential": self.mean_safety_potential,
            "mean_steps": self.mean_steps,
        }


class HeuristicPolicy:
    """Simple deterministic benchmark that dispatches and reorders aggressively."""

    def __init__(self, action_mode: ActionMode) -> None:
        self.action_mode = action_mode

    def predict(
        self,
        observation: np.ndarray,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        if self.action_mode == "continuous":
            return np.array([0.7, 0.8, 0.0, 0.6, 0.3], dtype=np.float32), state
        return np.array(23, dtype=np.int64), state


def evaluate(
    model: PredictableModel,
    *,
    action_mode: ActionMode,
    episodes: int,
    seed: int | None,
) -> EvaluationResult:
    rewards: list[float] = []
    service_levels: list[float] = []
    safety_values: list[float] = []
    steps: list[int] = []

    for episode in range(episodes):
        env = FivePLDigitalTwinEnv(
            EnvironmentConfig(
                action_mode=action_mode,
                random_seed=None if seed is None else seed + episode,
            )
        )
        observation, _ = env.reset(seed=None if seed is None else seed + episode)
        total_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        state: tuple[np.ndarray, ...] | None = None
        episode_start: np.ndarray | None = np.array([True], dtype=bool)

        while not terminated and not truncated:
            action_array, state = model.predict(
                observation,
                state=state,
                episode_start=episode_start,
                deterministic=True,
            )
            episode_start = np.array([False], dtype=bool)
            action: np.ndarray | int
            if action_mode == "discrete":
                action = int(np.asarray(action_array).item())
            else:
                action = np.asarray(action_array, dtype=np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

        snapshot = info.get("snapshot", {})
        rewards.append(total_reward)
        service_levels.append(float(snapshot.get("service_level", 0.0)))
        safety_values.append(float(snapshot.get("network_safety_potential", 0.0)))
        steps.append(step_count)
        env.close()

    return EvaluationResult(
        episodes=episodes,
        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
        mean_service_level=float(np.mean(service_levels)) if service_levels else 0.0,
        mean_safety_potential=float(np.mean(safety_values)) if safety_values else 0.0,
        mean_steps=float(np.mean(steps)) if steps else 0.0,
    )


def load_model(path: Path, algorithm: str) -> PredictableModel:
    model: BaseAlgorithm
    if algorithm.lower() == "ppo":
        model = PPO.load(path)
        return cast(PredictableModel, model)
    if algorithm.lower() == "dqn":
        model = DQN.load(path)
        return cast(PredictableModel, model)
    if algorithm.lower() == "heuristic":
        raise ValueError("heuristic does not load from disk.")
    raise ValueError(f"unsupported algorithm: {algorithm}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 5PL RL policies against heuristics.")
    parser.add_argument("--algorithm", choices=("ppo", "dqn", "heuristic"), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    action_mode: ActionMode = cast(ActionMode, "continuous" if args.algorithm == "ppo" else "discrete")
    if args.algorithm == "heuristic":
        model: PredictableModel = HeuristicPolicy(action_mode)
    else:
        if args.model_path is None:
            raise SystemExit("--model-path is required for PPO/DQN evaluation.")
        model = load_model(args.model_path, args.algorithm)

    result = evaluate(model, action_mode=action_mode, episodes=args.episodes, seed=args.seed)
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
