"""Joint PPO+DQN training from scratch over the corrected rolling 5PL MDP."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv
from src.learn.model_registry import ModelRegistry
from src.learn.sb3_trace_callback import TrainingTraceCallback
from src.think.policies import dqn_policy_kwargs, ppo_policy_kwargs


AgentKind = Literal["ppo", "dqn"]
CounterpartGetter = Callable[[], BaseAlgorithm | None]

DEFAULT_CONFIG = Path("configs/training_joint.json")
DEFAULT_OUTPUT_DIR = Path("models/checkpoints")
DEFAULT_COORDINATION_INTERVAL = 50_000


@dataclass(frozen=True, slots=True)
class JointTrainingConfig:
    shared: dict[str, Any]
    ppo: dict[str, Any]
    dqn: dict[str, Any]

    @property
    def total_timesteps(self) -> int:
        return _positive_int(self.shared, "total_timesteps")

    @property
    def seed(self) -> int:
        return int(self.shared.get("seed", 42))

    @property
    def max_steps(self) -> int:
        return int(self.shared.get("max_steps", 288))

    @property
    def save_freq(self) -> int:
        return int(self.shared.get("save_freq", 10_000))

    @property
    def log_interval(self) -> int:
        return int(self.shared.get("log_interval", 10))

    @property
    def coordination_interval(self) -> int:
        return int(self.shared.get("coordination_interval_steps", DEFAULT_COORDINATION_INTERVAL))

    @property
    def trace_sample_interval(self) -> int:
        return int(self.shared.get("trace_sample_interval", 50))

    @property
    def trace_flush_batch_size(self) -> int:
        return int(self.shared.get("trace_flush_batch_size", 5_000))

    @property
    def environment_id(self) -> str:
        return str(self.shared.get("environment_id", "joint_rolling_5pl_288"))

    @property
    def team_id(self) -> str:
        return str(self.shared.get("team_id", "joint_from_scratch"))


class ModelHolder:
    def __init__(self) -> None:
        self.model: BaseAlgorithm | None = None


class JointRoleTrainingEnv(gym.Env[np.ndarray, np.ndarray | int]):
    """Role-specific SB3 proxy that executes atomic joint actions internally."""

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        *,
        role: AgentKind,
        counterpart_getter: CounterpartGetter,
        config: JointTrainingConfig,
        seed_offset: int = 0,
        global_reward_weight: float = 0.50,
        local_reward_weight: float = 0.50,
    ) -> None:
        super().__init__()
        self.role = role
        self.counterpart_getter = counterpart_getter
        self.config = config
        self.global_reward_weight = global_reward_weight
        self.local_reward_weight = local_reward_weight
        self.rng = np.random.default_rng(config.seed + seed_offset)
        self.env = FivePLDigitalTwinEnv(
            EnvironmentConfig(
                action_mode="joint",
                random_seed=config.seed + seed_offset,
                max_steps=config.max_steps,
            )
        )
        self.observation_space = self.env.observation_space
        if role == "ppo":
            self.action_space: spaces.Space[np.ndarray | int] = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(CONTINUOUS_ACTION_DIM,),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Discrete(DISCRETE_ACTION_COUNT)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return self.env.reset(seed=seed, options=options)

    def step(self, action: np.ndarray | int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation = self.env._observation()
        if self.role == "ppo":
            continuous = np.asarray(action, dtype=np.float32).reshape(-1)
            discrete = self._counterpart_discrete_action(observation)
        else:
            continuous = self._counterpart_continuous_action(observation)
            discrete = int(np.asarray(action).reshape(-1)[0])

        next_observation, _, terminated, truncated, info = self.env.step(
            {
                "continuous": continuous,
                "discrete": discrete,
            }
        )
        role_reward = self._role_reward(info)
        info = dict(info)
        info["training_role"] = self.role
        info["training_reward"] = role_reward
        info["counterpart_action_source"] = "model" if self.counterpart_getter() is not None else "bootstrap"
        return next_observation, role_reward, terminated, truncated, info

    def render(self) -> str:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def _counterpart_continuous_action(self, observation: np.ndarray) -> np.ndarray:
        model = self.counterpart_getter()
        if model is not None:
            action, _ = model.predict(observation, deterministic=False)
            return np.asarray(action, dtype=np.float32).reshape(-1)
        return self.rng.uniform(-0.35, 0.35, size=CONTINUOUS_ACTION_DIM).astype(np.float32)

    def _counterpart_discrete_action(self, observation: np.ndarray) -> int:
        model = self.counterpart_getter()
        if model is not None:
            action, _ = model.predict(observation, deterministic=False)
            return int(np.asarray(action).reshape(-1)[0])
        return int(self.rng.integers(0, DISCRETE_ACTION_COUNT))

    def _role_reward(self, info: Mapping[str, Any]) -> float:
        components = info.get("reward_components", {})
        if not isinstance(components, Mapping):
            return 0.0
        global_reward = _float_value(components, "global", _float_value(components, "total", 0.0))
        local_key = "ppo_local" if self.role == "ppo" else "dqn_local"
        local_reward = _float_value(components, local_key, global_reward)
        return (self.global_reward_weight * global_reward) + (self.local_reward_weight * local_reward)


class JointDiagnosticsCallback(BaseCallback):
    """Track decomposed rewards and role-specific policy diversity during training."""

    def __init__(
        self,
        *,
        role: AgentKind,
        metrics_path: Path,
        window_size: int = 2_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.role = role
        self.metrics_path = metrics_path
        self.window_size = window_size
        self.global_rewards: deque[float] = deque(maxlen=window_size)
        self.ppo_rewards: deque[float] = deque(maxlen=window_size)
        self.dqn_rewards: deque[float] = deque(maxlen=window_size)
        self.actions: deque[np.ndarray] = deque(maxlen=window_size)

    def _on_training_start(self) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        actions = np.asarray(self.locals.get("actions", []))
        if actions.size > 0:
            for action in actions.reshape(actions.shape[0], -1):
                self.actions.append(np.asarray(action, dtype=np.float64))

        for info in infos:
            if not isinstance(info, Mapping):
                continue
            components = info.get("reward_components", {})
            if not isinstance(components, Mapping):
                continue
            self.global_rewards.append(_float_value(components, "global", 0.0))
            self.ppo_rewards.append(_float_value(components, "ppo_local", 0.0))
            self.dqn_rewards.append(_float_value(components, "dqn_local", 0.0))

        if self.n_calls % self.window_size == 0:
            self._write_snapshot()
        return True

    def _on_training_end(self) -> None:
        self._write_snapshot()

    def _write_snapshot(self) -> None:
        if not self.global_rewards:
            return
        payload = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "role": self.role,
            "num_timesteps": int(self.num_timesteps),
            "mean_global_reward": _mean(self.global_rewards),
            "mean_ppo_local_reward": _mean(self.ppo_rewards),
            "mean_dqn_local_reward": _mean(self.dqn_rewards),
            "action_variance": self._action_variance(),
            "action_entropy": self._action_entropy(),
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        if self.verbose > 0:
            print(payload)

    def _action_variance(self) -> list[float]:
        if not self.actions:
            return []
        matrix = np.vstack(list(self.actions))
        return [float(value) for value in np.var(matrix, axis=0)]

    def _action_entropy(self) -> float:
        if not self.actions:
            return 0.0
        if self.role == "dqn":
            counts = Counter(int(action.reshape(-1)[0]) for action in self.actions)
            probabilities = np.asarray(list(counts.values()), dtype=np.float64)
            probabilities = probabilities / probabilities.sum()
            return float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
        matrix = np.vstack(list(self.actions))
        variances = np.var(matrix, axis=0)
        return float(np.mean(np.log(2.0 * math.pi * math.e * np.maximum(variances, 1e-12)) * 0.5))


def load_joint_config(path: Path) -> JointTrainingConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    shared = data.get("shared_global_parameters")
    ppo = data.get("ppo_hyperparameters")
    dqn = data.get("dqn_hyperparameters")
    if not isinstance(shared, dict) or not isinstance(ppo, dict) or not isinstance(dqn, dict):
        raise ValueError("training_joint.json must define shared_global_parameters, ppo_hyperparameters, and dqn_hyperparameters.")
    return JointTrainingConfig(shared=shared, ppo=ppo, dqn=dqn)


def train_joint_from_scratch(
    *,
    config_path: Path,
    output_root: Path,
    trace_logging: bool,
) -> tuple[Path, Path]:
    config = load_joint_config(config_path)
    if config.coordination_interval <= 0:
        raise ValueError("coordination_interval_steps must be positive.")

    joint_output_dir = output_root / "joint_from_scratch"
    ppo_output_dir = output_root / "ppo"
    dqn_output_dir = output_root / "dqn"
    joint_output_dir.mkdir(parents=True, exist_ok=True)
    ppo_output_dir.mkdir(parents=True, exist_ok=True)
    dqn_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = joint_output_dir / "training_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    ppo_holder = ModelHolder()
    dqn_holder = ModelHolder()

    ppo_env = DummyVecEnv(
        [
            lambda: Monitor(
                JointRoleTrainingEnv(
                    role="ppo",
                    counterpart_getter=lambda: dqn_holder.model,
                    config=config,
                    seed_offset=0,
                )
            )
        ]
    )
    dqn_env = DummyVecEnv(
        [
            lambda: Monitor(
                JointRoleTrainingEnv(
                    role="dqn",
                    counterpart_getter=lambda: ppo_holder.model,
                    config=config,
                    seed_offset=10_000,
                )
            )
        ]
    )

    try:
        ppo_model = PPO(
            str(config.ppo.get("policy", "MlpPolicy")),
            ppo_env,
            learning_rate=float(config.ppo["learning_rate"]),
            gamma=float(config.ppo["gamma"]),
            gae_lambda=float(config.ppo["gae_lambda"]),
            clip_range=float(config.ppo["clip_range"]),
            ent_coef=float(config.ppo["ent_coef"]),
            max_grad_norm=float(config.ppo["max_grad_norm"]),
            policy_kwargs=ppo_policy_kwargs(),
            verbose=1,
            seed=config.seed,
        )
        dqn_model = DQN(
            str(config.dqn.get("policy", "MlpPolicy")),
            dqn_env,
            learning_rate=float(config.dqn["learning_rate"]),
            gamma=float(config.dqn["gamma"]),
            buffer_size=int(config.dqn["buffer_size"]),
            batch_size=int(config.dqn["batch_size"]),
            exploration_fraction=float(config.dqn["exploration_fraction"]),
            exploration_final_eps=float(config.dqn["exploration_final_eps"]),
            target_update_interval=int(config.dqn.get("target_update_interval", 1_000)),
            policy_kwargs=dqn_policy_kwargs(),
            verbose=1,
            seed=config.seed + 1,
        )
        ppo_holder.model = ppo_model
        dqn_holder.model = dqn_model

        ppo_callbacks = _build_callbacks(
            role="ppo",
            config=config,
            output_dir=ppo_output_dir,
            metrics_path=metrics_path,
            trace_logging=trace_logging,
        )
        dqn_callbacks = _build_callbacks(
            role="dqn",
            config=config,
            output_dir=dqn_output_dir,
            metrics_path=metrics_path,
            trace_logging=trace_logging,
        )

        trained = 0
        while trained < config.total_timesteps:
            interval = min(config.coordination_interval, config.total_timesteps - trained)
            print(f"joint_cycle_start trained={trained:,} interval={interval:,}")
            ppo_model.learn(
                total_timesteps=interval,
                callback=ppo_callbacks,
                reset_num_timesteps=trained == 0,
                log_interval=config.log_interval,
            )
            ppo_holder.model = ppo_model
            dqn_model.learn(
                total_timesteps=interval,
                callback=dqn_callbacks,
                reset_num_timesteps=trained == 0,
                log_interval=config.log_interval,
            )
            dqn_holder.model = dqn_model
            trained += interval
            ppo_model.save(joint_output_dir / "ppo_joint_latest.zip")
            dqn_model.save(joint_output_dir / "dqn_joint_latest.zip")
            print(f"joint_cycle_done trained={trained:,}")

        ppo_final_path = ppo_output_dir / "ppo_joint_from_scratch_final.zip"
        dqn_final_path = dqn_output_dir / "dqn_joint_from_scratch_final.zip"
        ppo_model.save(ppo_final_path)
        dqn_model.save(dqn_final_path)
        ppo_model.save(joint_output_dir / "ppo_joint_from_scratch_final.zip")
        dqn_model.save(joint_output_dir / "dqn_joint_from_scratch_final.zip")

        registry = ModelRegistry()
        registry.register(
            algorithm="ppo",
            path=ppo_final_path,
            agent_role="continuous_control",
            metrics={},
            metadata={"source": "joint_from_scratch", "config_path": str(config_path)},
            status="active",
        )
        registry.register(
            algorithm="dqn",
            path=dqn_final_path,
            agent_role="tactical_dispatch",
            metrics={},
            metadata={"source": "joint_from_scratch", "config_path": str(config_path)},
            status="active",
        )
        return ppo_final_path, dqn_final_path
    finally:
        ppo_env.close()
        dqn_env.close()


def _build_callbacks(
    *,
    role: AgentKind,
    config: JointTrainingConfig,
    output_dir: Path,
    metrics_path: Path,
    trace_logging: bool,
) -> CallbackList:
    callbacks: list[BaseCallback] = [
        CheckpointCallback(
            save_freq=config.save_freq,
            save_path=str(output_dir),
            name_prefix=f"{role}_joint_from_scratch",
        ),
        JointDiagnosticsCallback(
            role=role,
            metrics_path=metrics_path,
            verbose=0,
        ),
    ]
    if trace_logging:
        callbacks.append(
            TrainingTraceCallback(
                algorithm=role,
                policy_id=f"{role}_joint_from_scratch",
                environment_id=config.environment_id,
                agent_id="ppo_strategic_controller" if role == "ppo" else "dqn_tactical_controller",
                agent_role="continuous_control" if role == "ppo" else "tactical_dispatch",
                team_id=config.team_id,
                flush_batch_size=config.trace_flush_batch_size,
                sample_every_n_steps=config.trace_sample_interval,
                log_episode_boundaries=True,
            )
        )
    return CallbackList(callbacks)


def _positive_int(mapping: Mapping[str, Any], key: str) -> int:
    value = int(mapping[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive.")
    return value


def _float_value(mapping: Mapping[str, Any], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    return float(value)


def _mean(values: deque[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train joint PPO+DQN 5PL policies from scratch.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--disable-trace-logging", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ppo_path, dqn_path = train_joint_from_scratch(
        config_path=cast(Path, args.config),
        output_root=cast(Path, args.output_root),
        trace_logging=not bool(args.disable_trace_logging),
    )
    print({"ppo_checkpoint": str(ppo_path), "dqn_checkpoint": str(dqn_path)})


if __name__ == "__main__":
    main()
