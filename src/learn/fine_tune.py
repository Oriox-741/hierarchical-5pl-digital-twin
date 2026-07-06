"""Fine-tune existing Stable-Baselines3 checkpoints on updated 5PL scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import DummyVecEnv

from src.act.env_5pl import ActionMode, EnvironmentConfig, FivePLDigitalTwinEnv
from src.learn.curriculum import CurriculumManager, PerformanceSummary
from src.learn.model_registry import Algorithm, ModelRegistry
from src.learn.sb3_trace_callback import TrainingTraceCallback


DEFAULT_JOINT_CONFIG = Path("configs/training_joint.json")
DEFAULT_OUTPUT_DIR = Path("models/checkpoints/fine_tuned")


def load_config(path: Path, algorithm: Algorithm) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    shared = raw.get("shared_global_parameters", {})
    section_name = "ppo_hyperparameters" if algorithm == "ppo" else "dqn_hyperparameters"
    local = raw.get(section_name, {})
    if not isinstance(shared, dict) or not isinstance(local, dict):
        raise ValueError(f"{path} must define shared_global_parameters and {section_name}.")
    return {**shared, **local}


def build_env(action_mode: ActionMode, seed: int | None, max_steps: int = 288) -> Monitor:
    return Monitor(
        FivePLDigitalTwinEnv(
            EnvironmentConfig(
                action_mode=action_mode,
                random_seed=seed,
                max_steps=max_steps,
            )
        )
    )


def load_checkpoint(
    path: Path,
    algorithm: Algorithm,
    env: DummyVecEnv,
    *,
    config: dict[str, Any],
) -> BaseAlgorithm:
    learning_rate = _learning_rate_from_config(config)
    custom_objects = {
        "learning_rate": learning_rate,
        "lr_schedule": constant_fn(learning_rate),
    }

    model: BaseAlgorithm
    if algorithm == "ppo":
        model = PPO.load(path, env=env, custom_objects=custom_objects)
    elif algorithm == "dqn":
        model = DQN.load(path, env=env, custom_objects=custom_objects)
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}")

    model.learning_rate = learning_rate
    model.lr_schedule = constant_fn(learning_rate)
    _override_optimizer_learning_rate(model, learning_rate)
    return model


def fine_tune(
    *,
    algorithm: Algorithm,
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
    timesteps: int | None,
    seed: int | None,
    register_active: bool,
    trace_logging: bool = True,
    trace_batch_size: int = 256,
) -> Path:
    config = load_config(config_path, algorithm)
    learning_rate = _learning_rate_from_config(config)
    total_timesteps = timesteps if timesteps is not None else int(config["total_timesteps"])
    action_mode: ActionMode = cast(ActionMode, "continuous" if algorithm == "ppo" else "discrete")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([lambda: build_env(action_mode, seed)])
    try:
        model = load_checkpoint(checkpoint_path, algorithm, env, config=config)
        callbacks: list[BaseCallback] = []
        if trace_logging:
            callbacks.append(
                TrainingTraceCallback(
                    algorithm=algorithm,
                    policy_id=f"{algorithm}_5pl_finetune",
                    environment_id=f"{algorithm}_fine_tune",
                    agent_id="global_controller",
                    agent_role="continuous_control" if algorithm == "ppo" else "tactical_dispatch",
                    team_id="fine_tune_single_agent",
                    flush_batch_size=trace_batch_size,
                )
            )

        callback = CallbackList(callbacks) if callbacks else None
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=callback)

        output_path = output_dir / f"{algorithm}_5pl_finetuned.zip"
        model.save(output_path)
    finally:
        env.close()

    registry = ModelRegistry()
    registry.register(
        algorithm=algorithm,
        path=output_path,
        agent_role="continuous_control" if algorithm == "ppo" else "tactical_dispatch",
        metrics={},
        metadata={
            "source_checkpoint": str(checkpoint_path),
            "timesteps": total_timesteps,
            "config_path": str(config_path),
            "learning_rate": learning_rate,
        },
        status="active" if register_active else "candidate",
    )
    return output_path


def curriculum_hint(
    *,
    mean_reward: float,
    service_level: float,
    safety_potential: float,
    completion_rate: float,
    episodes: int,
) -> dict[str, Any]:
    manager = CurriculumManager()
    difficulty = manager.choose_difficulty(
        PerformanceSummary(
            mean_reward=mean_reward,
            service_level=service_level,
            safety_potential=safety_potential,
            completion_rate=completion_rate,
            episodes=episodes,
        )
    )
    return {
        "stage": difficulty.stage.value,
        "disruption_probability": difficulty.disruption_probability,
        "max_disruption_severity": difficulty.max_disruption_severity,
        "demand_multiplier": difficulty.demand_multiplier,
        "capacity_loss_fraction": difficulty.capacity_loss_fraction,
        "metadata": difficulty.metadata,
    }


def _learning_rate_from_config(config: dict[str, Any]) -> float:
    value = config.get("learning_rate")
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("Fine-tuning config must define a numeric learning_rate.")
    learning_rate = float(value)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be greater than zero.")
    return learning_rate


def _override_optimizer_learning_rate(model: BaseAlgorithm, learning_rate: float) -> None:
    policy = getattr(model, "policy", None)
    optimizer = getattr(policy, "optimizer", None)
    param_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(param_groups, list):
        return
    for param_group in param_groups:
        if isinstance(param_group, dict):
            param_group["lr"] = learning_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PPO/DQN checkpoints for the 5PL digital twin.")
    parser.add_argument("--algorithm", choices=("ppo", "dqn"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--active", action="store_true", help="Mark the fine-tuned model as active in the registry.")
    parser.add_argument("--disable-trace-logging", action="store_true")
    parser.add_argument("--trace-batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    algorithm = cast(Algorithm, args.algorithm)
    config_path = args.config or DEFAULT_JOINT_CONFIG
    output_path = fine_tune(
        algorithm=algorithm,
        checkpoint_path=args.checkpoint,
        config_path=config_path,
        output_dir=args.output_dir,
        timesteps=args.timesteps,
        seed=args.seed,
        register_active=args.active,
        trace_logging=not args.disable_trace_logging,
        trace_batch_size=args.trace_batch_size,
    )
    print(f"Fine-tuned model saved to {output_path}")


if __name__ == "__main__":
    main()
