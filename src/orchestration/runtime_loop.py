"""Closed-loop autonomous Sense-Think-Act-Learn runtime."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Literal, cast
from uuid import UUID, uuid4

import numpy as np

from src.act.env_5pl import ActionMode, EnvironmentConfig, FivePLDigitalTwinEnv
from src.learn.episode_store import EpisodeStore
from src.learn.model_registry import Algorithm
from src.orchestration.decision_audit import DecisionAuditRecord, DecisionAuditService
from src.orchestration.policy_service import PolicyService
from src.sense.db_pool import DatabaseConfig, DatabasePool


RuntimeAlgorithm = Literal["ppo", "dqn", "joint"]


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    algorithm: RuntimeAlgorithm = "joint"
    action_mode: ActionMode = "joint"
    max_steps: int = 288
    decision_interval: float = 300.0
    seed: int | None = 42
    environment_id: str = "closed_loop_5pl_v1"
    agent_id: str = "joint_controller"
    agent_role: str = "control_tower"
    team_id: str = "joint_from_scratch"
    fallback_to_heuristic: bool = True


@dataclass(frozen=True, slots=True)
class RuntimeSummary:
    episode_id: UUID
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool


class AutonomousRuntimeLoop:
    """Coordinates Sense snapshots, Think inference, Act execution, and Learn logging."""

    def __init__(
        self,
        *,
        db: DatabasePool,
        policy_service: PolicyService | None = None,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.db = db
        self.config = config or RuntimeConfig()
        self.policy_service = policy_service or PolicyService()
        self.episode_store = EpisodeStore(db)
        self.audit_service = DecisionAuditService(self.episode_store)
        self.env = FivePLDigitalTwinEnv(
            EnvironmentConfig(
                action_mode=self.config.action_mode,
                decision_interval=self.config.decision_interval,
                max_steps=self.config.max_steps,
                random_seed=self.config.seed,
            )
        )

    async def run_episode(self) -> RuntimeSummary:
        episode_id = uuid4()
        observation, _ = self.env.reset(seed=self.config.seed)
        total_reward = 0.0
        terminated = False
        truncated = False
        step_id = 0

        while not terminated and not truncated and step_id < self.config.max_steps:
            if self.config.action_mode == "joint" or self.config.algorithm == "joint":
                inference = await asyncio.to_thread(
                    self.policy_service.predict_joint,
                    observation,
                    deterministic=True,
                    fallback_to_heuristic=self.config.fallback_to_heuristic,
                    agent_id=self.config.agent_id,
                    agent_role=self.config.agent_role,
                )
            else:
                inference = await asyncio.to_thread(
                    self.policy_service.predict,
                    observation,
                    algorithm=cast(Algorithm, self.config.algorithm),
                    deterministic=True,
                    fallback_to_heuristic=self.config.fallback_to_heuristic,
                    agent_id=self.config.agent_id,
                    agent_role=self.config.agent_role,
                )
            joint_action_id = uuid4()
            next_observation, reward, terminated, truncated, info = self.env.step(inference.action)
            await self.audit_service.record(
                DecisionAuditRecord(
                    episode_id=episode_id,
                    step_id=step_id,
                    observation=np.asarray(observation, dtype=np.float32),
                    action=inference.action,
                    reward=reward,
                    next_observation=np.asarray(next_observation, dtype=np.float32),
                    terminated=terminated,
                    truncated=truncated,
                    policy_id=str(inference.algorithm),
                    environment_id=self.config.environment_id,
                    agent_id=inference.agent_id,
                    agent_role=inference.agent_role,
                    team_id=self.config.team_id,
                    joint_action_id=joint_action_id,
                    projected_action=info.get("projected_action"),
                    info={
                        "model_path": inference.model_path,
                        "deterministic": inference.deterministic,
                        "agent_id": inference.agent_id,
                        "agent_role": inference.agent_role,
                        "team_id": self.config.team_id,
                        "joint_action_id": str(joint_action_id),
                        "projection_reasons": info.get("projection_reasons", ()),
                        "projected": info.get("projected", False),
                        "blocked": info.get("blocked", False),
                        "reward_components": info.get("reward_components", {}),
                        "simulation_time": info.get("simulation_time"),
                    },
                )
            )
            observation = next_observation
            total_reward += reward
            step_id += 1

        return RuntimeSummary(
            episode_id=episode_id,
            steps=step_id,
            total_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
        )


async def run_from_cli(config: RuntimeConfig) -> RuntimeSummary:
    db = DatabasePool(DatabaseConfig())
    await db.startup()
    try:
        loop = AutonomousRuntimeLoop(db=db, config=config)
        return await loop.run_episode()
    finally:
        await db.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the autonomous 5PL closed loop.")
    parser.add_argument("--algorithm", choices=("ppo", "dqn", "joint"), default="joint")
    parser.add_argument("--action-mode", choices=("continuous", "discrete", "joint"), default="joint")
    parser.add_argument("--max-steps", type=int, default=288)
    parser.add_argument("--decision-interval", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent-id", type=str, default="global_controller")
    parser.add_argument("--agent-role", type=str, default="control_tower")
    parser.add_argument("--team-id", type=str, default="joint_from_scratch")
    parser.add_argument("--no-heuristic-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = asyncio.run(
        run_from_cli(
            RuntimeConfig(
                algorithm=cast(RuntimeAlgorithm, args.algorithm),
                action_mode=cast(ActionMode, args.action_mode),
                max_steps=args.max_steps,
                decision_interval=args.decision_interval,
                seed=args.seed,
                agent_id=args.agent_id,
                agent_role=args.agent_role,
                team_id=args.team_id,
                fallback_to_heuristic=not args.no_heuristic_fallback,
            )
        )
    )
    print(
        {
            "episode_id": str(summary.episode_id),
            "steps": summary.steps,
            "total_reward": summary.total_reward,
            "terminated": summary.terminated,
            "truncated": summary.truncated,
        }
    )


if __name__ == "__main__":
    main()
