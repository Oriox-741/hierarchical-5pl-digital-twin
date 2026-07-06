"""Benchmark and disruption scenario runner for the closed-loop digital twin."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from src.act.disruptions import DisruptionEvent, DisruptionKind
from src.act.env_5pl import ActionMode, EnvironmentConfig, FivePLDigitalTwinEnv
from src.learn.curriculum import CurriculumManager, PerformanceSummary
from src.orchestration.policy_service import PolicyService


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    name: str
    algorithm: str
    steps: int
    total_reward: float
    service_level: float
    safety_potential: float
    terminated: bool
    truncated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "algorithm": self.algorithm,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "service_level": self.service_level,
            "safety_potential": self.safety_potential,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


class ScenarioRunner:
    """Run benchmark scenarios with active models or heuristic fallback."""

    def __init__(self, policy_service: PolicyService | None = None) -> None:
        self.policy_service = policy_service or PolicyService()

    def run(
        self,
        *,
        name: str,
        algorithm: str,
        max_steps: int,
        seed: int | None,
        inject_disruptions: bool,
    ) -> ScenarioResult:
        action_mode: ActionMode = "joint" if algorithm == "joint" else ("continuous" if algorithm == "ppo" else "discrete")
        env = FivePLDigitalTwinEnv(
            EnvironmentConfig(
                action_mode=action_mode,
                max_steps=max_steps,
                random_seed=seed,
            )
        )
        observation, _ = env.reset(seed=seed)
        agent_role = (
            "control_tower"
            if algorithm == "joint"
            else ("continuous_control" if algorithm == "ppo" else "tactical_dispatch")
        )
        if inject_disruptions:
            self._inject_curriculum_disruptions(env)

        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        steps = 0
        while not terminated and not truncated and steps < max_steps:
            if algorithm == "joint":
                inference = self.policy_service.predict_joint(
                    observation,
                    fallback_to_heuristic=True,
                    agent_id="scenario_runner",
                    agent_role=agent_role,
                )
            else:
                inference = self.policy_service.predict(
                    observation,
                    algorithm="ppo" if algorithm == "ppo" else "dqn",
                    fallback_to_heuristic=True,
                    agent_id="scenario_runner",
                    agent_role=agent_role,
                )
            observation, reward, terminated, truncated, info = env.step(inference.action)
            total_reward += reward
            steps += 1

        snapshot = info.get("snapshot", env.simulation.snapshot())
        env.close()
        return ScenarioResult(
            name=name,
            algorithm=algorithm,
            steps=steps,
            total_reward=total_reward,
            service_level=float(snapshot.get("service_level", 0.0)),
            safety_potential=float(snapshot.get("network_safety_potential", 0.0)),
            terminated=terminated,
            truncated=truncated,
        )

    def _inject_curriculum_disruptions(self, env: FivePLDigitalTwinEnv) -> None:
        manager = CurriculumManager()
        difficulty = manager.choose_difficulty(
            PerformanceSummary(
                mean_reward=0.0,
                service_level=0.95,
                safety_potential=0.10,
                completion_rate=0.90,
                episodes=1,
            )
        )
        target_ids: list[UUID] = list(env.simulation.hubs.keys())
        for event in manager.build_disruptions(
            difficulty=difficulty,
            target_node_ids=target_ids,
            start_time=env.config.decision_interval,
            interval=env.config.decision_interval,
        ):
            env.simulation.schedule_disruption(event)
        route_arcs = list(env.simulation.route_network.arcs.keys())
        if route_arcs:
            origin, destination = route_arcs[0]
            env.simulation.schedule_disruption(
                DisruptionEvent(
                    event_id=uuid4(),
                    kind=DisruptionKind.ARC_DELAY,
                    starts_at=env.config.decision_interval,
                    duration=env.config.decision_interval * 2.0,
                    severity=0.50,
                    target_id=origin,
                    payload={"origin_node_id": str(origin), "destination_node_id": str(destination)},
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 5PL benchmark/disruption scenarios.")
    parser.add_argument("--algorithm", choices=("ppo", "dqn", "joint"), default="joint")
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--max-steps", type=int, default=288)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disruptions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = ScenarioRunner().run(
        name=args.name,
        algorithm=args.algorithm,
        max_steps=args.max_steps,
        seed=args.seed,
        inject_disruptions=args.disruptions,
    )
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
