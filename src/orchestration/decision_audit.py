"""Decision audit persistence for observation, action, reward, and projection details."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from src.learn.episode_store import EpisodeStore, EpisodeTransition


@dataclass(frozen=True, slots=True)
class DecisionAuditRecord:
    episode_id: UUID
    step_id: int
    observation: np.ndarray
    action: np.ndarray | int | Mapping[str, Any]
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    policy_id: str | None
    environment_id: str
    agent_id: str | None = "global_controller"
    agent_role: str | None = "control_tower"
    team_id: str | None = "joint_from_scratch"
    joint_action_id: UUID | None = None
    projected_action: dict[str, Any] | None = None
    info: dict[str, Any] = field(default_factory=dict)
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DecisionAuditService:
    """Converts runtime decisions into Learn-layer episode traces."""

    def __init__(self, episode_store: EpisodeStore) -> None:
        self.episode_store = episode_store

    async def record(self, audit: DecisionAuditRecord) -> int:
        reward_components = audit.info.get("reward_components", {})
        if not isinstance(reward_components, dict):
            reward_components = {}
        ppo_local_reward = _optional_float(reward_components.get("ppo_local"))
        dqn_local_reward = _optional_float(reward_components.get("dqn_local"))
        global_reward = _optional_float(reward_components.get("global")) or audit.reward
        transition = EpisodeTransition(
            trace_id=uuid4(),
            episode_id=audit.episode_id,
            step_id=audit.step_id,
            recorded_at=audit.recorded_at,
            observation={"vector": _vector_list(audit.observation)},
            action=_action_payload(audit.action),
            next_observation={"vector": _vector_list(audit.next_observation)},
            projected_action=audit.projected_action,
            reward=audit.reward,
            terminated=audit.terminated,
            truncated=audit.truncated,
            policy_id=audit.policy_id,
            environment_id=audit.environment_id,
            agent_id=audit.agent_id,
            agent_role=audit.agent_role,
            team_id=audit.team_id,
            joint_action_id=audit.joint_action_id,
            local_reward=audit.reward,
            global_reward=global_reward,
            ppo_local_reward=ppo_local_reward,
            dqn_local_reward=dqn_local_reward,
            info=audit.info,
        )
        return await self.episode_store.write_one(transition)


def _vector_list(vector: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(vector, dtype=np.float32).reshape(-1)]


def _action_payload(action: np.ndarray | int | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(action, Mapping):
        output: dict[str, Any] = {}
        continuous = action.get("continuous")
        if continuous is not None:
            output["continuous"] = _vector_list(np.asarray(continuous, dtype=np.float32))
        discrete = action.get("discrete")
        if discrete is not None:
            output["discrete"] = int(np.asarray(discrete).reshape(-1)[0])
        return output
    if isinstance(action, int):
        return {"discrete": action}
    return {"vector": _vector_list(action)}


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    return float(value)
