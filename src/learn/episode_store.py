"""Async episode trace store backed by PostgreSQL `sense_episode_traces`."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import asyncpg

from src.sense.db_pool import DatabasePool


JsonMapping = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class EpisodeTransition:
    """One persisted state-action-reward transition."""

    episode_id: UUID
    step_id: int
    recorded_at: datetime
    observation: JsonMapping
    action: JsonMapping
    reward: float
    next_observation: JsonMapping | None = None
    projected_action: JsonMapping | None = None
    terminated: bool = False
    truncated: bool = False
    policy_id: str | None = None
    environment_id: str | None = None
    agent_id: str | None = None
    agent_role: str | None = None
    team_id: str | None = None
    joint_action_id: UUID | None = None
    local_reward: float | None = None
    global_reward: float | None = None
    ppo_local_reward: float | None = None
    dqn_local_reward: float | None = None
    info: JsonMapping = field(default_factory=dict)
    trace_id: UUID = field(default_factory=uuid4)

    def to_copy_row(self) -> tuple[Any, ...]:
        return (
            self.trace_id,
            self.episode_id,
            self.step_id,
            _as_utc(self.recorded_at),
            json.dumps(self.observation, separators=(",", ":"), default=str),
            json.dumps(self.action, separators=(",", ":"), default=str),
            json.dumps(self.next_observation, separators=(",", ":"), default=str)
            if self.next_observation is not None
            else None,
            json.dumps(self.projected_action, separators=(",", ":"), default=str)
            if self.projected_action is not None
            else None,
            self.reward,
            self.terminated,
            self.truncated,
            self.policy_id,
            self.environment_id,
            self.agent_id,
            self.agent_role,
            self.team_id,
            self.joint_action_id,
            self.local_reward,
            self.global_reward,
            self.ppo_local_reward,
            self.dqn_local_reward,
            json.dumps(self.info, separators=(",", ":"), default=str),
        )


class EpisodeStore:
    """Read/write service aligned with the Phase 2 trace table."""

    COPY_COLUMNS: tuple[str, ...] = (
        "trace_id",
        "episode_id",
        "step_id",
        "recorded_at",
        "observation",
        "action",
        "next_observation",
        "projected_action",
        "reward",
        "terminated",
        "truncated",
        "policy_id",
        "environment_id",
        "agent_id",
        "agent_role",
        "team_id",
        "joint_action_id",
        "local_reward",
        "global_reward",
        "ppo_local_reward",
        "dqn_local_reward",
        "info",
    )

    CREATE_STAGE_SQL = """
        CREATE TEMP TABLE IF NOT EXISTS _learn_episode_trace_stage (
            trace_id UUID NOT NULL,
            episode_id UUID NOT NULL,
            step_id INTEGER NOT NULL,
            recorded_at TIMESTAMPTZ(6) NOT NULL,
            observation JSONB NOT NULL,
            action JSONB NOT NULL,
            next_observation JSONB,
            projected_action JSONB,
            reward FLOAT8 NOT NULL,
            terminated BOOLEAN NOT NULL,
            truncated BOOLEAN NOT NULL,
            policy_id TEXT,
            environment_id TEXT,
            agent_id TEXT,
            agent_role TEXT,
            team_id TEXT,
            joint_action_id UUID,
            local_reward FLOAT8,
            global_reward FLOAT8,
            ppo_local_reward FLOAT8,
            dqn_local_reward FLOAT8,
            info JSONB NOT NULL
        ) ON COMMIT DROP;
    """

    INSERT_SQL = """
        INSERT INTO sense_episode_traces (
            trace_id,
            episode_id,
            step_id,
            recorded_at,
            observation,
            action,
            next_observation,
            projected_action,
            reward,
            terminated,
            truncated,
            policy_id,
            environment_id,
            agent_id,
            agent_role,
            team_id,
            joint_action_id,
            local_reward,
            global_reward,
            ppo_local_reward,
            dqn_local_reward,
            info
        )
        SELECT
            trace_id,
            episode_id,
            step_id,
            recorded_at,
            observation,
            action,
            next_observation,
            projected_action,
            reward,
            terminated,
            truncated,
            policy_id,
            environment_id,
            agent_id,
            agent_role,
            team_id,
            joint_action_id,
            local_reward,
            global_reward,
            ppo_local_reward,
            dqn_local_reward,
            info
        FROM _learn_episode_trace_stage
        ON CONFLICT (recorded_at, episode_id, step_id) DO UPDATE SET
            trace_id = EXCLUDED.trace_id,
            observation = EXCLUDED.observation,
            action = EXCLUDED.action,
            next_observation = EXCLUDED.next_observation,
            projected_action = EXCLUDED.projected_action,
            reward = EXCLUDED.reward,
            terminated = EXCLUDED.terminated,
            truncated = EXCLUDED.truncated,
            policy_id = EXCLUDED.policy_id,
            environment_id = EXCLUDED.environment_id,
            agent_id = EXCLUDED.agent_id,
            agent_role = EXCLUDED.agent_role,
            team_id = EXCLUDED.team_id,
            joint_action_id = EXCLUDED.joint_action_id,
            local_reward = EXCLUDED.local_reward,
            global_reward = EXCLUDED.global_reward,
            ppo_local_reward = EXCLUDED.ppo_local_reward,
            dqn_local_reward = EXCLUDED.dqn_local_reward,
            info = EXCLUDED.info,
            persisted_at = now();
    """

    AGENT_COLUMN_MIGRATION_SQL = """
        ALTER TABLE sense_episode_traces
            ADD COLUMN IF NOT EXISTS agent_id TEXT,
            ADD COLUMN IF NOT EXISTS agent_role TEXT,
            ADD COLUMN IF NOT EXISTS team_id TEXT,
            ADD COLUMN IF NOT EXISTS joint_action_id UUID,
            ADD COLUMN IF NOT EXISTS local_reward FLOAT8,
            ADD COLUMN IF NOT EXISTS global_reward FLOAT8,
            ADD COLUMN IF NOT EXISTS ppo_local_reward FLOAT8,
            ADD COLUMN IF NOT EXISTS dqn_local_reward FLOAT8;
    """

    JOINT_TRACE_CONSTRAINT_SQL = """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'chk_sense_episode_traces_joint_complete'
                  AND conrelid = 'sense_episode_traces'::regclass
            ) THEN
                ALTER TABLE sense_episode_traces
                    ADD CONSTRAINT chk_sense_episode_traces_joint_complete
                    CHECK (
                        joint_action_id IS NULL
                        OR (
                            agent_role IS NOT NULL
                            AND local_reward IS NOT NULL
                            AND global_reward IS NOT NULL
                            AND ppo_local_reward IS NOT NULL
                            AND dqn_local_reward IS NOT NULL
                        )
                    )
                    NOT VALID;
            END IF;
        END
        $$;
    """

    QUERY_EPISODE_SQL = """
        SELECT *
        FROM sense_episode_traces
        WHERE episode_id = $1
        ORDER BY
            step_id ASC,
            recorded_at ASC,
            policy_id ASC NULLS LAST,
            agent_role ASC NULLS LAST,
            agent_id ASC NULLS LAST,
            trace_id ASC;
    """

    QUERY_RECENT_SQL = """
        SELECT *
        FROM sense_episode_traces
        WHERE ($1::text IS NULL OR policy_id = $1)
          AND recorded_at >= now() - ($2::FLOAT8 * INTERVAL '1 second')
          AND ($4::text IS NULL OR agent_id = $4)
          AND ($5::text IS NULL OR agent_role = $5)
        ORDER BY
            recorded_at DESC,
            episode_id ASC,
            step_id ASC,
            policy_id ASC NULLS LAST,
            agent_role ASC NULLS LAST,
            agent_id ASC NULLS LAST,
            trace_id ASC
        LIMIT $3;
    """

    QUERY_RANGE_SQL = """
        SELECT *
        FROM sense_episode_traces
        WHERE recorded_at >= $1
          AND recorded_at < $2
          AND ($3::text IS NULL OR policy_id = $3)
          AND ($4::text IS NULL OR agent_id = $4)
          AND ($5::text IS NULL OR agent_role = $5)
        ORDER BY
            recorded_at ASC,
            episode_id ASC,
            step_id ASC,
            policy_id ASC NULLS LAST,
            agent_role ASC NULLS LAST,
            agent_id ASC NULLS LAST,
            trace_id ASC;
    """

    def __init__(self, db: DatabasePool | asyncpg.Pool, *, batch_size: int = 5_000) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        self._pool = db.pool if isinstance(db, DatabasePool) else db
        self._batch_size = batch_size
        self._schema_ready = False

    async def write_many(self, transitions: Iterable[EpisodeTransition]) -> int:
        inserted = 0
        batch: list[EpisodeTransition] = []
        for transition in transitions:
            self._validate(transition)
            batch.append(transition)
            if len(batch) >= self._batch_size:
                inserted += await self._write_batch(_collision_safe_transitions(batch))
                batch.clear()
        if batch:
            inserted += await self._write_batch(_collision_safe_transitions(batch))
        return inserted

    async def write_one(self, transition: EpisodeTransition) -> int:
        return await self.write_many((transition,))

    async def read_episode(self, episode_id: UUID) -> list[EpisodeTransition]:
        rows = await self._pool.fetch(self.QUERY_EPISODE_SQL, episode_id)
        return [self._from_record(row) for row in rows]

    async def read_recent(
        self,
        *,
        within_seconds: float,
        limit: int = 10_000,
        policy_id: str | None = None,
        agent_id: str | None = None,
        agent_role: str | None = None,
    ) -> list[EpisodeTransition]:
        rows = await self._pool.fetch(self.QUERY_RECENT_SQL, policy_id, within_seconds, limit, agent_id, agent_role)
        return [self._from_record(row) for row in rows]

    async def read_range(
        self,
        *,
        start: datetime,
        end: datetime,
        policy_id: str | None = None,
        agent_id: str | None = None,
        agent_role: str | None = None,
    ) -> list[EpisodeTransition]:
        rows = await self._pool.fetch(
            self.QUERY_RANGE_SQL,
            _as_utc(start),
            _as_utc(end),
            policy_id,
            agent_id,
            agent_role,
        )
        return [self._from_record(row) for row in rows]

    async def _write_batch(self, transitions: Sequence[EpisodeTransition]) -> int:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                if not self._schema_ready:
                    await connection.execute(self.AGENT_COLUMN_MIGRATION_SQL)
                    await connection.execute(self.JOINT_TRACE_CONSTRAINT_SQL)
                    self._schema_ready = True
                await connection.execute(self.CREATE_STAGE_SQL)
                await connection.execute("TRUNCATE TABLE _learn_episode_trace_stage;")
                await connection.copy_records_to_table(
                    "_learn_episode_trace_stage",
                    records=[transition.to_copy_row() for transition in transitions],
                    columns=self.COPY_COLUMNS,
                )
                await connection.execute(self.INSERT_SQL)
        return len(transitions)

    @staticmethod
    def _validate(transition: EpisodeTransition) -> None:
        if transition.step_id < 0:
            raise ValueError("step_id must not be negative.")
        if transition.terminated and transition.truncated:
            raise ValueError("terminated and truncated cannot both be true.")
        if transition.joint_action_id is not None:
            missing = [
                name
                for name, value in {
                    "agent_role": transition.agent_role,
                    "local_reward": transition.local_reward,
                    "global_reward": transition.global_reward,
                    "ppo_local_reward": transition.ppo_local_reward,
                    "dqn_local_reward": transition.dqn_local_reward,
                }.items()
                if value is None
            ]
            if missing:
                raise ValueError(f"joint trace row is missing required fields: {', '.join(missing)}.")

    @staticmethod
    def _from_record(row: asyncpg.Record) -> EpisodeTransition:
        return EpisodeTransition(
            trace_id=row["trace_id"],
            episode_id=row["episode_id"],
            step_id=row["step_id"],
            recorded_at=row["recorded_at"],
            observation=_json_mapping(row["observation"]),
            action=_json_mapping(row["action"]),
            next_observation=_optional_json_mapping(row["next_observation"]),
            projected_action=_optional_json_mapping(row["projected_action"]),
            reward=float(row["reward"]),
            terminated=bool(row["terminated"]),
            truncated=bool(row["truncated"]),
            policy_id=row["policy_id"],
            environment_id=row["environment_id"],
            agent_id=_record_value(row, "agent_id"),
            agent_role=_record_value(row, "agent_role"),
            team_id=_record_value(row, "team_id"),
            joint_action_id=_record_value(row, "joint_action_id"),
            local_reward=_optional_float(_record_value(row, "local_reward")),
            global_reward=_optional_float(_record_value(row, "global_reward")),
            ppo_local_reward=_optional_float(_record_value(row, "ppo_local_reward")),
            dqn_local_reward=_optional_float(_record_value(row, "dqn_local_reward")),
            info=_json_mapping(row["info"]),
        )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _collision_safe_transitions(transitions: Sequence[EpisodeTransition]) -> list[EpisodeTransition]:
    """Preserve both role rows when callers accidentally reuse a step timestamp."""

    seen: dict[tuple[datetime, UUID, int], int] = {}
    safe: list[EpisodeTransition] = []
    for transition in transitions:
        base_recorded_at = _as_utc(transition.recorded_at)
        key = (base_recorded_at, transition.episode_id, transition.step_id)
        offset = seen.get(key, 0)
        seen[key] = offset + 1
        if offset == 0 and base_recorded_at == transition.recorded_at:
            safe.append(transition)
            continue
        safe.append(replace(transition, recorded_at=base_recorded_at + timedelta(microseconds=offset)))
    return safe


def _json_mapping(value: Any) -> JsonMapping:
    if value is None:
        return {}
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, Mapping):
            return parsed
        raise TypeError("Expected JSON object.")
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"Expected mapping, got {type(value).__name__}.")


def _optional_json_mapping(value: Any) -> JsonMapping | None:
    if value is None:
        return None
    return _json_mapping(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _record_value(row: asyncpg.Record, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError):
        return default
