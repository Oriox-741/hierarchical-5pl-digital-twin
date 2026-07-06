"""State-action-reward trace persistence for downstream Learn-layer workflows."""

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
class EpisodeTraceRecord:
    """One reinforcement-learning transition emitted by Act/Think execution."""

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
        """Convert to a COPY-compatible row matching TraceWriterService.COPY_COLUMNS."""
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


class TraceWriterService:
    """Bulk persistence service for state-action-reward transition traces."""

    TABLE_NAME = "sense_episode_traces"

    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS sense_episode_traces (
            trace_id UUID NOT NULL,
            episode_id UUID NOT NULL,
            step_id INTEGER NOT NULL,
            recorded_at TIMESTAMPTZ(6) NOT NULL,
            observation JSONB NOT NULL,
            action JSONB NOT NULL,
            next_observation JSONB,
            projected_action JSONB,
            reward FLOAT8 NOT NULL,
            terminated BOOLEAN NOT NULL DEFAULT FALSE,
            truncated BOOLEAN NOT NULL DEFAULT FALSE,
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
            info JSONB NOT NULL DEFAULT '{}'::jsonb,
            persisted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (recorded_at, episode_id, step_id),
            UNIQUE (recorded_at, trace_id),
            CHECK (NOT (terminated AND truncated)),
            CHECK (step_id >= 0),
            CONSTRAINT chk_sense_episode_traces_joint_complete
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
        );
    """

    ALTER_TABLE_SQL = """
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

    CREATE_INDEXES_SQL = (
        """
        CREATE INDEX IF NOT EXISTS idx_sense_episode_traces_recorded_at_brin
            ON sense_episode_traces
            USING BRIN (recorded_at)
            WITH (pages_per_range = 64);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_sense_episode_traces_episode_step
            ON sense_episode_traces (episode_id, step_id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_sense_episode_traces_policy_time
            ON sense_episode_traces (policy_id, recorded_at DESC)
            WHERE policy_id IS NOT NULL;
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_sense_episode_traces_agent_time
            ON sense_episode_traces (agent_id, recorded_at DESC)
            WHERE agent_id IS NOT NULL;
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_sense_episode_traces_role_time
            ON sense_episode_traces (agent_role, recorded_at DESC)
            WHERE agent_role IS NOT NULL;
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_sense_episode_traces_policy_role_time
            ON sense_episode_traces (policy_id, agent_role, recorded_at DESC)
            WHERE policy_id IS NOT NULL AND agent_role IS NOT NULL;
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_sense_episode_traces_team_time
            ON sense_episode_traces (team_id, recorded_at DESC)
            WHERE team_id IS NOT NULL;
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_sense_episode_traces_joint_action
            ON sense_episode_traces (joint_action_id)
            WHERE joint_action_id IS NOT NULL;
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_sense_episode_traces_info_gin
            ON sense_episode_traces
            USING GIN (info jsonb_path_ops);
        """,
    )

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

    CREATE_TEMP_TABLE_SQL = """
        CREATE TEMP TABLE IF NOT EXISTS _episode_trace_stage (
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
        FROM _episode_trace_stage
        ON CONFLICT (recorded_at, episode_id, step_id) DO UPDATE SET
            trace_id = EXCLUDED.trace_id,
            recorded_at = EXCLUDED.recorded_at,
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

    TRUNCATE_TEMP_TABLE_SQL = "TRUNCATE TABLE _episode_trace_stage;"

    def __init__(self, db: DatabasePool | asyncpg.Pool, *, batch_size: int = 5_000) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        self._pool = db.pool if isinstance(db, DatabasePool) else db
        self._batch_size = batch_size
        self._schema_ready = False

    async def ensure_schema(self) -> None:
        """Create the trace table and indexes needed by the future Learn layer."""
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(self.CREATE_TABLE_SQL)
                await connection.execute(self.ALTER_TABLE_SQL)
                await connection.execute(self.JOINT_TRACE_CONSTRAINT_SQL)
                self._schema_ready = True
                for statement in self.CREATE_INDEXES_SQL:
                    await connection.execute(statement)

    async def write_one(self, record: EpisodeTraceRecord, *, ensure_schema: bool = False) -> int:
        """Persist one transition trace."""
        return await self.write_many((record,), ensure_schema=ensure_schema)

    async def write_many(
        self,
        records: Iterable[EpisodeTraceRecord],
        *,
        ensure_schema: bool = False,
    ) -> int:
        """Persist transition traces in bounded COPY batches."""
        if ensure_schema:
            await self.ensure_schema()

        inserted = 0
        batch: list[EpisodeTraceRecord] = []
        for record in records:
            self._validate_record(record)
            batch.append(record)
            if len(batch) >= self._batch_size:
                inserted += await self._write_batch(_collision_safe_records(batch))
                batch.clear()

        if batch:
            inserted += await self._write_batch(_collision_safe_records(batch))

        return inserted

    async def write_rows(
        self,
        rows: Iterable[JsonMapping],
        *,
        ensure_schema: bool = False,
    ) -> int:
        """Accept dict-like rows and persist them as typed episode traces."""
        return await self.write_many(
            (self._record_from_mapping(row) for row in rows),
            ensure_schema=ensure_schema,
        )

    async def _write_batch(self, records: Sequence[EpisodeTraceRecord]) -> int:
        if not records:
            return 0

        async with self._pool.acquire() as connection:
            async with connection.transaction():
                if not self._schema_ready:
                    await connection.execute(self.ALTER_TABLE_SQL)
                    await connection.execute(self.JOINT_TRACE_CONSTRAINT_SQL)
                    self._schema_ready = True
                await connection.execute(self.CREATE_TEMP_TABLE_SQL)
                await connection.execute(self.TRUNCATE_TEMP_TABLE_SQL)
                await connection.copy_records_to_table(
                    "_episode_trace_stage",
                    records=[record.to_copy_row() for record in records],
                    columns=self.COPY_COLUMNS,
                )
                await connection.execute(self.INSERT_SQL)

        return len(records)

    @staticmethod
    def _record_from_mapping(row: JsonMapping) -> EpisodeTraceRecord:
        return EpisodeTraceRecord(
            trace_id=_parse_uuid(row.get("trace_id", uuid4())),
            episode_id=_parse_uuid(row["episode_id"]),
            step_id=int(row["step_id"]),
            recorded_at=_parse_datetime(row.get("recorded_at", datetime.now(timezone.utc))),
            observation=_mapping_or_empty(row.get("observation")),
            action=_mapping_or_empty(row.get("action")),
            next_observation=_optional_mapping(row.get("next_observation")),
            projected_action=_optional_mapping(row.get("projected_action")),
            reward=float(row["reward"]),
            terminated=bool(row.get("terminated", False)),
            truncated=bool(row.get("truncated", False)),
            policy_id=_optional_string(row.get("policy_id")),
            environment_id=_optional_string(row.get("environment_id")),
            agent_id=_optional_string(row.get("agent_id")),
            agent_role=_optional_string(row.get("agent_role")),
            team_id=_optional_string(row.get("team_id")),
            joint_action_id=_optional_uuid(row.get("joint_action_id")),
            local_reward=_optional_float(row.get("local_reward")),
            global_reward=_optional_float(row.get("global_reward")),
            ppo_local_reward=_optional_float(row.get("ppo_local_reward")),
            dqn_local_reward=_optional_float(row.get("dqn_local_reward")),
            info=_mapping_or_empty(row.get("info")),
        )

    @staticmethod
    def _validate_record(record: EpisodeTraceRecord) -> None:
        if record.step_id < 0:
            raise ValueError("step_id must not be negative.")
        if record.terminated and record.truncated:
            raise ValueError("terminated and truncated cannot both be true.")
        if record.joint_action_id is not None:
            missing = [
                name
                for name, value in {
                    "agent_role": record.agent_role,
                    "local_reward": record.local_reward,
                    "global_reward": record.global_reward,
                    "ppo_local_reward": record.ppo_local_reward,
                    "dqn_local_reward": record.dqn_local_reward,
                }.items()
                if value is None
            ]
            if missing:
                raise ValueError(f"joint trace row is missing required fields: {', '.join(missing)}.")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _collision_safe_records(records: Sequence[EpisodeTraceRecord]) -> list[EpisodeTraceRecord]:
    """Preserve multi-role rows when callers accidentally reuse a step timestamp."""

    seen: dict[tuple[datetime, UUID, int], int] = {}
    safe: list[EpisodeTraceRecord] = []
    for record in records:
        base_recorded_at = _as_utc(record.recorded_at)
        key = (base_recorded_at, record.episode_id, record.step_id)
        offset = seen.get(key, 0)
        seen[key] = offset + 1
        if offset == 0 and base_recorded_at == record.recorded_at:
            safe.append(record)
            continue
        safe.append(replace(record, recorded_at=base_recorded_at + timedelta(microseconds=offset)))
    return safe


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, str):
        return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    raise TypeError(f"Unsupported datetime value: {value!r}")


def _parse_uuid(value: object) -> UUID:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    raise TypeError(f"Unsupported UUID value: {value!r}")


def _optional_uuid(value: object) -> UUID | None:
    if value is None:
        return None
    return _parse_uuid(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"Expected numeric or string float value, got {type(value).__name__}.")
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"Expected numeric or string float value, got {type(value).__name__}.")


def _mapping_or_empty(value: object) -> JsonMapping:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"Expected mapping, got {type(value).__name__}.")


def _optional_mapping(value: object) -> JsonMapping | None:
    if value is None:
        return None
    return _mapping_or_empty(value)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
