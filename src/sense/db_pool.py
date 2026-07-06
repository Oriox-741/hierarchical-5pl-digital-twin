"""Asynchronous PostgreSQL connection-pool lifecycle for the Sense layer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Final, Self

import asyncpg


DEFAULT_HOST: Final[str] = "localhost"
DEFAULT_PORT: Final[int] = 5432
DEFAULT_USER: Final[str] = "postgres"
DEFAULT_PASSWORD: Final[str] = "1234"
DEFAULT_DATABASE: Final[str] = "postgres"


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Connection settings for the PostgreSQL/TimescaleDB Sense database."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    user: str = DEFAULT_USER
    password: str = DEFAULT_PASSWORD
    database: str = DEFAULT_DATABASE
    min_size: int = 2
    max_size: int = 20
    command_timeout: float = 60.0
    statement_cache_size: int = 1_024


class DatabasePool:
    """Robust asyncpg pool manager with explicit startup/shutdown semantics."""

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        self._config = config or DatabaseConfig()
        self._pool: asyncpg.Pool | None = None
        self._lock = asyncio.Lock()

    @property
    def pool(self) -> asyncpg.Pool:
        """Return the active pool or fail fast when lifecycle was not started."""
        if self._pool is None:
            raise RuntimeError("Database pool is not started. Call startup() first.")
        return self._pool

    async def startup(self) -> asyncpg.Pool:
        """Create the asyncpg connection pool exactly once."""
        async with self._lock:
            if self._pool is not None:
                return self._pool

            self._pool = await asyncpg.create_pool(
                host=self._config.host,
                port=self._config.port,
                user=self._config.user,
                password=self._config.password,
                database=self._config.database,
                min_size=self._config.min_size,
                max_size=self._config.max_size,
                command_timeout=self._config.command_timeout,
                statement_cache_size=self._config.statement_cache_size,
                init=self._initialize_connection,
            )
            return self._pool

    async def shutdown(self) -> None:
        """Close all pooled connections cleanly and make shutdown idempotent."""
        async with self._lock:
            if self._pool is None:
                return

            pool = self._pool
            self._pool = None
            await pool.close()

    def acquire(self) -> asyncpg.pool.PoolAcquireContext:
        """Return asyncpg's acquisition context for callers that need a connection."""
        return self.pool.acquire()

    async def execute(self, sql: str, *args: object, timeout: float | None = None) -> str:
        """Execute a statement through the active pool."""
        return await self.pool.execute(sql, *args, timeout=timeout)

    async def fetch(self, sql: str, *args: object, timeout: float | None = None) -> list[asyncpg.Record]:
        """Fetch many rows through the active pool."""
        return await self.pool.fetch(sql, *args, timeout=timeout)

    async def fetchrow(
        self,
        sql: str,
        *args: object,
        timeout: float | None = None,
    ) -> asyncpg.Record | None:
        """Fetch one row through the active pool."""
        return await self.pool.fetchrow(sql, *args, timeout=timeout)

    async def __aenter__(self) -> Self:
        await self.startup()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.shutdown()

    @staticmethod
    async def _initialize_connection(connection: asyncpg.Connection) -> None:
        """Prepare every session for PostGIS/TimescaleDB-heavy workloads."""
        await connection.execute("SET TIME ZONE 'UTC';")
