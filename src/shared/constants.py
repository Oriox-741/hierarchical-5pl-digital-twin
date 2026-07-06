"""Shared constants for Sense/Think/Act/Learn integration."""

from __future__ import annotations

from typing import Final


SRID_WGS84: Final[int] = 4326

DEFAULT_DB_HOST: Final[str] = "localhost"
DEFAULT_DB_PORT: Final[int] = 5432
DEFAULT_DB_USER: Final[str] = "postgres"
DEFAULT_DB_PASSWORD: Final[str] = "1234"
DEFAULT_DB_NAME: Final[str] = "postgres"

DEFAULT_TELEMETRY_BATCH_SIZE: Final[int] = 10_000
DEFAULT_TRACE_BATCH_SIZE: Final[int] = 5_000
DEFAULT_SYNTHETIC_BATCH_SIZE: Final[int] = 250

MIN_LATITUDE: Final[float] = -90.0
MAX_LATITUDE: Final[float] = 90.0
MIN_LONGITUDE: Final[float] = -180.0
MAX_LONGITUDE: Final[float] = 180.0

MIN_NORMALIZED_ACTION: Final[float] = -1.0
MAX_NORMALIZED_ACTION: Final[float] = 1.0

MIN_SENSOR_QUALITY: Final[int] = 0
MAX_SENSOR_QUALITY: Final[int] = 100
