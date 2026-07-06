"""High-throughput async ingestion for ubiquitous IIoT asset telemetry."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import asyncpg


JsonMapping = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    """Generic telemetry envelope for any tracked asset, device, hub, drone, or pallet."""

    recorded_at: datetime
    container_id: UUID
    device_id: UUID
    longitude: float
    latitude: float
    shipment_id: UUID | None = None
    speed_mps: float | None = None
    heading_deg: float | None = None
    altitude_m: float | None = None
    temperature_c: float | None = None
    humidity_pct: float | None = None
    vibration_g: float | None = None
    battery_pct: float | None = None
    door_open: bool | None = None
    shock_event: bool = False
    sensor_quality: int = 100
    source_stream: str = "iiot"
    state_vector: JsonMapping = field(default_factory=dict)

    def to_copy_row(self) -> tuple[Any, ...]:
        """Convert to a COPY-compatible row matching TelemetryIngestService.COPY_COLUMNS."""
        return (
            _as_utc(self.recorded_at),
            self.container_id,
            self.device_id,
            self.shipment_id,
            self.longitude,
            self.latitude,
            self.speed_mps,
            self.heading_deg,
            self.altitude_m,
            self.temperature_c,
            self.humidity_pct,
            self.vibration_g,
            self.battery_pct,
            self.door_open,
            self.shock_event,
            self.sensor_quality,
            self.source_stream,
            json.dumps(self.state_vector, separators=(",", ":"), default=str),
        )


class TelemetryIngestService:
    """Bulk writer for the iiot_container_telemetry TimescaleDB hypertable."""

    COPY_COLUMNS: tuple[str, ...] = (
        "recorded_at",
        "container_id",
        "device_id",
        "shipment_id",
        "longitude",
        "latitude",
        "speed_mps",
        "heading_deg",
        "altitude_m",
        "temperature_c",
        "humidity_pct",
        "vibration_g",
        "battery_pct",
        "door_open",
        "shock_event",
        "sensor_quality",
        "source_stream",
        "state_vector",
    )

    INSERT_SQL = """
        INSERT INTO iiot_container_telemetry (
            recorded_at,
            container_id,
            device_id,
            shipment_id,
            position,
            position_geog,
            speed_mps,
            heading_deg,
            altitude_m,
            temperature_c,
            humidity_pct,
            vibration_g,
            battery_pct,
            door_open,
            shock_event,
            sensor_quality,
            source_stream,
            state_vector
        )
        SELECT
            recorded_at,
            container_id,
            device_id,
            shipment_id,
            ST_SetSRID(ST_MakePoint(longitude, latitude), 4326),
            ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography,
            speed_mps,
            heading_deg,
            altitude_m,
            temperature_c,
            humidity_pct,
            vibration_g,
            battery_pct,
            door_open,
            shock_event,
            sensor_quality,
            source_stream,
            state_vector::jsonb
        FROM _telemetry_ingest_stage;
    """

    CREATE_TEMP_TABLE_SQL = """
        CREATE TEMP TABLE IF NOT EXISTS _telemetry_ingest_stage (
            recorded_at TIMESTAMPTZ(6) NOT NULL,
            container_id UUID NOT NULL,
            device_id UUID NOT NULL,
            shipment_id UUID,
            longitude FLOAT8 NOT NULL,
            latitude FLOAT8 NOT NULL,
            speed_mps REAL,
            heading_deg REAL,
            altitude_m REAL,
            temperature_c REAL,
            humidity_pct REAL,
            vibration_g REAL,
            battery_pct REAL,
            door_open BOOLEAN,
            shock_event BOOLEAN NOT NULL,
            sensor_quality SMALLINT NOT NULL,
            source_stream TEXT NOT NULL,
            state_vector JSONB NOT NULL
        ) ON COMMIT DROP;
    """

    TRUNCATE_TEMP_TABLE_SQL = "TRUNCATE TABLE _telemetry_ingest_stage;"

    def __init__(self, pool: asyncpg.Pool, *, batch_size: int = 10_000) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self._pool = pool
        self._batch_size = batch_size

    async def ingest_many(self, records: Iterable[TelemetryRecord]) -> int:
        """Ingest records in bounded batches using asyncpg binary COPY."""
        inserted = 0
        batch: list[TelemetryRecord] = []

        for record in records:
            self._validate_record(record)
            batch.append(record)
            if len(batch) >= self._batch_size:
                inserted += await self._ingest_batch(batch)
                batch.clear()

        if batch:
            inserted += await self._ingest_batch(batch)

        return inserted

    async def ingest_rows(self, rows: Iterable[JsonMapping]) -> int:
        """Accept dict-like rows and normalize them into typed telemetry records."""
        return await self.ingest_many(self._record_from_mapping(row) for row in rows)

    async def _ingest_batch(self, records: Sequence[TelemetryRecord]) -> int:
        if not records:
            return 0

        copy_rows = [record.to_copy_row() for record in records]

        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(self.CREATE_TEMP_TABLE_SQL)
                await connection.execute(self.TRUNCATE_TEMP_TABLE_SQL)
                await connection.copy_records_to_table(
                    "_telemetry_ingest_stage",
                    records=copy_rows,
                    columns=self.COPY_COLUMNS,
                )
                await connection.execute(self.INSERT_SQL)

        return len(records)

    @staticmethod
    def _record_from_mapping(row: JsonMapping) -> TelemetryRecord:
        """Normalize common upstream payload shapes into the canonical ingest record."""
        longitude = row.get("longitude", row.get("lon"))
        latitude = row.get("latitude", row.get("lat"))

        if longitude is None or latitude is None:
            raise ValueError("Telemetry row must include longitude/latitude or lon/lat.")

        return TelemetryRecord(
            recorded_at=_parse_datetime(row["recorded_at"]),
            container_id=_parse_uuid(row["container_id"]),
            device_id=_parse_uuid(row["device_id"]),
            shipment_id=_parse_optional_uuid(row.get("shipment_id")),
            longitude=float(longitude),
            latitude=float(latitude),
            speed_mps=_optional_float(row.get("speed_mps")),
            heading_deg=_optional_float(row.get("heading_deg")),
            altitude_m=_optional_float(row.get("altitude_m")),
            temperature_c=_optional_float(row.get("temperature_c")),
            humidity_pct=_optional_float(row.get("humidity_pct")),
            vibration_g=_optional_float(row.get("vibration_g")),
            battery_pct=_optional_float(row.get("battery_pct")),
            door_open=_optional_bool(row.get("door_open")),
            shock_event=bool(row.get("shock_event", False)),
            sensor_quality=int(row.get("sensor_quality", 100)),
            source_stream=str(row.get("source_stream", "iiot")),
            state_vector=_mapping_or_empty(row.get("state_vector")),
        )

    @staticmethod
    def _validate_record(record: TelemetryRecord) -> None:
        """Validate hot-path constraints before COPY to avoid aborting entire batches."""
        if not -180.0 <= record.longitude <= 180.0:
            raise ValueError(f"longitude out of range: {record.longitude}")
        if not -90.0 <= record.latitude <= 90.0:
            raise ValueError(f"latitude out of range: {record.latitude}")
        if record.heading_deg is not None and not 0.0 <= record.heading_deg < 360.0:
            raise ValueError(f"heading_deg out of range: {record.heading_deg}")
        if record.humidity_pct is not None and not 0.0 <= record.humidity_pct <= 100.0:
            raise ValueError(f"humidity_pct out of range: {record.humidity_pct}")
        if record.battery_pct is not None and not 0.0 <= record.battery_pct <= 100.0:
            raise ValueError(f"battery_pct out of range: {record.battery_pct}")
        if not 0 <= record.sensor_quality <= 100:
            raise ValueError(f"sensor_quality out of range: {record.sensor_quality}")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, str):
        return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    raise TypeError(f"Unsupported datetime value: {value!r}")

def _parse_uuid(value: Any) -> UUID:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        return UUID(value)
    raise TypeError(f"Unsupported UUID value: {value!r}")

def _parse_optional_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    return _parse_uuid(value)

def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)

def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)

def _mapping_or_empty(value: Any) -> JsonMapping:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"state_vector must be a mapping, got {type(value).__name__}.")
