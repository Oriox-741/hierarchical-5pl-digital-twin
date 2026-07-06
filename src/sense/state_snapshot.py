"""Fast latest-state snapshot retrieval for Act and Think layer consumers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg

from src.sense.db_pool import DatabasePool


@dataclass(frozen=True, slots=True)
class AssetTelemetrySnapshot:
    """Latest physical telemetry for a tracked asset/device pair."""

    telemetry_id: int
    recorded_at: datetime
    ingested_at: datetime
    container_id: UUID
    device_id: UUID
    shipment_id: UUID | None
    longitude: float
    latitude: float
    speed_mps: float | None
    heading_deg: float | None
    altitude_m: float | None
    temperature_c: float | None
    humidity_pct: float | None
    vibration_g: float | None
    battery_pct: float | None
    door_open: bool | None
    shock_event: bool
    sensor_quality: int
    source_stream: str
    state_vector: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ShipmentStateSnapshot:
    """Latest operational state for a shipment in the 5PL network."""

    state_id: int
    recorded_at: datetime
    updated_at: datetime
    shipment_id: UUID
    container_id: UUID | None
    current_node_id: UUID | None
    next_node_id: UUID | None
    shipment_status: str
    longitude: float | None
    latitude: float | None
    eta_at: datetime | None
    dwell_seconds: int
    delay_seconds: int
    inventory_units: int | None
    capacity_utilization: float | None
    congestion_score: float
    disruption_score: float
    safety_potential: float
    state_vector: dict[str, Any]
    metadata: dict[str, Any]


class StateSnapshotService:
    """Read-optimized snapshot service backed by asyncpg and Sense indexes."""

    LATEST_ASSET_SQL = """
        SELECT
            telemetry_id,
            recorded_at,
            ingested_at,
            container_id,
            device_id,
            shipment_id,
            ST_X(position::geometry) AS longitude,
            ST_Y(position::geometry) AS latitude,
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
        FROM iiot_container_telemetry
        WHERE container_id = $1
        ORDER BY recorded_at DESC
        LIMIT 1;
    """

    LATEST_ASSETS_SQL = """
        SELECT DISTINCT ON (container_id)
            telemetry_id,
            recorded_at,
            ingested_at,
            container_id,
            device_id,
            shipment_id,
            ST_X(position::geometry) AS longitude,
            ST_Y(position::geometry) AS latitude,
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
        FROM iiot_container_telemetry
        WHERE container_id = ANY($1::uuid[])
        ORDER BY container_id, recorded_at DESC;
    """

    RECENT_ASSETS_SQL = """
        SELECT DISTINCT ON (container_id)
            telemetry_id,
            recorded_at,
            ingested_at,
            container_id,
            device_id,
            shipment_id,
            ST_X(position::geometry) AS longitude,
            ST_Y(position::geometry) AS latitude,
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
        FROM iiot_container_telemetry
        WHERE recorded_at >= now() - ($1::FLOAT8 * INTERVAL '1 second')
        ORDER BY container_id, recorded_at DESC
        LIMIT $2;
    """

    LATEST_SHIPMENT_SQL = """
        SELECT
            state_id,
            recorded_at,
            updated_at,
            shipment_id,
            container_id,
            current_node_id,
            next_node_id,
            shipment_status,
            CASE WHEN current_position IS NULL THEN NULL ELSE ST_X(current_position::geometry) END AS longitude,
            CASE WHEN current_position IS NULL THEN NULL ELSE ST_Y(current_position::geometry) END AS latitude,
            eta_at,
            dwell_seconds,
            delay_seconds,
            inventory_units,
            capacity_utilization,
            congestion_score,
            disruption_score,
            safety_potential,
            state_vector,
            metadata
        FROM ops_shipment_states
        WHERE shipment_id = $1
        ORDER BY recorded_at DESC
        LIMIT 1;
    """

    LATEST_SHIPMENTS_SQL = """
        SELECT DISTINCT ON (shipment_id)
            state_id,
            recorded_at,
            updated_at,
            shipment_id,
            container_id,
            current_node_id,
            next_node_id,
            shipment_status,
            CASE WHEN current_position IS NULL THEN NULL ELSE ST_X(current_position::geometry) END AS longitude,
            CASE WHEN current_position IS NULL THEN NULL ELSE ST_Y(current_position::geometry) END AS latitude,
            eta_at,
            dwell_seconds,
            delay_seconds,
            inventory_units,
            capacity_utilization,
            congestion_score,
            disruption_score,
            safety_potential,
            state_vector,
            metadata
        FROM ops_shipment_states
        WHERE shipment_id = ANY($1::uuid[])
        ORDER BY shipment_id, recorded_at DESC;
    """

    def __init__(self, db: DatabasePool | asyncpg.Pool) -> None:
        self._pool = db.pool if isinstance(db, DatabasePool) else db

    async def latest_asset(self, container_id: UUID) -> AssetTelemetrySnapshot | None:
        """Fetch the newest telemetry row for a generic tracked physical entity."""
        row = await self._pool.fetchrow(self.LATEST_ASSET_SQL, container_id)
        return self._asset_snapshot(row) if row is not None else None

    async def latest_assets(self, container_ids: Sequence[UUID]) -> list[AssetTelemetrySnapshot]:
        """Fetch newest telemetry rows for many tracked entities in one indexed query."""
        if not container_ids:
            return []
        rows = await self._pool.fetch(self.LATEST_ASSETS_SQL, list(container_ids))
        return [self._asset_snapshot(row) for row in rows]

    async def recent_network_assets(
        self,
        *,
        within_seconds: float = 300.0,
        limit: int = 5_000,
    ) -> list[AssetTelemetrySnapshot]:
        """Fetch latest rows for assets active inside a recent time window."""
        rows = await self._pool.fetch(self.RECENT_ASSETS_SQL, within_seconds, limit)
        return [self._asset_snapshot(row) for row in rows]

    async def latest_shipment(self, shipment_id: UUID) -> ShipmentStateSnapshot | None:
        """Fetch the newest operational state for one shipment."""
        row = await self._pool.fetchrow(self.LATEST_SHIPMENT_SQL, shipment_id)
        return self._shipment_snapshot(row) if row is not None else None

    async def latest_shipments(self, shipment_ids: Sequence[UUID]) -> list[ShipmentStateSnapshot]:
        """Fetch newest operational states for many shipments in one indexed query."""
        if not shipment_ids:
            return []
        rows = await self._pool.fetch(self.LATEST_SHIPMENTS_SQL, list(shipment_ids))
        return [self._shipment_snapshot(row) for row in rows]

    @staticmethod
    def _asset_snapshot(row: asyncpg.Record) -> AssetTelemetrySnapshot:
        return AssetTelemetrySnapshot(
            telemetry_id=row["telemetry_id"],
            recorded_at=row["recorded_at"],
            ingested_at=row["ingested_at"],
            container_id=row["container_id"],
            device_id=row["device_id"],
            shipment_id=row["shipment_id"],
            longitude=float(row["longitude"]),
            latitude=float(row["latitude"]),
            speed_mps=_optional_float(row["speed_mps"]),
            heading_deg=_optional_float(row["heading_deg"]),
            altitude_m=_optional_float(row["altitude_m"]),
            temperature_c=_optional_float(row["temperature_c"]),
            humidity_pct=_optional_float(row["humidity_pct"]),
            vibration_g=_optional_float(row["vibration_g"]),
            battery_pct=_optional_float(row["battery_pct"]),
            door_open=row["door_open"],
            shock_event=row["shock_event"],
            sensor_quality=row["sensor_quality"],
            source_stream=row["source_stream"],
            state_vector=_json_dict(row["state_vector"]),
        )

    @staticmethod
    def _shipment_snapshot(row: asyncpg.Record) -> ShipmentStateSnapshot:
        return ShipmentStateSnapshot(
            state_id=row["state_id"],
            recorded_at=row["recorded_at"],
            updated_at=row["updated_at"],
            shipment_id=row["shipment_id"],
            container_id=row["container_id"],
            current_node_id=row["current_node_id"],
            next_node_id=row["next_node_id"],
            shipment_status=row["shipment_status"],
            longitude=_optional_float(row["longitude"]),
            latitude=_optional_float(row["latitude"]),
            eta_at=row["eta_at"],
            dwell_seconds=row["dwell_seconds"],
            delay_seconds=row["delay_seconds"],
            inventory_units=row["inventory_units"],
            capacity_utilization=_optional_float(row["capacity_utilization"]),
            congestion_score=float(row["congestion_score"]),
            disruption_score=float(row["disruption_score"]),
            safety_potential=float(row["safety_potential"]),
            state_vector=_json_dict(row["state_vector"]),
            metadata=_json_dict(row["metadata"]),
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _json_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
        raise TypeError("Expected JSON object.")
    if isinstance(value, dict):
        return dict(value)
    return dict(value)
