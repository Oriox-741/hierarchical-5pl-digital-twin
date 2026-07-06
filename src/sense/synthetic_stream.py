"""Synthetic continuous IIoT stream generation for early Sense-layer development."""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4

import asyncpg

from src.sense.db_pool import DatabasePool
from src.sense.telemetry_ingest import TelemetryIngestService, TelemetryRecord


@dataclass(frozen=True, slots=True)
class SyntheticAsset:
    """A generic mobile or fixed logistics entity with telemetry behavior."""

    container_id: UUID
    device_id: UUID
    asset_type: str
    base_latitude: float
    base_longitude: float
    shipment_id: UUID | None = None
    nominal_speed_mps: float = 12.0
    battery_pct: float = 100.0
    route_radius_deg: float = 0.03


@dataclass(frozen=True, slots=True)
class SyntheticStreamConfig:
    """Runtime controls for synthetic telemetry emission."""

    asset_count: int = 8
    batch_size: int = 250
    interval_seconds: float = 1.0
    source_stream: str = "synthetic_phase3_stream"
    center_latitude: float = 41.0082
    center_longitude: float = 28.9784
    random_seed: int | None = None
    asset_types: Sequence[str] = field(
        default_factory=lambda: ("vehicle", "drone", "hub", "pallet")
    )


class SyntheticTelemetryStream:
    """Generate realistic telemetry batches and optionally ingest them continuously."""

    def __init__(
        self,
        ingest_service: TelemetryIngestService,
        config: SyntheticStreamConfig | None = None,
    ) -> None:
        self._ingest_service = ingest_service
        self._config = config or SyntheticStreamConfig()
        self._rng = random.Random(self._config.random_seed)
        self._assets = self._build_assets()
        self._step = 0

    @classmethod
    def from_pool(
        cls,
        pool: asyncpg.Pool,
        config: SyntheticStreamConfig | None = None,
    ) -> SyntheticTelemetryStream:
        """Create a stream using an existing asyncpg pool."""
        return cls(TelemetryIngestService(pool), config)

    @classmethod
    def from_database_pool(
        cls,
        db: DatabasePool,
        config: SyntheticStreamConfig | None = None,
    ) -> SyntheticTelemetryStream:
        """Create a stream using the project's DatabasePool lifecycle wrapper."""
        return cls.from_pool(db.pool, config)

    async def records(self) -> AsyncIterator[TelemetryRecord]:
        """Yield individual synthetic records forever."""
        while True:
            for record in self.generate_batch():
                yield record
            await asyncio.sleep(self._config.interval_seconds)

    def generate_batch(self, count: int | None = None) -> list[TelemetryRecord]:
        """Generate one bounded telemetry batch without touching the database."""
        target_count = count or self._config.batch_size
        now = datetime.now(timezone.utc)
        records: list[TelemetryRecord] = []

        for _ in range(target_count):
            asset = self._assets[self._step % len(self._assets)]
            phase = (self._step / max(len(self._assets), 1)) * 0.15
            latitude, longitude = self._position_for(asset, phase)
            speed = self._speed_for(asset)
            battery_pct = max(0.0, asset.battery_pct - (self._step * 0.004))
            heading_deg = (math.degrees(math.atan2(latitude - asset.base_latitude, longitude - asset.base_longitude)) + 360.0) % 360.0

            records.append(
                TelemetryRecord(
                    recorded_at=now,
                    container_id=asset.container_id,
                    device_id=asset.device_id,
                    shipment_id=asset.shipment_id,
                    latitude=latitude,
                    longitude=longitude,
                    speed_mps=speed,
                    heading_deg=heading_deg,
                    altitude_m=self._altitude_for(asset.asset_type),
                    temperature_c=self._rng.uniform(2.0, 31.0),
                    humidity_pct=self._rng.uniform(35.0, 88.0),
                    vibration_g=self._rng.uniform(0.02, 2.4 if asset.asset_type != "hub" else 0.3),
                    battery_pct=battery_pct,
                    door_open=self._rng.random() < 0.04,
                    shock_event=self._rng.random() < 0.015,
                    sensor_quality=self._rng.randint(90, 100),
                    source_stream=self._config.source_stream,
                    state_vector={
                        "asset_type": asset.asset_type,
                        "synthetic_step": self._step,
                        "nominal_speed_mps": asset.nominal_speed_mps,
                        "route_radius_deg": asset.route_radius_deg,
                    },
                )
            )
            self._step += 1

        return records

    async def run_forever(self) -> None:
        """Continuously generate and ingest telemetry until cancelled."""
        while True:
            inserted = await self.ingest_once()
            if inserted <= 0:
                raise RuntimeError("Synthetic stream generated no records.")
            await asyncio.sleep(self._config.interval_seconds)

    async def run_for_batches(self, batch_count: int) -> int:
        """Generate and ingest a fixed number of batches."""
        if batch_count < 0:
            raise ValueError("batch_count must not be negative.")

        inserted = 0
        for _ in range(batch_count):
            inserted += await self.ingest_once()
            await asyncio.sleep(self._config.interval_seconds)
        return inserted

    async def ingest_once(self, count: int | None = None) -> int:
        """Generate one batch and ingest it through TelemetryIngestService."""
        return await self._ingest_service.ingest_many(self.generate_batch(count))

    def _build_assets(self) -> list[SyntheticAsset]:
        if self._config.asset_count <= 0:
            raise ValueError("asset_count must be greater than zero.")
        if not self._config.asset_types:
            raise ValueError("asset_types must not be empty.")

        assets: list[SyntheticAsset] = []
        for index in range(self._config.asset_count):
            asset_type = self._config.asset_types[index % len(self._config.asset_types)]
            assets.append(
                SyntheticAsset(
                    container_id=uuid4(),
                    device_id=uuid4(),
                    shipment_id=uuid4() if asset_type != "hub" else None,
                    asset_type=asset_type,
                    base_latitude=self._config.center_latitude + self._rng.uniform(-0.12, 0.12),
                    base_longitude=self._config.center_longitude + self._rng.uniform(-0.12, 0.12),
                    nominal_speed_mps=self._nominal_speed_for(asset_type),
                    battery_pct=self._rng.uniform(65.0, 100.0),
                    route_radius_deg=self._rng.uniform(0.005, 0.045),
                )
            )
        return assets

    def _position_for(self, asset: SyntheticAsset, phase: float) -> tuple[float, float]:
        if asset.asset_type == "hub":
            jitter = asset.route_radius_deg * 0.03
            return (
                asset.base_latitude + self._rng.uniform(-jitter, jitter),
                asset.base_longitude + self._rng.uniform(-jitter, jitter),
            )

        return (
            asset.base_latitude + math.sin(phase) * asset.route_radius_deg,
            asset.base_longitude + math.cos(phase) * asset.route_radius_deg,
        )

    def _speed_for(self, asset: SyntheticAsset) -> float:
        if asset.asset_type == "hub":
            return 0.0
        return max(0.0, self._rng.gauss(asset.nominal_speed_mps, asset.nominal_speed_mps * 0.12))

    def _nominal_speed_for(self, asset_type: str) -> float:
        match asset_type:
            case "drone":
                return self._rng.uniform(14.0, 24.0)
            case "vehicle":
                return self._rng.uniform(8.0, 22.0)
            case "pallet":
                return self._rng.uniform(0.0, 2.5)
            case "hub":
                return 0.0
            case _:
                return self._rng.uniform(1.0, 12.0)

    def _altitude_for(self, asset_type: str) -> float:
        if asset_type == "drone":
            return self._rng.uniform(45.0, 160.0)
        return self._rng.uniform(0.0, 35.0)
