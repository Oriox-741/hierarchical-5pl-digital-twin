"""Shared domain types for the autonomous 5PL framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


JsonDict = dict[str, Any]


class AssetKind(StrEnum):
    VEHICLE = "vehicle"
    DRONE = "drone"
    HUB = "hub"
    PALLET = "pallet"
    CONTAINER = "container"
    WAREHOUSE = "warehouse"


class ShipmentStatus(StrEnum):
    CREATED = "created"
    IN_TRANSIT = "in_transit"
    DWELLING = "dwelling"
    DELAYED = "delayed"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class EventKind(StrEnum):
    TELEMETRY = "telemetry"
    DELAY = "delay"
    DISRUPTION = "disruption"
    CAPACITY_BREACH = "capacity_breach"
    STOCKOUT_RISK = "stockout_risk"
    SAFETY_INTERVENTION = "safety_intervention"


@dataclass(frozen=True, slots=True)
class GeoPoint:
    latitude: float
    longitude: float
    altitude_m: float | None = None


@dataclass(frozen=True, slots=True)
class ObservationEnvelope:
    observed_at: datetime
    asset_id: UUID | None = None
    shipment_id: UUID | None = None
    state_vector: JsonDict = field(default_factory=dict)
    safety_potential: float = 0.0


@dataclass(frozen=True, slots=True)
class TraceEnvelope:
    episode_id: UUID
    step_id: int
    recorded_at: datetime
    observation: JsonDict
    action: JsonDict
    reward: float
    next_observation: JsonDict | None = None
    projected_action: JsonDict | None = None
    terminated: bool = False
    truncated: bool = False
    info: JsonDict = field(default_factory=dict)
