"""Domain entities for the SimPy-powered 5PL digital twin."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from src.shared.metrics import clamp
from src.shared.types import AssetKind, GeoPoint, JsonDict, ShipmentStatus


class VehicleTier(StrEnum):
    """Two-echelon fleet tiers used by LRP-MPPD-2E routing constraints."""

    PRIMARY = "primary"
    SECONDARY = "secondary"


@dataclass(slots=True)
class Product:
    """A movable SKU or product family in the multi-echelon network."""

    product_id: UUID
    sku: str
    unit_weight_kg: float = 1.0
    unit_volume_m3: float = 0.01
    storage_cost_per_unit_time: float = 0.0
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.unit_weight_kg < 0.0:
            raise ValueError("unit_weight_kg must not be negative.")
        if self.unit_volume_m3 < 0.0:
            raise ValueError("unit_volume_m3 must not be negative.")


@dataclass(slots=True)
class Hub:
    """Candidate hub, depot, warehouse, or processing center."""

    hub_id: UUID
    name: str
    location: GeoPoint
    echelon_level: int
    storage_capacity_units: float
    throughput_units_per_time: float
    activation_cost: float = 0.0
    active: bool = True
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.echelon_level < 0:
            raise ValueError("echelon_level must not be negative.")
        if self.storage_capacity_units < 0.0:
            raise ValueError("storage_capacity_units must not be negative.")
        if self.throughput_units_per_time <= 0.0:
            raise ValueError("throughput_units_per_time must be positive.")


@dataclass(slots=True)
class Customer:
    """Demand destination with a service time window."""

    customer_id: UUID
    name: str
    location: GeoPoint
    demand_units: dict[UUID, float]
    time_window_start: float = 0.0
    time_window_end: float = float("inf")
    service_time: float = 0.0
    visited: bool = False
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time_window_start > self.time_window_end:
            raise ValueError("time_window_start must be <= time_window_end.")
        if self.service_time < 0.0:
            raise ValueError("service_time must not be negative.")
        if any(quantity < 0.0 for quantity in self.demand_units.values()):
            raise ValueError("customer demand quantities must not be negative.")

    @property
    def total_demand_units(self) -> float:
        return sum(self.demand_units.values())

    def can_be_served_at(self, simulation_time: float) -> bool:
        return self.time_window_start <= simulation_time <= self.time_window_end


@dataclass(slots=True)
class Vehicle:
    """Primary or secondary vehicle with capacity and route state."""

    vehicle_id: UUID
    asset_kind: AssetKind
    tier: VehicleTier
    capacity_units: float
    capacity_weight_kg: float
    capacity_volume_m3: float
    nominal_speed_mps: float
    fixed_usage_cost: float = 0.0
    transport_cost_per_meter: float = 0.0
    current_location: GeoPoint | None = None
    current_node_id: UUID | None = None
    active: bool = True
    load_units: float = 0.0
    load_weight_kg: float = 0.0
    load_volume_m3: float = 0.0
    route: list[UUID] = field(default_factory=list)
    baseline_speed_mps: float | None = None
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.capacity_units <= 0.0:
            raise ValueError("capacity_units must be positive.")
        if self.capacity_weight_kg <= 0.0:
            raise ValueError("capacity_weight_kg must be positive.")
        if self.capacity_volume_m3 <= 0.0:
            raise ValueError("capacity_volume_m3 must be positive.")
        if self.nominal_speed_mps <= 0.0:
            raise ValueError("nominal_speed_mps must be positive.")
        if self.baseline_speed_mps is None:
            self.baseline_speed_mps = self.nominal_speed_mps
        if self.baseline_speed_mps <= 0.0:
            raise ValueError("baseline_speed_mps must be positive.")
        self.metadata.setdefault("baseline_speed_mps", self.baseline_speed_mps)

    @property
    def remaining_units(self) -> float:
        return max(0.0, self.capacity_units - self.load_units)

    @property
    def utilization(self) -> float:
        return clamp(self.load_units / self.capacity_units, 0.0, 1.0)

    def can_load(self, *, units: float, weight_kg: float, volume_m3: float) -> bool:
        return (
            units >= 0.0
            and weight_kg >= 0.0
            and volume_m3 >= 0.0
            and self.load_units + units <= self.capacity_units
            and self.load_weight_kg + weight_kg <= self.capacity_weight_kg
            and self.load_volume_m3 + volume_m3 <= self.capacity_volume_m3
        )

    def load(self, *, units: float, weight_kg: float, volume_m3: float) -> None:
        if not self.can_load(units=units, weight_kg=weight_kg, volume_m3=volume_m3):
            raise ValueError("vehicle capacity constraint violated.")
        self.load_units += units
        self.load_weight_kg += weight_kg
        self.load_volume_m3 += volume_m3

    def unload(self, *, units: float, weight_kg: float, volume_m3: float) -> None:
        self.load_units = max(0.0, self.load_units - units)
        self.load_weight_kg = max(0.0, self.load_weight_kg - weight_kg)
        self.load_volume_m3 = max(0.0, self.load_volume_m3 - volume_m3)


@dataclass(slots=True)
class OrderLine:
    """Product quantity requested or transported by an order."""

    product_id: UUID
    quantity_units: float

    def __post_init__(self) -> None:
        if self.quantity_units <= 0.0:
            raise ValueError("quantity_units must be positive.")


@dataclass(slots=True)
class Order:
    """Pickup-delivery order with time-window and service-state tracking."""

    order_id: UUID
    customer_id: UUID
    origin_node_id: UUID
    destination_node_id: UUID
    lines: list[OrderLine]
    release_time: float = 0.0
    due_time: float = float("inf")
    pickup_time: float | None = None
    delivery_time: float | None = None
    assigned_vehicle_id: UUID | None = None
    status: ShipmentStatus = ShipmentStatus.CREATED
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.lines:
            raise ValueError("order must contain at least one line.")
        if self.release_time > self.due_time:
            raise ValueError("release_time must be <= due_time.")

    @property
    def total_units(self) -> float:
        return sum(line.quantity_units for line in self.lines)

    def mark_in_transit(self, vehicle_id: UUID, time_now: float) -> None:
        self.assigned_vehicle_id = vehicle_id
        self.pickup_time = time_now
        self.status = ShipmentStatus.IN_TRANSIT

    def mark_delivered(self, time_now: float) -> None:
        self.delivery_time = time_now
        self.status = ShipmentStatus.DELIVERED

    @property
    def is_late(self) -> bool:
        return self.delivery_time is not None and self.delivery_time > self.due_time


@dataclass(frozen=True, slots=True)
class RouteArc:
    """Directed logistics arc connecting two network nodes."""

    origin_node_id: UUID
    destination_node_id: UUID
    distance_m: float
    nominal_speed_mps: float
    capacity_units: float = float("inf")
    traversal_cost_per_unit: float = 0.0
    active: bool = True
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.distance_m < 0.0:
            raise ValueError("distance_m must not be negative.")
        if self.nominal_speed_mps <= 0.0:
            raise ValueError("nominal_speed_mps must be positive.")
        if self.capacity_units <= 0.0:
            raise ValueError("capacity_units must be positive.")


def new_entity_id() -> UUID:
    """Generate an entity UUID without coupling callers to uuid internals."""
    return uuid4()
