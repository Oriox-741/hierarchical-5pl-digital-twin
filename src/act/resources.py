"""SimPy resource wrappers for warehouse, fleet, dock, and berth capacity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import simpy

from src.shared.metrics import capacity_utilization, clamp


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    resource_id: UUID
    capacity: float
    used: float
    utilization: float
    queue_length: int


class WarehouseCapacity:
    """Inventory-capacity container for a hub or warehouse node."""

    def __init__(self, env: simpy.Environment, node_id: UUID, capacity_units: float, initial_units: float = 0.0) -> None:
        if capacity_units <= 0.0:
            raise ValueError("capacity_units must be positive.")
        if not 0.0 <= initial_units <= capacity_units:
            raise ValueError("initial_units must be within [0, capacity_units].")

        self.node_id = node_id
        self._container = simpy.Container(env, capacity=capacity_units, init=initial_units)

    @property
    def capacity_units(self) -> float:
        return float(self._container.capacity)

    @property
    def level_units(self) -> float:
        return float(self._container.level)

    @property
    def available_units(self) -> float:
        return max(0.0, self.capacity_units - self.level_units)

    def put(self, units: float) -> simpy.Event:
        if units < 0.0:
            raise ValueError("units must not be negative.")
        return self._container.put(units)

    def get(self, units: float) -> simpy.Event:
        if units < 0.0:
            raise ValueError("units must not be negative.")
        return self._container.get(units)

    def snapshot(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            resource_id=self.node_id,
            capacity=self.capacity_units,
            used=self.level_units,
            utilization=capacity_utilization(self.level_units, self.capacity_units),
            queue_length=len(self._container.put_queue) + len(self._container.get_queue),
        )


class FleetCapacity:
    """SimPy resource governing vehicle availability for a fleet tier or depot."""

    def __init__(self, env: simpy.Environment, fleet_id: UUID, vehicle_count: int) -> None:
        if vehicle_count <= 0:
            raise ValueError("vehicle_count must be positive.")
        self.fleet_id = fleet_id
        self._capacity = vehicle_count
        self._resource = simpy.Resource(env, capacity=vehicle_count)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def in_use(self) -> int:
        return self._resource.count

    def request(self) -> Any:
        return self._resource.request()

    def release(self, request: Any) -> None:
        self._resource.release(request)

    def snapshot(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            resource_id=self.fleet_id,
            capacity=float(self._capacity),
            used=float(self.in_use),
            utilization=capacity_utilization(float(self.in_use), float(self._capacity)),
            queue_length=len(self._resource.queue),
        )


class DockBerthCapacity:
    """Priority resource for docks, berths, cross-dock doors, and loading bays."""

    def __init__(self, env: simpy.Environment, node_id: UUID, berth_count: int) -> None:
        if berth_count <= 0:
            raise ValueError("berth_count must be positive.")
        self.node_id = node_id
        self._capacity = berth_count
        self._resource = simpy.PriorityResource(env, capacity=berth_count)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def in_use(self) -> int:
        return self._resource.count

    def request(self, priority: int = 0) -> Any:
        return self._resource.request(priority=priority)

    def release(self, request: Any) -> None:
        self._resource.release(request)

    def snapshot(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            resource_id=self.node_id,
            capacity=float(self._capacity),
            used=float(self.in_use),
            utilization=capacity_utilization(float(self.in_use), float(self._capacity)),
            queue_length=len(self._resource.queue),
        )


class CapacityState:
    """Mutable capacity multiplier used by disruptions and recovery processes."""

    def __init__(self, base_capacity: float) -> None:
        if base_capacity <= 0.0:
            raise ValueError("base_capacity must be positive.")
        self.base_capacity = base_capacity
        self.multiplier = 1.0

    @property
    def effective_capacity(self) -> float:
        return self.base_capacity * self.multiplier

    @property
    def capacity_pressure(self) -> float:
        return clamp(1.0 - self.multiplier, 0.0, 1.0)

    def reduce(self, fraction_lost: float) -> None:
        self.multiplier = clamp(self.multiplier * (1.0 - fraction_lost), 0.0, 1.0)

    def restore(self, fraction_recovered: float) -> None:
        self.multiplier = clamp(self.multiplier + fraction_recovered, 0.0, 1.0)
