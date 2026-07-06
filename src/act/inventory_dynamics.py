"""Multi-echelon inventory dynamics and MEIO simulation rules."""

from __future__ import annotations

import math
from collections.abc import Generator
from dataclasses import dataclass, field
from uuid import UUID

import simpy

from src.shared.metrics import backlog_ratio, clamp, safety_potential
from src.shared.types import JsonDict


SERVICE_LEVEL_Z: dict[float, float] = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.99: 2.3263,
}


@dataclass(slots=True)
class InventoryPosition:
    """MEIO state for one product at one echelon/node."""

    node_id: UUID
    product_id: UUID
    echelon_level: int
    on_hand_units: float = 0.0
    reserved_units: float = 0.0
    in_transit_units: float = 0.0
    backlog_units: float = 0.0
    safety_stock_units: float = 0.0
    demand_mean: float = 0.0
    demand_variance: float = 0.0
    lead_time_mean: float = 0.0
    lead_time_variance: float = 0.0
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.echelon_level < 0:
            raise ValueError("echelon_level must not be negative.")
        self._assert_nonnegative()

    @property
    def available_units(self) -> float:
        return max(0.0, self.on_hand_units - self.reserved_units)

    @property
    def inventory_pressure(self) -> float:
        denominator = max(self.safety_stock_units + self.demand_mean, 1.0)
        return clamp(1.0 - (self.available_units / denominator), 0.0, 1.0)

    @property
    def backlog_pressure(self) -> float:
        return backlog_ratio(self.backlog_units, max(self.demand_mean, self.backlog_units, 1.0))

    def target_safety_stock(self, service_level: float = 0.95) -> float:
        """MEIO safety stock: z * sqrt(L * demand_var + demand_mean^2 * lead_time_var)."""
        z = SERVICE_LEVEL_Z.get(service_level, SERVICE_LEVEL_Z[0.95])
        variance_term = max(0.0, self.lead_time_mean * self.demand_variance)
        lead_time_term = max(0.0, (self.demand_mean**2) * self.lead_time_variance)
        return z * math.sqrt(variance_term + lead_time_term)

    def reorder_quantity(self, target_service_level: float = 0.95) -> float:
        target_safety = self.target_safety_stock(target_service_level)
        target_position = target_safety + self.demand_mean * max(1.0, self.lead_time_mean)
        current_position = self.on_hand_units + self.in_transit_units - self.backlog_units
        return max(0.0, target_position - current_position)

    def reserve(self, units: float) -> float:
        units = _positive(units, "units")
        reserved = min(units, self.available_units)
        self.reserved_units += reserved
        shortfall = units - reserved
        self.backlog_units += shortfall
        self._assert_nonnegative()
        return reserved

    def ship_reserved(self, units: float) -> float:
        units = _positive(units, "units")
        shipped = min(units, self.reserved_units, self.on_hand_units)
        self.reserved_units -= shipped
        self.on_hand_units -= shipped
        self.in_transit_units += shipped
        self._assert_nonnegative()
        return shipped

    def receive(self, units: float) -> None:
        units = _positive(units, "units")
        self.in_transit_units = max(0.0, self.in_transit_units - units)
        if self.backlog_units > 0.0:
            filled = min(units, self.backlog_units)
            self.backlog_units -= filled
            units -= filled
        self.on_hand_units += units
        self._assert_nonnegative()

    def consume(self, units: float) -> float:
        units = _positive(units, "units")
        consumed = min(units, self.available_units)
        self.on_hand_units -= consumed
        self.backlog_units += units - consumed
        self._assert_nonnegative()
        return consumed

    def update_forecast(self, observed_demand: float, alpha: float = 0.20) -> None:
        observed_demand = _positive(observed_demand, "observed_demand")
        alpha = clamp(alpha, 0.0, 1.0)
        previous_mean = self.demand_mean
        self.demand_mean = (alpha * observed_demand) + ((1.0 - alpha) * self.demand_mean)
        innovation = observed_demand - previous_mean
        self.demand_variance = (alpha * (innovation**2)) + ((1.0 - alpha) * self.demand_variance)

    def _assert_nonnegative(self) -> None:
        values = (
            self.on_hand_units,
            self.reserved_units,
            self.in_transit_units,
            self.backlog_units,
            self.safety_stock_units,
            self.demand_mean,
            self.demand_variance,
            self.lead_time_mean,
            self.lead_time_variance,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("inventory values must not be negative.")


class InventoryNetwork:
    """Collection of MEIO inventory positions with SimPy replenishment processes."""

    def __init__(self, env: simpy.Environment, positions: list[InventoryPosition] | None = None) -> None:
        self.env = env
        self._positions: dict[tuple[UUID, UUID], InventoryPosition] = {}
        for position in positions or []:
            self.add_position(position)

    @property
    def positions(self) -> dict[tuple[UUID, UUID], InventoryPosition]:
        return self._positions

    def add_position(self, position: InventoryPosition) -> None:
        self._positions[(position.node_id, position.product_id)] = position

    def get_position(self, node_id: UUID, product_id: UUID) -> InventoryPosition:
        try:
            return self._positions[(node_id, product_id)]
        except KeyError as exc:
            raise KeyError(f"missing inventory position node={node_id} product={product_id}") from exc

    def transfer(
        self,
        *,
        source_node_id: UUID,
        destination_node_id: UUID,
        product_id: UUID,
        units: float,
        lead_time: float,
    ) -> simpy.Process:
        """Move stock between echelons with a discrete-event lead time."""
        return self.env.process(
            self._transfer_process(
                source_node_id=source_node_id,
                destination_node_id=destination_node_id,
                product_id=product_id,
                units=units,
                lead_time=lead_time,
            )
        )

    def network_safety_potential(self, congestion_score: float = 0.0, disruption_score: float = 0.0) -> float:
        if not self._positions:
            return safety_potential(
                congestion_score=congestion_score,
                disruption_score=disruption_score,
                capacity_pressure=0.0,
                backlog_pressure=0.0,
            )

        inventory_pressure = sum(position.inventory_pressure for position in self._positions.values()) / len(self._positions)
        backlog_pressure_avg = sum(position.backlog_pressure for position in self._positions.values()) / len(self._positions)
        return safety_potential(
            congestion_score=congestion_score,
            disruption_score=disruption_score,
            capacity_pressure=inventory_pressure,
            backlog_pressure=backlog_pressure_avg,
        )

    def snapshot(self) -> dict[str, JsonDict]:
        return {
            f"{node_id}:{product_id}": {
                "node_id": str(node_id),
                "product_id": str(product_id),
                "echelon_level": position.echelon_level,
                "on_hand_units": position.on_hand_units,
                "reserved_units": position.reserved_units,
                "in_transit_units": position.in_transit_units,
                "backlog_units": position.backlog_units,
                "safety_stock_units": position.safety_stock_units,
                "demand_mean": position.demand_mean,
                "demand_variance": position.demand_variance,
                "inventory_pressure": position.inventory_pressure,
                "backlog_pressure": position.backlog_pressure,
            }
            for (node_id, product_id), position in self._positions.items()
        }

    def _transfer_process(
        self,
        *,
        source_node_id: UUID,
        destination_node_id: UUID,
        product_id: UUID,
        units: float,
        lead_time: float,
    ) -> Generator[simpy.Event, None, None]:
        if lead_time < 0.0:
            raise ValueError("lead_time must not be negative.")

        source = self.get_position(source_node_id, product_id)
        destination = self.get_position(destination_node_id, product_id)
        shipped = source.reserve(units)
        source.ship_reserved(shipped)
        yield self.env.timeout(lead_time)
        destination.in_transit_units += shipped
        destination.receive(shipped)


def _positive(value: float, name: str) -> float:
    if value < 0.0:
        raise ValueError(f"{name} must not be negative.")
    return value
