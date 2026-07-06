"""Capacity, time-window, flow, and inventory feasibility checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from uuid import UUID

from src.act.entities import Customer, Order, RouteArc, Vehicle
from src.act.inventory_dynamics import InventoryNetwork
from src.act.routing_physics import RouteNetwork, RoutingPhysics


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    feasible: bool
    violations: tuple[str, ...] = field(default_factory=tuple)

    def require(self) -> None:
        if not self.feasible:
            raise ValueError("; ".join(self.violations))


class FeasibilityChecker:
    """Executable OR constraints for LRP-MPPD-2E and MEIO state transitions."""

    def __init__(self, route_network: RouteNetwork, routing_physics: RoutingPhysics) -> None:
        self.route_network = route_network
        self.routing_physics = routing_physics

    def vehicle_capacity(self, vehicle: Vehicle, *, units: float, weight_kg: float, volume_m3: float) -> FeasibilityReport:
        violations: list[str] = []
        if not vehicle.active:
            violations.append("vehicle_inactive")
        if units > vehicle.remaining_units:
            violations.append("vehicle_unit_capacity_exceeded")
        if vehicle.load_weight_kg + weight_kg > vehicle.capacity_weight_kg:
            violations.append("vehicle_weight_capacity_exceeded")
        if vehicle.load_volume_m3 + volume_m3 > vehicle.capacity_volume_m3:
            violations.append("vehicle_volume_capacity_exceeded")
        return FeasibilityReport(not violations, tuple(violations))

    def time_window(self, customer: Customer, arrival_time: float) -> FeasibilityReport:
        violations = []
        if not customer.can_be_served_at(arrival_time):
            violations.append(f"time_window_violated:{customer.customer_id}")
        return FeasibilityReport(not violations, tuple(violations))

    def flow_conservation(self, node_path: Sequence[UUID]) -> FeasibilityReport:
        violations: list[str] = []
        if len(node_path) < 2:
            violations.append("route_path_too_short")
        for origin, destination in zip(node_path, node_path[1:]):
            try:
                self.route_network.get_arc(origin, destination)
            except KeyError:
                violations.append(f"missing_arc:{origin}->{destination}")
        return FeasibilityReport(not violations, tuple(violations))

    def inventory_available(
        self,
        inventory_network: InventoryNetwork,
        *,
        node_id: UUID,
        product_quantities: Mapping[UUID, float],
    ) -> FeasibilityReport:
        violations: list[str] = []
        for product_id, quantity in product_quantities.items():
            try:
                position = inventory_network.get_position(node_id, product_id)
            except KeyError:
                violations.append(f"missing_inventory:{node_id}:{product_id}")
                continue
            if position.available_units < quantity:
                violations.append(f"insufficient_inventory:{node_id}:{product_id}")
        return FeasibilityReport(not violations, tuple(violations))

    def route_order(
        self,
        *,
        order: Order,
        vehicle: Vehicle,
        node_path: Sequence[UUID],
        customer_visits: Mapping[UUID, Customer],
        start_time: float,
    ) -> FeasibilityReport:
        flow = self.flow_conservation(node_path)
        route = self.route_network.evaluate_route(
            vehicle=vehicle,
            node_path=node_path,
            customer_visits=customer_visits,
            start_time=start_time,
            physics=self.routing_physics,
            delivered_units=order.total_units,
        )
        violations = [*flow.violations, *route.violations]
        return FeasibilityReport(not violations, tuple(violations))

    @staticmethod
    def arc_capacity(arc: RouteArc, units: float) -> FeasibilityReport:
        violations = []
        if not arc.active:
            violations.append("arc_inactive")
        if units > arc.capacity_units:
            violations.append("arc_capacity_exceeded")
        return FeasibilityReport(not violations, tuple(violations))

