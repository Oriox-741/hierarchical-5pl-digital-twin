"""SimPy orchestration engine for the Act-layer digital twin."""

from __future__ import annotations

import random
from collections.abc import Generator
from dataclasses import dataclass, field
from uuid import UUID

import simpy

from src.act.disruptions import DisruptionEvent, DisruptionManager
from src.act.entities import Customer, Hub, Order, Product, RouteArc, Vehicle
from src.act.inventory_dynamics import InventoryNetwork, InventoryPosition
from src.act.resources import CapacityState, DockBerthCapacity, FleetCapacity, WarehouseCapacity
from src.act.routing_physics import RouteFeasibilityResult, RouteNetwork, RoutingPhysics
from src.shared.metrics import service_level
from src.shared.types import JsonDict, ShipmentStatus


@dataclass(slots=True)
class SimulationConfig:
    decision_interval: float = 300.0
    random_seed: int | None = None
    default_service_time: float = 60.0


@dataclass(slots=True)
class SimulationState:
    delivered_orders: int = 0
    late_orders: int = 0
    failed_orders: int = 0
    total_transport_cost: float = 0.0
    event_log: list[JsonDict] = field(default_factory=list)

    @property
    def on_time_service_level(self) -> float:
        return service_level(self.delivered_orders - self.late_orders, self.delivered_orders)


class DigitalTwinSimulation:
    """Discrete-event orchestration of routing, inventory, capacity, and disruptions."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.env = simpy.Environment()
        self.rng = random.Random(self.config.random_seed)
        self.state = SimulationState()

        self.products: dict[UUID, Product] = {}
        self.hubs: dict[UUID, Hub] = {}
        self.customers: dict[UUID, Customer] = {}
        self.vehicles: dict[UUID, Vehicle] = {}
        self.orders: dict[UUID, Order] = {}

        self.routing_physics = RoutingPhysics(rng=self.rng)
        self.route_network = RouteNetwork()
        self.inventory_network = InventoryNetwork(self.env)

        self.warehouse_capacity: dict[UUID, WarehouseCapacity] = {}
        self.fleet_capacity: dict[UUID, FleetCapacity] = {}
        self.dock_capacity: dict[UUID, DockBerthCapacity] = {}
        self.capacity_states: dict[UUID, CapacityState] = {}
        self.disruptions = DisruptionManager(
            self.env,
            routing_physics=self.routing_physics,
            inventory_network=self.inventory_network,
            capacity_states=self.capacity_states,
            rng=self.rng,
        )

    def add_product(self, product: Product) -> None:
        self.products[product.product_id] = product

    def add_hub(self, hub: Hub, *, initial_inventory_units: float = 0.0, dock_count: int = 1) -> None:
        self.hubs[hub.hub_id] = hub
        self.warehouse_capacity[hub.hub_id] = WarehouseCapacity(
            self.env,
            hub.hub_id,
            hub.storage_capacity_units,
            min(initial_inventory_units, hub.storage_capacity_units),
        )
        self.dock_capacity[hub.hub_id] = DockBerthCapacity(self.env, hub.hub_id, dock_count)
        self.capacity_states[hub.hub_id] = CapacityState(hub.throughput_units_per_time)

    def add_customer(self, customer: Customer) -> None:
        self.customers[customer.customer_id] = customer

    def add_vehicle(self, vehicle: Vehicle) -> None:
        self.vehicles[vehicle.vehicle_id] = vehicle

    def add_fleet_capacity(self, fleet_id: UUID, vehicle_count: int) -> None:
        self.fleet_capacity[fleet_id] = FleetCapacity(self.env, fleet_id, vehicle_count)

    def add_route_arc(self, arc: RouteArc) -> None:
        self.route_network.add_arc(arc)

    def add_inventory_position(self, position: InventoryPosition) -> None:
        self.inventory_network.add_position(position)

    def add_order(self, order: Order) -> None:
        self.orders[order.order_id] = order

    def schedule_disruption(self, event: DisruptionEvent) -> simpy.Process:
        return self.disruptions.schedule(event)

    def dispatch_order(self, order_id: UUID, vehicle_id: UUID, node_path: list[UUID]) -> simpy.Process:
        order = self.orders[order_id]
        vehicle = self.vehicles[vehicle_id]
        return self.env.process(self._dispatch_process(order, vehicle, node_path))

    def evaluate_order_route(
        self,
        order: Order,
        vehicle: Vehicle,
        node_path: list[UUID],
        *,
        start_time: float | None = None,
    ) -> RouteFeasibilityResult:
        customer_visits = {
            customer_id: customer
            for customer_id, customer in self.customers.items()
            if customer_id == order.customer_id
        }
        return self.route_network.evaluate_route(
            vehicle=vehicle,
            node_path=node_path,
            customer_visits=customer_visits,
            start_time=self.env.now if start_time is None else float(start_time),
            physics=self.routing_physics,
            delivered_units=order.total_units,
        )

    def run(self, until: float | None = None) -> None:
        self.env.run(until=until)

    def run_until_next_decision(self) -> None:
        self.env.run(until=self.env.now + self.config.decision_interval)

    def snapshot(self) -> JsonDict:
        return {
            "time": self.env.now,
            "orders": {
                str(order_id): {
                    "status": order.status.value,
                    "assigned_vehicle_id": str(order.assigned_vehicle_id) if order.assigned_vehicle_id else None,
                    "pickup_time": order.pickup_time,
                    "delivery_time": order.delivery_time,
                    "due_time": order.due_time,
                    "is_late": order.is_late,
                }
                for order_id, order in self.orders.items()
            },
            "vehicles": {
                str(vehicle_id): {
                    "tier": vehicle.tier.value,
                    "asset_kind": vehicle.asset_kind.value,
                    "load_units": vehicle.load_units,
                    "utilization": vehicle.utilization,
                    "current_node_id": str(vehicle.current_node_id) if vehicle.current_node_id else None,
                }
                for vehicle_id, vehicle in self.vehicles.items()
            },
            "inventory": self.inventory_network.snapshot(),
            "disruption_score": self.disruptions.disruption_score(),
            "network_safety_potential": self.inventory_network.network_safety_potential(
                disruption_score=self.disruptions.disruption_score()
            ),
            "service_level": self.state.on_time_service_level,
            "total_transport_cost": self.state.total_transport_cost,
            "event_log": [*self.state.event_log, *self.disruptions.event_log],
        }

    def _dispatch_process(self, order: Order, vehicle: Vehicle, node_path: list[UUID]) -> Generator[simpy.Event, None, None]:
        if self.env.now < order.release_time:
            yield self.env.timeout(order.release_time - self.env.now)

        product_weights = self._order_weight_volume(order)
        feasibility = self.evaluate_order_route(order, vehicle, node_path, start_time=self.env.now)
        if not feasibility.feasible:
            order.status = ShipmentStatus.DELAYED
            order.assigned_vehicle_id = None
            order.pickup_time = None
            self.state.failed_orders += 1
            self.state.event_log.append(
                {
                    "time": self.env.now,
                    "event": "route_feasibility_failed",
                    "order_id": str(order.order_id),
                    "vehicle_id": str(vehicle.vehicle_id),
                    "violations": list(feasibility.violations),
                    "failed_route_vehicle_rollback_count": 1,
                }
            )
            return

        vehicle.load(
            units=order.total_units,
            weight_kg=product_weights["weight_kg"],
            volume_m3=product_weights["volume_m3"],
        )
        order.mark_in_transit(vehicle.vehicle_id, self.env.now)

        if vehicle.fixed_usage_cost > 0.0:
            self.state.total_transport_cost += vehicle.fixed_usage_cost
            self.state.event_log.append(
                {
                    "time": self.env.now,
                    "event": "vehicle_fixed_usage_cost",
                    "order_id": str(order.order_id),
                    "vehicle_id": str(vehicle.vehicle_id),
                    "vehicle_tier": vehicle.tier.value,
                    "fixed_usage_cost": vehicle.fixed_usage_cost,
                }
            )

        for arc in self.route_network.route_arcs(node_path):
            travel_time = self.routing_physics.transit_time_seconds(arc, vehicle)
            yield self.env.timeout(travel_time)
            vehicle.current_node_id = arc.destination_node_id
            cost = self.routing_physics.traversal_cost_breakdown(arc, vehicle)
            self.state.total_transport_cost += cost.total_cost
            self.state.event_log.append(
                {
                    "time": self.env.now,
                    "event": "route_arc_traversed",
                    "order_id": str(order.order_id),
                    "vehicle_id": str(vehicle.vehicle_id),
                    "origin_node_id": str(arc.origin_node_id),
                    "destination_node_id": str(arc.destination_node_id),
                    **cost.as_dict(),
                }
            )

        order.mark_delivered(self.env.now)
        vehicle.unload(
            units=order.total_units,
            weight_kg=product_weights["weight_kg"],
            volume_m3=product_weights["volume_m3"],
        )

        customer = self.customers.get(order.customer_id)
        if customer is not None:
            customer.visited = True

        self.state.delivered_orders += 1
        if order.is_late:
            self.state.late_orders += 1

        self.state.event_log.append(
            {
                "time": self.env.now,
                "event": "order_delivered",
                "order_id": str(order.order_id),
                "vehicle_id": str(vehicle.vehicle_id),
                "late": order.is_late,
                "transport_cost": feasibility.total_cost,
            }
        )

    def _order_weight_volume(self, order: Order) -> dict[str, float]:
        weight = 0.0
        volume = 0.0
        for line in order.lines:
            product = self.products.get(line.product_id)
            if product is None:
                weight += line.quantity_units
                volume += line.quantity_units * 0.01
            else:
                weight += line.quantity_units * product.unit_weight_kg
                volume += line.quantity_units * product.unit_volume_m3
        return {"weight_kg": weight, "volume_m3": volume}
