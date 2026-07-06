"""Routing physics and LRP-MPPD-2E feasibility rules for the digital twin."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from uuid import UUID

from src.act.entities import Customer, RouteArc, Vehicle, VehicleTier
from src.shared.constants import MAX_LATITUDE, MAX_LONGITUDE, MIN_LATITUDE, MIN_LONGITUDE
from src.shared.metrics import clamp
from src.shared.types import GeoPoint, JsonDict


EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True, slots=True)
class RouteFeasibilityResult:
    feasible: bool
    violations: tuple[str, ...] = ()
    total_distance_m: float = 0.0
    total_transit_seconds: float = 0.0
    total_cost: float = 0.0


@dataclass(frozen=True, slots=True)
class RouteTraversalCost:
    """Cost breakdown for one arc traversal under physical speed economics."""

    total_cost: float
    distance_cost: float
    speed_cost: float
    speed_ratio: float
    energy_multiplier: float
    risk_multiplier: float
    effective_speed_mps: float
    baseline_speed_mps: float

    def as_dict(self) -> JsonDict:
        return {
            "total_cost": self.total_cost,
            "distance_cost": self.distance_cost,
            "speed_cost": self.speed_cost,
            "speed_ratio": self.speed_ratio,
            "energy_multiplier": self.energy_multiplier,
            "risk_multiplier": self.risk_multiplier,
            "effective_speed_mps": self.effective_speed_mps,
            "baseline_speed_mps": self.baseline_speed_mps,
        }


@dataclass(slots=True)
class RoutingPhysics:
    """Dynamic travel-time model with congestion, disruption, and noise."""

    congestion_by_arc: dict[tuple[UUID, UUID], float] = field(default_factory=dict)
    arc_delay_multiplier: dict[tuple[UUID, UUID], float] = field(default_factory=dict)
    speed_cost_coefficient: float = 1.25
    speed_risk_coefficient: float = 0.75
    traversal_cost_multiplier: float = 1.0
    disruption_risk_multiplier: float = 1.0
    stochastic_noise_std: float = 0.03
    rng: random.Random = field(default_factory=random.Random)

    def transit_time_seconds(self, arc: RouteArc, vehicle: Vehicle) -> float:
        """Calculate dynamic arc travel time as distance / speed with modifiers."""
        if not arc.active:
            return float("inf")

        key = (arc.origin_node_id, arc.destination_node_id)
        congestion = clamp(self.congestion_by_arc.get(key, 0.0), 0.0, 1.0)
        delay_multiplier = max(1.0, self.arc_delay_multiplier.get(key, 1.0))
        base_speed = effective_speed_mps(arc, vehicle)
        congestion_multiplier = 1.0 + (2.5 * congestion)
        noise = max(0.0, self.rng.gauss(0.0, self.stochastic_noise_std))

        return (arc.distance_m / base_speed) * congestion_multiplier * delay_multiplier * (1.0 + noise)

    def traversal_cost(self, arc: RouteArc, vehicle: Vehicle) -> float:
        return self.traversal_cost_breakdown(arc, vehicle).total_cost

    def traversal_cost_breakdown(self, arc: RouteArc, vehicle: Vehicle) -> RouteTraversalCost:
        if self.speed_cost_coefficient < 0.0:
            raise ValueError("speed_cost_coefficient must not be negative.")
        if self.speed_risk_coefficient < 0.0:
            raise ValueError("speed_risk_coefficient must not be negative.")

        effective_speed = effective_speed_mps(arc, vehicle)
        baseline_speed = vehicle_baseline_speed_mps(vehicle)
        speed_ratio = effective_speed / baseline_speed
        excess_speed_ratio = max(0.0, speed_ratio - 1.0)
        distance_cost = arc.distance_m * (arc.traversal_cost_per_unit + vehicle.transport_cost_per_meter)
        energy_multiplier = 1.0 + (self.speed_cost_coefficient * (excess_speed_ratio**2))
        risk_multiplier = 1.0 + (
            max(0.0, self.disruption_risk_multiplier)
            * self.speed_risk_coefficient
            * (excess_speed_ratio**3)
        )
        speed_cost = distance_cost * (energy_multiplier + risk_multiplier - 2.0)
        total_cost = (distance_cost + speed_cost) * max(0.0, self.traversal_cost_multiplier)
        return RouteTraversalCost(
            total_cost=max(0.0, total_cost),
            distance_cost=max(0.0, distance_cost),
            speed_cost=max(0.0, speed_cost),
            speed_ratio=max(0.0, speed_ratio),
            energy_multiplier=max(1.0, energy_multiplier),
            risk_multiplier=max(1.0, risk_multiplier),
            effective_speed_mps=effective_speed,
            baseline_speed_mps=baseline_speed,
        )

    def set_congestion(self, origin_node_id: UUID, destination_node_id: UUID, score: float) -> None:
        self.congestion_by_arc[(origin_node_id, destination_node_id)] = clamp(score, 0.0, 1.0)

    def set_arc_delay_multiplier(self, origin_node_id: UUID, destination_node_id: UUID, multiplier: float) -> None:
        self.arc_delay_multiplier[(origin_node_id, destination_node_id)] = max(1.0, multiplier)

    def clear_arc_delay(self, origin_node_id: UUID, destination_node_id: UUID) -> None:
        self.arc_delay_multiplier.pop((origin_node_id, destination_node_id), None)


class RouteNetwork:
    """Directed logistics graph with feasibility checks for route execution."""

    def __init__(self, arcs: Sequence[RouteArc] = ()) -> None:
        self._arcs: dict[tuple[UUID, UUID], RouteArc] = {}
        self._adjacency: dict[UUID, set[UUID]] = {}
        for arc in arcs:
            self.add_arc(arc)

    @property
    def arcs(self) -> Mapping[tuple[UUID, UUID], RouteArc]:
        return self._arcs

    def add_arc(self, arc: RouteArc) -> None:
        key = (arc.origin_node_id, arc.destination_node_id)
        self._arcs[key] = arc
        self._adjacency.setdefault(arc.origin_node_id, set()).add(arc.destination_node_id)
        self._adjacency.setdefault(arc.destination_node_id, set())

    def get_arc(self, origin_node_id: UUID, destination_node_id: UUID) -> RouteArc:
        try:
            return self._arcs[(origin_node_id, destination_node_id)]
        except KeyError as exc:
            raise KeyError(f"missing route arc {origin_node_id} -> {destination_node_id}") from exc

    def route_arcs(self, node_path: Sequence[UUID]) -> list[RouteArc]:
        if len(node_path) < 2:
            return []
        return [self.get_arc(a, b) for a, b in zip(node_path, node_path[1:])]

    def shortest_path(self, origin_node_id: UUID, destination_node_id: UUID) -> list[UUID]:
        """Dijkstra shortest path over active arcs using physical distance as cost."""
        if origin_node_id == destination_node_id:
            return [origin_node_id]

        distances: dict[UUID, float] = {origin_node_id: 0.0}
        previous: dict[UUID, UUID] = {}
        unvisited = set(self._adjacency)

        while unvisited:
            current = min(unvisited, key=lambda node: distances.get(node, float("inf")))
            if current == destination_node_id or distances.get(current, float("inf")) == float("inf"):
                break
            unvisited.remove(current)

            for neighbor in self._adjacency.get(current, ()):
                arc = self._arcs[(current, neighbor)]
                if not arc.active:
                    continue
                candidate = distances[current] + arc.distance_m
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    previous[neighbor] = current

        if destination_node_id not in distances:
            raise ValueError(f"no active path from {origin_node_id} to {destination_node_id}")

        path = [destination_node_id]
        while path[-1] != origin_node_id:
            path.append(previous[path[-1]])
        path.reverse()
        return path

    def evaluate_route(
        self,
        *,
        vehicle: Vehicle,
        node_path: Sequence[UUID],
        customer_visits: Mapping[UUID, Customer],
        start_time: float,
        physics: RoutingPhysics,
        delivered_units: float,
    ) -> RouteFeasibilityResult:
        """Encode LRP-MPPD-2E route feasibility as executable checks."""
        violations: list[str] = []
        if not vehicle.active:
            violations.append("vehicle_inactive")
        if vehicle.tier is VehicleTier.SECONDARY and len(customer_visits) != len(set(customer_visits)):
            violations.append("customer_visit_uniqueness_failed")
        if delivered_units > vehicle.capacity_units:
            violations.append("vehicle_capacity_exceeded")

        current_time = start_time
        total_distance = 0.0
        total_cost = vehicle.fixed_usage_cost
        route_customer_visits: set[UUID] = set()

        for arc in self.route_arcs(node_path):
            if not arc.active:
                violations.append(f"inactive_arc:{arc.origin_node_id}->{arc.destination_node_id}")
            if delivered_units > arc.capacity_units:
                violations.append(f"arc_capacity_exceeded:{arc.origin_node_id}->{arc.destination_node_id}")

            transit_time = physics.transit_time_seconds(arc, vehicle)
            current_time += transit_time
            total_distance += arc.distance_m
            total_cost += physics.traversal_cost(arc, vehicle)

            customer = customer_visits.get(arc.destination_node_id)
            if customer is not None:
                if customer.customer_id in route_customer_visits:
                    violations.append(f"customer_revisited:{customer.customer_id}")
                route_customer_visits.add(customer.customer_id)
                if not customer.can_be_served_at(current_time):
                    violations.append(f"time_window_violated:{customer.customer_id}")
                current_time += customer.service_time

        return RouteFeasibilityResult(
            feasible=not violations,
            violations=tuple(violations),
            total_distance_m=total_distance,
            total_transit_seconds=max(0.0, current_time - start_time),
            total_cost=total_cost,
        )


def haversine_distance_m(a: GeoPoint, b: GeoPoint) -> float:
    """Distance between WGS84 points in meters."""
    _validate_point(a)
    _validate_point(b)
    lat1 = math.radians(a.latitude)
    lat2 = math.radians(b.latitude)
    dlat = lat2 - lat1
    dlon = math.radians(b.longitude - a.longitude)
    hav = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(hav))


def effective_speed_mps(arc: RouteArc, vehicle: Vehicle) -> float:
    return max(0.1, min(vehicle.nominal_speed_mps, arc.nominal_speed_mps))


def vehicle_baseline_speed_mps(vehicle: Vehicle) -> float:
    baseline = vehicle.baseline_speed_mps
    if baseline is None:
        metadata_value = vehicle.metadata.get("baseline_speed_mps")
        try:
            baseline = float(metadata_value) if metadata_value is not None else vehicle.nominal_speed_mps
        except (TypeError, ValueError):
            baseline = vehicle.nominal_speed_mps
        vehicle.baseline_speed_mps = baseline
    baseline = max(0.1, float(baseline))
    vehicle.metadata["baseline_speed_mps"] = baseline
    return baseline


def make_arc_from_points(
    *,
    origin_node_id: UUID,
    destination_node_id: UUID,
    origin: GeoPoint,
    destination: GeoPoint,
    nominal_speed_mps: float,
    capacity_units: float = float("inf"),
    metadata: JsonDict | None = None,
) -> RouteArc:
    return RouteArc(
        origin_node_id=origin_node_id,
        destination_node_id=destination_node_id,
        distance_m=haversine_distance_m(origin, destination),
        nominal_speed_mps=nominal_speed_mps,
        capacity_units=capacity_units,
        metadata=metadata or {},
    )


def _validate_point(point: GeoPoint) -> None:
    if not MIN_LATITUDE <= point.latitude <= MAX_LATITUDE:
        raise ValueError(f"latitude out of range: {point.latitude}")
    if not MIN_LONGITUDE <= point.longitude <= MAX_LONGITUDE:
        raise ValueError(f"longitude out of range: {point.longitude}")
