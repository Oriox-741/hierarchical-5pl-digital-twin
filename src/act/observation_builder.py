"""Build flat normalized observation vectors from simulation and database snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from src.act.sim_engine import DigitalTwinSimulation
from src.shared.metrics import clamp


LEGACY_OBSERVATION_FEATURES: Final[tuple[str, ...]] = (
    "time_ratio",
    "pending_order_ratio",
    "delivered_order_ratio",
    "late_delivery_ratio",
    "failed_order_ratio",
    "service_level",
    "transport_cost_ratio",
    "on_hand_inventory_share",
    "reserved_inventory_share",
    "in_transit_inventory_share",
    "backlog_inventory_share",
    "safety_stock_inventory_share",
    "demand_mean_pressure",
    "demand_variance_pressure",
    "lead_time_mean_pressure",
    "lead_time_variance_pressure",
    "vehicle_utilization",
    "active_vehicle_ratio",
    "capacity_pressure",
    "disruption_score",
    "network_safety_potential",
    "hub_count_ratio",
    "customer_count_ratio",
    "route_arc_count_ratio",
    "db_asset_count_ratio",
    "db_avg_speed_ratio",
    "db_avg_battery_ratio",
    "db_avg_sensor_quality_ratio",
    "db_safety_potential",
    "db_disruption_score",
    "db_congestion_score",
    "bias",
)

REALITY_OBSERVATION_FEATURES: Final[tuple[str, ...]] = (
    "inventory_coverage_ratio",
    "stockout_risk",
    "safety_stock_target_gap",
    "demand_volatility_pressure",
    "lead_time_volatility_pressure",
    "backlog_age_pressure",
    "urgent_order_ratio",
    "speed_cost_exposure",
    "mean_speed_ratio",
    "premium_fleet_exposure",
    "route_disruption_pressure",
    "capacity_slack",
)

DISPATCH_FEASIBILITY_OBSERVATION_FEATURES: Final[tuple[str, ...]] = (
    "normalized_available_vehicle_count",
    "normalized_dispatchable_order_count",
    "feasible_dispatch_opportunity",
    "feasible_dispatch_ratio",
    "vehicle_availability_pressure",
)

REAL_WORLD_STRESS_OBSERVATION_FEATURES: Final[tuple[str, ...]] = (
    "normalized_holding_cost_pressure",
    "normalized_stockout_penalty_pressure",
    "capacity_shock_pressure",
    "scenario_route_disruption_pressure",
    "route_cost_pressure",
    "premium_sla_pressure",
    "supplier_delay_pressure",
    "scenario_lead_time_volatility_pressure",
)

ROUTE_CANDIDATE_OBSERVATION_FEATURES: Final[tuple[str, ...]] = (
    "shortest_candidate_score",
    "low_congestion_candidate_score",
    "high_resilience_candidate_score",
    "shortest_candidate_score_gap",
    "low_congestion_candidate_score_gap",
    "high_resilience_candidate_score_gap",
    "route_pressure_reliability_share",
    "route_pressure_congestion_share",
    "route_pressure_balance",
    "route_alt_pressure_imbalance",
    "shortest_candidate_near_best",
    "shortest_secondary_safe_context",
    "shortest_secondary_brittle_risk",
    "secondary_fleet_feasible_dispatch_ratio",
    "secondary_fleet_vehicle_availability_pressure",
    "useful_dispatch_opportunity",
)

OBSERVATION_FEATURES: Final[tuple[str, ...]] = (
    LEGACY_OBSERVATION_FEATURES
    + REALITY_OBSERVATION_FEATURES
    + DISPATCH_FEASIBILITY_OBSERVATION_FEATURES
    + REAL_WORLD_STRESS_OBSERVATION_FEATURES
    + ROUTE_CANDIDATE_OBSERVATION_FEATURES
)
OBSERVATION_FEATURE_INDEX: Final[dict[str, int]] = {name: index for index, name in enumerate(OBSERVATION_FEATURES)}
OBSERVATION_DIM: Final[int] = len(OBSERVATION_FEATURES)


@dataclass(frozen=True, slots=True)
class ObservationBuildResult:
    vector: np.ndarray
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ObservationHotPathContext:
    positions: tuple[Any, ...]
    vehicles: tuple[Any, ...]
    orders: tuple[Any, ...]
    capacity_states: tuple[Any, ...]
    current_time: float
    dispatchable_orders: tuple[Any, ...]
    minimum_required_units: float
    available_vehicle_count: float
    secondary_available_count: float
    route_congestion: float

    @property
    def total_order_count(self) -> int:
        return len(self.orders)

    @property
    def total_vehicle_count(self) -> int:
        return len(self.vehicles)


class ObservationBuilder:
    """Convert SimPy and optional DB snapshots into a fixed Box-compatible vector."""

    def __init__(self, *, max_time: float = 86_400.0, max_cost: float = 1_000_000.0) -> None:
        if max_time <= 0.0:
            raise ValueError("max_time must be positive.")
        if max_cost <= 0.0:
            raise ValueError("max_cost must be positive.")
        self.max_time = max_time
        self.max_cost = max_cost

    @property
    def dimension(self) -> int:
        return OBSERVATION_DIM

    def build(
        self,
        simulation: DigitalTwinSimulation,
        *,
        snapshot: Mapping[str, Any] | None = None,
        db_snapshots: Sequence[Mapping[str, Any]] | None = None,
        scenario_stress_state: Mapping[str, Any] | None = None,
        hot_path_context: _ObservationHotPathContext | None = None,
    ) -> ObservationBuildResult:
        snapshot = snapshot if snapshot is not None else simulation.snapshot()
        context = hot_path_context or self._build_hot_path_context(simulation, snapshot)
        inventory = context.positions
        vehicles = context.vehicles
        orders = context.orders
        capacity_states = context.capacity_states

        pending_orders = [order for order in orders if order.delivery_time is None]
        delivered_orders = [order for order in orders if order.delivery_time is not None]
        late_orders = [order for order in delivered_orders if order.is_late]

        on_hand = sum(position.on_hand_units for position in inventory)
        reserved = sum(position.reserved_units for position in inventory)
        in_transit = sum(position.in_transit_units for position in inventory)
        backlog = sum(position.backlog_units for position in inventory)
        safety_stock = sum(position.safety_stock_units for position in inventory)
        demand_mean = sum(position.demand_mean for position in inventory)
        demand_var = sum(position.demand_variance for position in inventory)
        lead_mean = sum(position.lead_time_mean for position in inventory)
        lead_var = sum(position.lead_time_variance for position in inventory)

        vehicle_count = max(len(vehicles), 1)
        vehicle_util = sum(vehicle.utilization for vehicle in vehicles) / vehicle_count
        active_vehicle_ratio = sum(1 for vehicle in vehicles if vehicle.active) / vehicle_count
        capacity_pressure = (
            sum(state.capacity_pressure for state in capacity_states) / len(capacity_states)
            if capacity_states
            else 0.0
        )
        db_metrics = self._db_metrics(db_snapshots or ())

        reality_features = _reality_features(simulation, snapshot, context=context)
        dispatch_feasibility_features = _dispatch_feasibility_features(simulation, context=context)
        stress_features = _real_world_stress_features(scenario_stress_state or {})
        route_candidate_features = _route_candidate_visibility_features(
            simulation,
            snapshot,
            scenario_stress_state or {},
            dispatch_feasibility_features=dispatch_feasibility_features,
            stress_features=stress_features,
            context=context,
        )

        values = [
            snapshot["time"] / self.max_time,
            len(pending_orders) / max(len(orders), 1),
            len(delivered_orders) / max(len(orders), 1),
            len(late_orders) / max(len(delivered_orders), 1),
            simulation.state.failed_orders / max(len(orders), 1),
            snapshot["service_level"],
            snapshot["total_transport_cost"] / self.max_cost,
            on_hand / max(on_hand + reserved + in_transit + backlog + safety_stock, 1.0),
            reserved / max(on_hand + reserved + in_transit + backlog + safety_stock, 1.0),
            in_transit / max(on_hand + reserved + in_transit + backlog + safety_stock, 1.0),
            backlog / max(on_hand + reserved + in_transit + backlog + safety_stock, 1.0),
            safety_stock / max(on_hand + reserved + in_transit + backlog + safety_stock, 1.0),
            demand_mean / max(demand_mean + on_hand + 1.0, 1.0),
            demand_var / max(demand_var + demand_mean + 1.0, 1.0),
            lead_mean / max(lead_mean + 1_000.0, 1.0),
            lead_var / max(lead_var + 1_000.0, 1.0),
            vehicle_util,
            active_vehicle_ratio,
            capacity_pressure,
            snapshot["disruption_score"],
            snapshot["network_safety_potential"],
            len(simulation.hubs) / 100.0,
            len(simulation.customers) / 10_000.0,
            len(simulation.route_network.arcs) / 100_000.0,
            db_metrics["asset_count"] / 10_000.0,
            db_metrics["avg_speed_mps"] / 50.0,
            db_metrics["avg_battery_pct"] / 100.0,
            db_metrics["avg_sensor_quality"] / 100.0,
            db_metrics["db_safety_potential"],
            db_metrics["db_disruption_score"],
            db_metrics["db_congestion_score"],
            1.0,
            reality_features["inventory_coverage_ratio"],
            reality_features["stockout_risk"],
            reality_features["safety_stock_target_gap"],
            reality_features["demand_volatility_pressure"],
            reality_features["lead_time_volatility_pressure"],
            reality_features["backlog_age_pressure"],
            reality_features["urgent_order_ratio"],
            reality_features["speed_cost_exposure"],
            reality_features["mean_speed_ratio"],
            reality_features["premium_fleet_exposure"],
            reality_features["route_disruption_pressure"],
            reality_features["capacity_slack"],
            dispatch_feasibility_features["normalized_available_vehicle_count"],
            dispatch_feasibility_features["normalized_dispatchable_order_count"],
            dispatch_feasibility_features["feasible_dispatch_opportunity"],
            dispatch_feasibility_features["feasible_dispatch_ratio"],
            dispatch_feasibility_features["vehicle_availability_pressure"],
            stress_features["normalized_holding_cost_pressure"],
            stress_features["normalized_stockout_penalty_pressure"],
            stress_features["capacity_shock_pressure"],
            stress_features["scenario_route_disruption_pressure"],
            stress_features["route_cost_pressure"],
            stress_features["premium_sla_pressure"],
            stress_features["supplier_delay_pressure"],
            stress_features["scenario_lead_time_volatility_pressure"],
            route_candidate_features["shortest_candidate_score"],
            route_candidate_features["low_congestion_candidate_score"],
            route_candidate_features["high_resilience_candidate_score"],
            route_candidate_features["shortest_candidate_score_gap"],
            route_candidate_features["low_congestion_candidate_score_gap"],
            route_candidate_features["high_resilience_candidate_score_gap"],
            route_candidate_features["route_pressure_reliability_share"],
            route_candidate_features["route_pressure_congestion_share"],
            route_candidate_features["route_pressure_balance"],
            route_candidate_features["route_alt_pressure_imbalance"],
            route_candidate_features["shortest_candidate_near_best"],
            route_candidate_features["shortest_secondary_safe_context"],
            route_candidate_features["shortest_secondary_brittle_risk"],
            route_candidate_features["secondary_fleet_feasible_dispatch_ratio"],
            route_candidate_features["secondary_fleet_vehicle_availability_pressure"],
            route_candidate_features["useful_dispatch_opportunity"],
        ]

        vector = np.array([clamp(float(value), 0.0, 1.0) for value in values], dtype=np.float32)
        if vector.shape[0] != OBSERVATION_DIM:
            raise RuntimeError(f"observation dimension mismatch: expected {OBSERVATION_DIM}, got {vector.shape[0]}.")
        return ObservationBuildResult(
            vector=vector,
            raw={
                "snapshot": snapshot,
                "features": {
                    name: float(vector[index])
                    for index, name in enumerate(OBSERVATION_FEATURES)
                },
            },
        )

    def _build_hot_path_context(
        self,
        simulation: DigitalTwinSimulation,
        snapshot: Mapping[str, Any],
    ) -> _ObservationHotPathContext:
        return _build_hot_path_context(simulation, snapshot)

    @staticmethod
    def _db_metrics(db_snapshots: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        if not db_snapshots:
            return {
                "asset_count": 0.0,
                "avg_speed_mps": 0.0,
                "avg_battery_pct": 0.0,
                "avg_sensor_quality": 0.0,
                "db_safety_potential": 0.0,
                "db_disruption_score": 0.0,
                "db_congestion_score": 0.0,
            }

        count = float(len(db_snapshots))
        return {
            "asset_count": count,
            "avg_speed_mps": _avg(db_snapshots, "speed_mps"),
            "avg_battery_pct": _avg(db_snapshots, "battery_pct"),
            "avg_sensor_quality": _avg(db_snapshots, "sensor_quality"),
            "db_safety_potential": _avg(db_snapshots, "safety_potential"),
            "db_disruption_score": _avg(db_snapshots, "disruption_score"),
            "db_congestion_score": _avg(db_snapshots, "congestion_score"),
        }


def _avg(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _build_hot_path_context(
    simulation: DigitalTwinSimulation,
    snapshot: Mapping[str, Any],
) -> _ObservationHotPathContext:
    positions = tuple(simulation.inventory_network.positions.values())
    vehicles = tuple(simulation.vehicles.values())
    orders = tuple(simulation.orders.values())
    capacity_states = tuple(simulation.capacity_states.values())
    dispatchable_orders = tuple(order for order in orders if _is_dispatchable_order(order))
    minimum_required_units = (
        min((max(0.0, float(getattr(order, "total_units", 0.0))) for order in dispatchable_orders), default=0.0)
        if dispatchable_orders
        else 0.0
    )
    available_vehicle_count = float(
        sum(
            1
            for vehicle in vehicles
            if bool(getattr(vehicle, "active", False))
            and float(getattr(vehicle, "remaining_units", 0.0)) >= minimum_required_units
            and float(getattr(vehicle, "remaining_units", 0.0)) > 0.0
        )
    )
    secondary_available_count = float(
        sum(
            1
            for vehicle in vehicles
            if _tier_value(vehicle) == "secondary"
            and bool(getattr(vehicle, "active", False))
            and float(getattr(vehicle, "remaining_units", 0.0)) >= minimum_required_units
            and float(getattr(vehicle, "remaining_units", 0.0)) > 0.0
        )
    )
    route_congestion = _mean(
        clamp(float(item), 0.0, 1.0)
        for item in simulation.routing_physics.congestion_by_arc.values()
    )
    return _ObservationHotPathContext(
        positions=positions,
        vehicles=vehicles,
        orders=orders,
        capacity_states=capacity_states,
        current_time=float(snapshot.get("time", 0.0)),
        dispatchable_orders=dispatchable_orders,
        minimum_required_units=minimum_required_units,
        available_vehicle_count=available_vehicle_count,
        secondary_available_count=secondary_available_count,
        route_congestion=route_congestion,
    )


def _reality_features(
    simulation: DigitalTwinSimulation,
    snapshot: Mapping[str, Any],
    *,
    context: _ObservationHotPathContext | None = None,
) -> dict[str, float]:
    context = context or _build_hot_path_context(simulation, snapshot)
    positions = context.positions
    vehicles = context.vehicles
    orders = context.orders
    capacity_states = context.capacity_states

    return {
        "inventory_coverage_ratio": _inventory_coverage_ratio(positions),
        "stockout_risk": _stockout_risk(positions),
        "safety_stock_target_gap": _safety_stock_target_gap(positions),
        "demand_volatility_pressure": _mean(
            clamp(position.demand_variance / max(position.demand_mean + position.demand_variance + 1.0, 1.0), 0.0, 1.0)
            for position in positions
        ),
        "lead_time_volatility_pressure": _mean(
            clamp(
                position.lead_time_variance / max(position.lead_time_mean + position.lead_time_variance + 1.0, 1.0),
                0.0,
                1.0,
            )
            for position in positions
        ),
        "backlog_age_pressure": _backlog_age_pressure(orders, current_time=context.current_time),
        "urgent_order_ratio": _urgent_order_ratio(orders, current_time=context.current_time),
        "speed_cost_exposure": _speed_cost_exposure(vehicles),
        "mean_speed_ratio": _mean_speed_ratio(vehicles),
        "premium_fleet_exposure": _premium_fleet_exposure(snapshot),
        "route_disruption_pressure": _route_disruption_pressure(
            simulation,
            snapshot,
            route_congestion=context.route_congestion,
        ),
        "capacity_slack": 1.0 - _capacity_pressure(capacity_states),
    }


def _inventory_coverage_ratio(positions: Sequence[Any]) -> float:
    return _mean(
        clamp(
            position.on_hand_units
            / max((position.demand_mean * max(position.lead_time_mean, 1.0)) + 1.0, 1.0),
            0.0,
            1.0,
        )
        for position in positions
    )


def _stockout_risk(positions: Sequence[Any]) -> float:
    return _mean(
        clamp(
            (position.backlog_units + position.reserved_units)
            / max(position.on_hand_units + position.in_transit_units + position.safety_stock_units, 1.0),
            0.0,
            1.0,
        )
        for position in positions
    )


def _safety_stock_target_gap(positions: Sequence[Any]) -> float:
    return _mean(
        clamp(
            (position.target_safety_stock() - position.safety_stock_units)
            / max(position.target_safety_stock(), 1.0),
            0.0,
            1.0,
        )
        for position in positions
    )


def _backlog_age_pressure(orders: Sequence[Any], *, current_time: float) -> float:
    pressures: list[float] = []
    for order in orders:
        if getattr(order, "delivery_time", None) is not None:
            continue
        release_time = float(getattr(order, "release_time", current_time))
        due_time = float(getattr(order, "due_time", current_time))
        service_window = max(due_time - release_time, 1.0)
        age = max(0.0, current_time - release_time)
        pressures.append(clamp(age / service_window, 0.0, 1.0))
    return _mean(pressures)


def _urgent_order_ratio(orders: Sequence[Any], *, current_time: float) -> float:
    pending = 0
    urgent = 0
    for order in orders:
        if getattr(order, "delivery_time", None) is not None:
            continue
        pending += 1
        due_time = float(getattr(order, "due_time", current_time))
        if due_time - current_time <= 7_200.0:
            urgent += 1
    return clamp(urgent / max(pending, 1), 0.0, 1.0)


def _speed_cost_exposure(vehicles: Sequence[Any]) -> float:
    return _mean(max(0.0, _vehicle_speed_ratio(vehicle) - 1.0) ** 2 for vehicle in vehicles)


def _mean_speed_ratio(vehicles: Sequence[Any]) -> float:
    return _mean(clamp(_vehicle_speed_ratio(vehicle) / 1.35, 0.0, 1.0) for vehicle in vehicles)


def _vehicle_speed_ratio(vehicle: Any) -> float:
    baseline = getattr(vehicle, "baseline_speed_mps", None)
    if baseline is None:
        metadata = getattr(vehicle, "metadata", {})
        baseline = metadata.get("baseline_speed_mps", getattr(vehicle, "nominal_speed_mps", 1.0))
    return float(getattr(vehicle, "nominal_speed_mps", 0.0)) / max(float(baseline), 1e-6)


def _premium_fleet_exposure(snapshot: Mapping[str, Any]) -> float:
    events = snapshot.get("event_log", ())
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes, bytearray)):
        return 0.0
    dispatch_count = 0
    primary_count = 0
    for event in events:
        if not isinstance(event, Mapping) or event.get("event") != "dispatch_decision":
            continue
        dispatch_count += 1
        if event.get("mode_decision") == "primary_fleet":
            primary_count += 1
    if dispatch_count <= 0:
        return 0.0
    return clamp(primary_count / dispatch_count, 0.0, 1.0)


def _route_disruption_pressure(
    simulation: DigitalTwinSimulation,
    snapshot: Mapping[str, Any],
    *,
    route_congestion: float | None = None,
) -> float:
    delay_values = [
        clamp((float(value) - 1.0) / 4.0, 0.0, 1.0)
        for value in simulation.routing_physics.arc_delay_multiplier.values()
    ]
    congestion = (
        clamp(float(route_congestion), 0.0, 1.0)
        if route_congestion is not None
        else _mean(clamp(float(value), 0.0, 1.0) for value in simulation.routing_physics.congestion_by_arc.values())
    )
    return clamp(
        max(
            float(snapshot.get("disruption_score", 0.0)),
            congestion,
            _mean(delay_values),
        ),
        0.0,
        1.0,
    )


def _capacity_pressure(capacity_states: Sequence[Any]) -> float:
    if not capacity_states:
        return 0.0
    return clamp(sum(float(state.capacity_pressure) for state in capacity_states) / len(capacity_states), 0.0, 1.0)


def _dispatch_feasibility_features(
    simulation: DigitalTwinSimulation,
    *,
    context: _ObservationHotPathContext | None = None,
) -> dict[str, float]:
    context = context or _build_hot_path_context(simulation, simulation.snapshot())
    orders = context.orders
    vehicles = context.vehicles
    dispatchable_orders = context.dispatchable_orders
    available_vehicle_count = float(context.available_vehicle_count)
    dispatchable_order_count = float(len(dispatchable_orders))
    feasible_dispatch_ratio = (
        clamp(min(available_vehicle_count, dispatchable_order_count) / max(dispatchable_order_count, 1.0), 0.0, 1.0)
        if dispatchable_order_count > 0.0
        else 0.0
    )
    vehicle_availability_pressure = (
        1.0 - clamp(available_vehicle_count / max(dispatchable_order_count, 1.0), 0.0, 1.0)
        if dispatchable_order_count > 0.0
        else 0.0
    )
    return {
        "normalized_available_vehicle_count": clamp(available_vehicle_count / max(float(len(vehicles)), 1.0), 0.0, 1.0),
        "normalized_dispatchable_order_count": clamp(
            dispatchable_order_count / max(float(len(orders)), 1.0),
            0.0,
            1.0,
        ),
        "feasible_dispatch_opportunity": 1.0
        if available_vehicle_count > 0.0 and dispatchable_order_count > 0.0
        else 0.0,
        "feasible_dispatch_ratio": feasible_dispatch_ratio,
        "vehicle_availability_pressure": vehicle_availability_pressure,
    }


def _real_world_stress_features(stress: Mapping[str, Any]) -> dict[str, float]:
    def value(name: str, default: float) -> float:
        try:
            return float(stress.get(name, default))
        except (TypeError, ValueError):
            return default

    holding_pressure = clamp(
        (
            max(
                value("holding_cost_multiplier", 1.0),
                value("excess_inventory_penalty_multiplier", 1.0),
                value("planned_replenishment_cost_multiplier", 1.0),
            )
            - 1.0
        )
        / 3.0,
        0.0,
        1.0,
    )
    stockout_penalty_pressure = clamp(
        (
            max(
                value("stockout_penalty_multiplier", 1.0),
                value("inventory_shortfall_penalty_multiplier", 1.0),
            )
            - 1.0
        )
        / 3.0,
        0.0,
        1.0,
    )
    capacity_shock_pressure = clamp(
        max(
            1.0 - value("vehicle_availability_multiplier", 1.0),
            1.0 - value("fleet_capacity_multiplier", 1.0),
            value("capacity_shock_severity", 0.0),
        ),
        0.0,
        1.0,
    )
    scenario_route_disruption_pressure = clamp(
        max(
            value("route_disruption_probability", 0.0),
            (value("congestion_multiplier", 1.0) - 1.0) / 3.0,
            (value("shortest_route_risk_multiplier", 1.0) - 1.0) / 3.0,
            (value("disruption_risk_multiplier", 1.0) - 1.0) / 3.0,
        ),
        0.0,
        1.0,
    )
    route_cost_pressure = clamp(
        (
            max(
                value("traversal_cost_multiplier", 1.0),
                value("route_fleet_cost_multiplier", 1.0),
            )
            - 1.0
        )
        / 3.0,
        0.0,
        1.0,
    )
    premium_sla_pressure = clamp(
        max(
            value("premium_sla_ratio", 0.0),
            max(0.0, value("service_target", 0.95) - 0.95) / 0.05,
            max(0.0, 1.0 - value("urgent_due_window_multiplier", 1.0)),
            (value("premium_lateness_penalty_multiplier", 1.0) - 1.0) / 3.0,
        ),
        0.0,
        1.0,
    )
    supplier_delay_pressure = clamp(value("supplier_delay_probability", 0.0), 0.0, 1.0)
    scenario_lead_time_volatility_pressure = clamp(
        max(
            (value("lead_time_mean_multiplier", 1.0) - 1.0) / 3.0,
            (value("lead_time_variance_multiplier", 1.0) - 1.0) / 4.0,
            supplier_delay_pressure,
        ),
        0.0,
        1.0,
    )
    return {
        "normalized_holding_cost_pressure": holding_pressure,
        "normalized_stockout_penalty_pressure": stockout_penalty_pressure,
        "capacity_shock_pressure": capacity_shock_pressure,
        "scenario_route_disruption_pressure": scenario_route_disruption_pressure,
        "route_cost_pressure": route_cost_pressure,
        "premium_sla_pressure": premium_sla_pressure,
        "supplier_delay_pressure": supplier_delay_pressure,
        "scenario_lead_time_volatility_pressure": scenario_lead_time_volatility_pressure,
    }


def _route_candidate_visibility_features(
    simulation: DigitalTwinSimulation,
    snapshot: Mapping[str, Any],
    stress: Mapping[str, Any],
    *,
    dispatch_feasibility_features: Mapping[str, float],
    stress_features: Mapping[str, float],
    context: _ObservationHotPathContext | None = None,
) -> dict[str, float]:
    context = context or _build_hot_path_context(simulation, snapshot)

    def value(name: str, default: float) -> float:
        try:
            return float(stress.get(name, default))
        except (TypeError, ValueError):
            return default

    dispatchable_order_count = float(len(context.dispatchable_orders))
    secondary_available_count = float(context.secondary_available_count)
    secondary_fleet_feasible_dispatch_ratio = (
        clamp(
            min(secondary_available_count, dispatchable_order_count) / max(dispatchable_order_count, 1.0),
            0.0,
            1.0,
        )
        if dispatchable_order_count > 0.0
        else 0.0
    )
    secondary_fleet_vehicle_availability_pressure = (
        1.0 - clamp(secondary_available_count / max(dispatchable_order_count, 1.0), 0.0, 1.0)
        if dispatchable_order_count > 0.0
        else 0.0
    )
    secondary_fleet_dispatch_available = (
        1.0 if secondary_available_count > 0.0 and dispatchable_order_count > 0.0 else 0.0
    )

    feasible_dispatch_ratio = clamp(float(dispatch_feasibility_features.get("feasible_dispatch_ratio", 0.0)), 0.0, 1.0)
    feasible_dispatch_opportunity = clamp(
        float(dispatch_feasibility_features.get("feasible_dispatch_opportunity", 0.0)),
        0.0,
        1.0,
    )
    useful_dispatch_opportunity = clamp(
        feasible_dispatch_opportunity * max(feasible_dispatch_ratio, secondary_fleet_feasible_dispatch_ratio),
        0.0,
        1.0,
    )

    route_congestion = context.route_congestion
    disruption = clamp(float(snapshot.get("disruption_score", 0.0)), 0.0, 1.0)
    scenario_congestion_pressure = clamp((value("congestion_multiplier", 1.0) - 1.0) / 3.0, 0.0, 1.0)
    route_cost_pressure = clamp(float(stress_features.get("route_cost_pressure", 0.0)), 0.0, 1.0)
    scenario_route_disruption_pressure = clamp(
        float(stress_features.get("scenario_route_disruption_pressure", 0.0)),
        0.0,
        1.0,
    )
    route_disruption_reliability_pressure = clamp(
        max(
            disruption,
            value("route_disruption_probability", 0.0),
            (value("shortest_route_risk_multiplier", 1.0) - 1.0) / 3.0,
            (value("disruption_risk_multiplier", 1.0) - 1.0) / 3.0,
        ),
        0.0,
        1.0,
    )
    route_congestion_cost_pressure = clamp(
        max(route_congestion, scenario_congestion_pressure, route_cost_pressure),
        0.0,
        1.0,
    )
    route_pressure_total = route_disruption_reliability_pressure + route_congestion_cost_pressure
    if route_pressure_total > 1e-9:
        route_pressure_reliability_share = route_disruption_reliability_pressure / route_pressure_total
        route_pressure_congestion_share = route_congestion_cost_pressure / route_pressure_total
        route_pressure_balance = min(
            route_disruption_reliability_pressure,
            route_congestion_cost_pressure,
        ) / max(route_disruption_reliability_pressure, route_congestion_cost_pressure, 1e-9)
    else:
        route_pressure_reliability_share = 0.0
        route_pressure_congestion_share = 0.0
        route_pressure_balance = 0.0
    route_alt_pressure_imbalance = abs(route_pressure_congestion_share - route_pressure_reliability_share)
    route_adaptation_pressure = max(
        disruption,
        route_congestion_cost_pressure,
        route_disruption_reliability_pressure,
        route_cost_pressure,
    )

    observed_deliveries = float(getattr(simulation.state, "delivered_orders", 0))
    service_level = clamp(float(snapshot.get("service_level", 1.0)), 0.0, 1.0)
    visibility_service_level = service_level if observed_deliveries > 0.0 else 1.0
    service_target = clamp(value("service_target", 0.95), 0.50, 1.0)
    service_health_factor = _smoothstep_unit((visibility_service_level - (service_target - 0.05)) / 0.08)
    true_lateness_pressure = _pending_lateness_pressure(context.orders, current_time=context.current_time)
    operational_health_factor = clamp(visibility_service_level * (1.0 - true_lateness_pressure), 0.0, 1.0)
    route_candidate_service_factor = min(service_health_factor, operational_health_factor)
    route_candidate_execution_factor = route_candidate_service_factor * useful_dispatch_opportunity

    max_route_pressure = max(route_disruption_reliability_pressure, route_congestion_cost_pressure)
    shortest_candidate_moderate_pressure_support = _smoothstep_unit((0.45 - max_route_pressure) / 0.20)
    shortest_candidate_congestion_support = (
        _smoothstep_unit((route_congestion_cost_pressure - 0.34) / 0.06)
        * _smoothstep_unit((0.50 - route_congestion_cost_pressure) / 0.08)
    )
    shortest_candidate_reliability_support = (
        _smoothstep_unit((route_disruption_reliability_pressure - 0.16) / 0.06)
        * _smoothstep_unit((0.36 - route_disruption_reliability_pressure) / 0.10)
    )
    shortest_candidate_balance_support = _smoothstep_unit((route_pressure_balance - 0.45) / 0.20)
    shortest_candidate_observed_profile_support = min(
        shortest_candidate_congestion_support,
        shortest_candidate_reliability_support,
        shortest_candidate_balance_support,
    )
    shortest_candidate_pressure_support = max(
        shortest_candidate_moderate_pressure_support,
        shortest_candidate_observed_profile_support,
    )
    high_resilience_need = _smoothstep_unit(
        (max(disruption, route_disruption_reliability_pressure, 0.50 * route_cost_pressure) - 0.15) / 0.45
    )
    low_congestion_need = _smoothstep_unit((route_congestion_cost_pressure - 0.05) / 0.55)
    high_resilience_candidate_reliability_support = _smoothstep_unit(
        (route_disruption_reliability_pressure - 0.12) / 0.28
    )
    if route_adaptation_pressure > 0.0:
        shortest_candidate_score = clamp(shortest_candidate_pressure_support * route_candidate_execution_factor, 0.0, 1.0)
        low_congestion_candidate_score = clamp(
            low_congestion_need
            * (0.70 + (0.30 * route_pressure_congestion_share))
            * (0.75 + (0.25 * route_pressure_balance))
            * route_candidate_execution_factor,
            0.0,
            1.0,
        )
        high_resilience_candidate_score = clamp(
            max(high_resilience_need, high_resilience_candidate_reliability_support)
            * (0.70 + (0.30 * route_pressure_reliability_share))
            * (0.70 + (0.30 * route_pressure_balance))
            * route_candidate_execution_factor,
            0.0,
            1.0,
        )
    else:
        shortest_candidate_score = 0.0
        low_congestion_candidate_score = 0.0
        high_resilience_candidate_score = 0.0

    best_route_candidate_score = max(
        shortest_candidate_score,
        low_congestion_candidate_score,
        high_resilience_candidate_score,
    )
    shortest_candidate_score_gap = max(best_route_candidate_score - shortest_candidate_score, 0.0)
    low_congestion_candidate_score_gap = max(best_route_candidate_score - low_congestion_candidate_score, 0.0)
    high_resilience_candidate_score_gap = max(best_route_candidate_score - high_resilience_candidate_score, 0.0)
    shortest_candidate_near_best = (
        1.0 if shortest_candidate_score > 0.0 and shortest_candidate_score_gap <= 0.30 else 0.0
    )

    severe_pressure = clamp(
        max(
            _smoothstep_unit((max_route_pressure - 0.55) / 0.15),
            _smoothstep_unit((route_disruption_reliability_pressure - 0.45) / 0.15),
            _smoothstep_unit((scenario_route_disruption_pressure - 0.55) / 0.15),
            1.0 if route_adaptation_pressure > 0.0 and shortest_candidate_pressure_support <= 0.0 else 0.0,
        ),
        0.0,
        1.0,
    )
    moderate_pressure = (
        _smoothstep_unit((0.50 - max_route_pressure) / 0.10)
        * _smoothstep_unit((0.35 - route_disruption_reliability_pressure) / 0.10)
        * (1.0 - severe_pressure)
    )
    shortest_secondary_safe_context = clamp(
        shortest_candidate_near_best
        * useful_dispatch_opportunity
        * route_candidate_service_factor
        * secondary_fleet_dispatch_available
        * moderate_pressure,
        0.0,
        1.0,
    )
    weak_useful_work = 1.0 - useful_dispatch_opportunity
    poor_service = 1.0 - route_candidate_service_factor
    shortest_secondary_brittle_risk = clamp(
        max(
            severe_pressure,
            weak_useful_work,
            poor_service,
            secondary_fleet_vehicle_availability_pressure,
        )
        * (1.0 - shortest_secondary_safe_context),
        0.0,
        1.0,
    )

    return {
        "shortest_candidate_score": shortest_candidate_score,
        "low_congestion_candidate_score": low_congestion_candidate_score,
        "high_resilience_candidate_score": high_resilience_candidate_score,
        "shortest_candidate_score_gap": shortest_candidate_score_gap,
        "low_congestion_candidate_score_gap": low_congestion_candidate_score_gap,
        "high_resilience_candidate_score_gap": high_resilience_candidate_score_gap,
        "route_pressure_reliability_share": route_pressure_reliability_share,
        "route_pressure_congestion_share": route_pressure_congestion_share,
        "route_pressure_balance": route_pressure_balance,
        "route_alt_pressure_imbalance": route_alt_pressure_imbalance,
        "shortest_candidate_near_best": shortest_candidate_near_best,
        "shortest_secondary_safe_context": shortest_secondary_safe_context,
        "shortest_secondary_brittle_risk": shortest_secondary_brittle_risk,
        "secondary_fleet_feasible_dispatch_ratio": secondary_fleet_feasible_dispatch_ratio,
        "secondary_fleet_vehicle_availability_pressure": secondary_fleet_vehicle_availability_pressure,
        "useful_dispatch_opportunity": useful_dispatch_opportunity,
    }


def _tier_value(vehicle: Any) -> str:
    tier = getattr(vehicle, "tier", "")
    return str(getattr(tier, "value", tier))


def _pending_lateness_pressure(orders: Sequence[Any], *, current_time: float) -> float:
    pressures: list[float] = []
    for order in orders:
        if getattr(order, "delivery_time", None) is not None:
            continue
        due_time = float(getattr(order, "due_time", current_time))
        release_time = float(getattr(order, "release_time", current_time))
        window = max(due_time - release_time, 1.0)
        pressures.append(clamp((current_time - due_time) / window, 0.0, 1.0))
    return _mean(pressures)


def _smoothstep_unit(value: float) -> float:
    x = clamp(float(value), 0.0, 1.0)
    return x * x * (3.0 - (2.0 * x))


def _mean(values: Sequence[float] | Any) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        return 0.0
    return clamp(sum(materialized) / len(materialized), 0.0, 1.0)


def _is_dispatchable_order(order: Any) -> bool:
    status = getattr(order, "status", None)
    status_value = getattr(status, "value", status)
    return (
        getattr(order, "assigned_vehicle_id", None) is None
        and getattr(order, "delivery_time", None) is None
        and str(status_value) != "delayed"
    )
