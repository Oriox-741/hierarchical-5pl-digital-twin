"""Scenario override injection for evaluation-only stress tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv
from src.shared.metrics import clamp


OverrideStatus = Literal["APPLIED", "PARTIAL", "UNSUPPORTED", "UNKNOWN"]


@dataclass(frozen=True, slots=True)
class OverrideCapability:
    key: str
    status: OverrideStatus
    explanation: str

    def as_dict(self) -> dict[str, str]:
        return {"key": self.key, "status": self.status, "explanation": self.explanation}


CONFIG_MULTIPLIER_KEYS = {
    "rolling_demand_probability_multiplier",
    "rolling_demand_volume_multiplier",
    "order_units_multiplier",
    "urgent_order_probability_multiplier",
    "holding_cost_multiplier",
    "excess_inventory_penalty_multiplier",
    "planned_replenishment_cost_multiplier",
    "stockout_penalty_multiplier",
    "inventory_shortfall_penalty_multiplier",
    "emergency_procurement_cost_multiplier",
    "premium_sla_ratio",
    "urgent_due_window_multiplier",
    "service_target",
    "premium_lateness_penalty_multiplier",
    "route_fleet_cost_multiplier",
}

ENV_INSTANCE_KEYS = {
    "supplier_delay_probability",
    "lead_time_mean_multiplier",
    "lead_time_variance_multiplier",
    "vehicle_availability_multiplier",
    "fleet_capacity_multiplier",
    "capacity_shock_severity",
    "route_disruption_probability",
    "congestion_multiplier",
    "shortest_route_risk_multiplier",
    "traversal_cost_multiplier",
    "disruption_risk_multiplier",
    "clustered_spike_windows",
}

DIRECT_ENV_FIELDS = set(EnvironmentConfig.__dataclass_fields__)
SUPPORTED_OVERRIDE_KEYS = CONFIG_MULTIPLIER_KEYS | ENV_INSTANCE_KEYS | DIRECT_ENV_FIELDS


def apply_config_overrides(config: EnvironmentConfig, overrides: Mapping[str, Any]) -> dict[str, OverrideCapability]:
    """Apply supported pre-environment overrides to an EnvironmentConfig instance."""

    matrix: dict[str, OverrideCapability] = {}
    for key, value in overrides.items():
        if key in DIRECT_ENV_FIELDS:
            setattr(config, key, value)
            matrix[key] = OverrideCapability(key, "APPLIED", "direct EnvironmentConfig field applied")
        elif key == "rolling_demand_probability_multiplier":
            config.rolling_demand_probability = clamp(config.rolling_demand_probability * _positive(value), 0.0, 1.0)
            matrix[key] = OverrideCapability(key, "APPLIED", "multiplied rolling_demand_probability")
        elif key == "rolling_demand_volume_multiplier":
            multiplier = _positive(value)
            config.rolling_demand_max_orders_per_step = max(
                1,
                int(round(config.rolling_demand_max_orders_per_step * multiplier)),
            )
            matrix[key] = OverrideCapability(key, "APPLIED", "multiplied rolling_demand_max_orders_per_step")
        elif key == "order_units_multiplier":
            multiplier = _positive(value)
            config.rolling_demand_min_units *= multiplier
            config.rolling_demand_max_units *= multiplier
            matrix[key] = OverrideCapability(key, "APPLIED", "multiplied rolling demand order unit bounds")
        elif key == "urgent_order_probability_multiplier":
            config.urgent_order_probability = clamp(config.urgent_order_probability * _positive(value), 0.0, 1.0)
            matrix[key] = OverrideCapability(key, "APPLIED", "multiplied urgent_order_probability")
        elif key == "holding_cost_multiplier":
            multiplier = _positive(value)
            config.excess_inventory_penalty_weight *= multiplier
            config.planned_replenishment_cost_weight *= multiplier
            config.planned_replenishment_unit_cost *= multiplier
            matrix[key] = OverrideCapability(key, "APPLIED", "scaled excess inventory and planned replenishment costs")
        elif key == "excess_inventory_penalty_multiplier":
            config.excess_inventory_penalty_weight *= _positive(value)
            matrix[key] = OverrideCapability(key, "APPLIED", "scaled excess_inventory_penalty_weight")
        elif key == "planned_replenishment_cost_multiplier":
            multiplier = _positive(value)
            config.planned_replenishment_cost_weight *= multiplier
            config.planned_replenishment_unit_cost *= multiplier
            matrix[key] = OverrideCapability(key, "APPLIED", "scaled planned replenishment cost weight and unit cost")
        elif key == "stockout_penalty_multiplier":
            config.stockout_penalty_multiplier *= _nonnegative(value)
            matrix[key] = OverrideCapability(key, "APPLIED", "scaled stockout_risk used by reward economics")
        elif key == "inventory_shortfall_penalty_multiplier":
            config.inventory_shortfall_penalty_multiplier *= _nonnegative(value)
            matrix[key] = OverrideCapability(key, "APPLIED", "scaled inventory_shortfall used by reward economics")
        elif key == "emergency_procurement_cost_multiplier":
            multiplier = _positive(value)
            config.emergency_replenishment_unit_cost *= multiplier
            config.emergency_cost_weight *= multiplier
            matrix[key] = OverrideCapability(key, "APPLIED", "scaled emergency procurement unit cost and cost weight")
        elif key == "premium_sla_ratio":
            ratio = clamp(_nonnegative(value), 0.0, 1.0)
            config.urgent_order_probability = max(config.urgent_order_probability, ratio)
            matrix[key] = OverrideCapability(key, "APPLIED", "mapped premium SLA ratio to urgent order probability")
        elif key == "urgent_due_window_multiplier":
            config.urgent_due_window_multiplier = _positive(value)
            matrix[key] = OverrideCapability(key, "APPLIED", "scaled urgent rolling-demand due windows")
        elif key == "service_target":
            config.service_level_target = clamp(_positive(value), 0.0, 1.0)
            matrix[key] = OverrideCapability(key, "APPLIED", "set service_level_target")
        elif key == "premium_lateness_penalty_multiplier":
            config.delay_weight *= _positive(value)
            matrix[key] = OverrideCapability(key, "APPLIED", "scaled delay_weight for lateness pressure")
        elif key == "route_fleet_cost_multiplier":
            multiplier = _positive(value)
            config.primary_fleet_penalty *= multiplier
            config.high_resilience_route_penalty *= multiplier
            config.low_congestion_route_penalty *= multiplier
            matrix[key] = OverrideCapability(key, "APPLIED", "scaled route/fleet cost penalties")
        elif key in ENV_INSTANCE_KEYS:
            matrix[key] = OverrideCapability(key, "APPLIED", "applied after fresh evaluation environment construction")
        else:
            matrix[key] = OverrideCapability(key, "UNSUPPORTED", "unknown scenario override key")

    if "clustered_spike_windows" in overrides:
        matrix["clustered_spike_windows"] = OverrideCapability(
            "clustered_spike_windows",
            "PARTIAL",
            "not a time-window generator yet; demand multipliers provide aggregate spike stress",
        )
    return matrix


def apply_environment_overrides(
    env: FivePLDigitalTwinEnv,
    overrides: Mapping[str, Any],
    matrix: Mapping[str, OverrideCapability] | None = None,
) -> dict[str, OverrideCapability]:
    """Apply supported post-construction overrides to one fresh eval environment."""

    result = dict(matrix or {})
    _record_observation_stress(env, overrides)
    for key, value in overrides.items():
        if key == "supplier_delay_probability":
            probability = clamp(_nonnegative(value), 0.0, 1.0)
            multiplier = 1.0 + probability
            for position in env.simulation.inventory_network.positions.values():
                position.lead_time_mean *= multiplier
                position.lead_time_variance *= 1.0 + (2.0 * probability)
                position.metadata["supplier_delay_probability"] = probability
            result[key] = OverrideCapability(key, "APPLIED", "increased inventory lead-time mean/variance")
        elif key == "lead_time_mean_multiplier":
            multiplier = _positive(value)
            for position in env.simulation.inventory_network.positions.values():
                position.lead_time_mean *= multiplier
            result[key] = OverrideCapability(key, "APPLIED", "multiplied inventory lead_time_mean")
        elif key == "lead_time_variance_multiplier":
            multiplier = _positive(value)
            for position in env.simulation.inventory_network.positions.values():
                position.lead_time_variance *= multiplier
            result[key] = OverrideCapability(key, "APPLIED", "multiplied inventory lead_time_variance")
        elif key == "vehicle_availability_multiplier":
            multiplier = clamp(_nonnegative(value), 0.0, 1.0)
            vehicles = list(env.simulation.vehicles.values())
            active_count = max(1, int(round(len(vehicles) * multiplier))) if vehicles else 0
            for index, vehicle in enumerate(vehicles):
                vehicle.active = index < active_count
                vehicle.metadata["scenario_vehicle_availability_multiplier"] = multiplier
            result[key] = OverrideCapability(key, "APPLIED", "disabled a deterministic portion of vehicles")
        elif key == "fleet_capacity_multiplier":
            multiplier = _positive(value)
            for vehicle in env.simulation.vehicles.values():
                vehicle.capacity_units *= multiplier
                vehicle.capacity_weight_kg *= multiplier
                vehicle.capacity_volume_m3 *= multiplier
                vehicle.metadata["scenario_fleet_capacity_multiplier"] = multiplier
            for state in env.simulation.capacity_states.values():
                state.base_capacity *= multiplier
            result[key] = OverrideCapability(key, "APPLIED", "scaled vehicle and capacity-state capacity")
        elif key == "capacity_shock_severity":
            severity = clamp(_nonnegative(value), 0.0, 1.0)
            multiplier = max(0.05, 1.0 - severity)
            for state in env.simulation.capacity_states.values():
                state.base_capacity *= multiplier
                state.multiplier = min(state.multiplier, multiplier)
            result[key] = OverrideCapability(key, "APPLIED", "reduced capacity-state nominal capacity")
        elif key == "route_disruption_probability":
            probability = clamp(_nonnegative(value), 0.0, 1.0)
            for arc_key in env.simulation.route_network.arcs:
                env.simulation.routing_physics.set_arc_delay_multiplier(*arc_key, 1.0 + probability)
            result[key] = OverrideCapability(key, "APPLIED", "increased arc delay multipliers")
        elif key == "congestion_multiplier":
            multiplier = _positive(value)
            score = clamp(0.25 * multiplier, 0.0, 1.0)
            for arc_key in env.simulation.route_network.arcs:
                env.simulation.routing_physics.set_congestion(*arc_key, score)
            result[key] = OverrideCapability(key, "APPLIED", "set per-arc congestion score")
        elif key == "shortest_route_risk_multiplier":
            env.simulation.routing_physics.disruption_risk_multiplier *= _positive(value)
            result[key] = OverrideCapability(key, "APPLIED", "scaled routing physics disruption risk multiplier")
        elif key == "traversal_cost_multiplier":
            env.simulation.routing_physics.traversal_cost_multiplier *= _positive(value)
            result[key] = OverrideCapability(key, "APPLIED", "scaled route traversal costs")
        elif key == "disruption_risk_multiplier":
            env.simulation.routing_physics.disruption_risk_multiplier *= _positive(value)
            result[key] = OverrideCapability(key, "APPLIED", "scaled route speed-risk economics")
    return result


def _record_observation_stress(env: FivePLDigitalTwinEnv, overrides: Mapping[str, Any]) -> None:
    stress = getattr(env, "scenario_stress_state", None)
    if not isinstance(stress, dict):
        return
    for key in (
        "holding_cost_multiplier",
        "excess_inventory_penalty_multiplier",
        "planned_replenishment_cost_multiplier",
        "stockout_penalty_multiplier",
        "inventory_shortfall_penalty_multiplier",
        "vehicle_availability_multiplier",
        "fleet_capacity_multiplier",
        "capacity_shock_severity",
        "route_disruption_probability",
        "congestion_multiplier",
        "shortest_route_risk_multiplier",
        "traversal_cost_multiplier",
        "disruption_risk_multiplier",
        "route_fleet_cost_multiplier",
        "premium_sla_ratio",
        "service_target",
        "urgent_due_window_multiplier",
        "premium_lateness_penalty_multiplier",
        "supplier_delay_probability",
        "lead_time_mean_multiplier",
        "lead_time_variance_multiplier",
    ):
        if key in overrides:
            stress[key] = float(overrides[key])


def capability_matrix_as_dict(matrix: Mapping[str, OverrideCapability]) -> list[dict[str, str]]:
    return [matrix[key].as_dict() for key in sorted(matrix)]


def _positive(value: Any) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError("override value must be positive")
    return parsed


def _nonnegative(value: Any) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise ValueError("override value must not be negative")
    return parsed
