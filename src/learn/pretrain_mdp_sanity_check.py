"""Deterministic non-training sanity checks for the physical-reality MDP contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.act.discrete_action_mapper import (
    DISCRETE_ACTION_COUNT,
    DispatchDecision,
    DiscreteLogisticsAction,
    ModeDecision,
    ReorderDecision,
    RouteDecision,
)
from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv
from src.act.observation_builder import OBSERVATION_DIM, OBSERVATION_FEATURE_INDEX
from src.learn.joint_buffers import dqn_training_reward
from src.learn.train_joint_torch import MDP_CONTRACT_VERSION
from src.shared.types import ShipmentStatus


@dataclass(frozen=True, slots=True)
class SanityCheckResult:
    name: str
    passed: bool
    details: dict[str, float]


def run_all_checks() -> list[SanityCheckResult]:
    """Run pre-training physical/economic checks without creating policies or gradients."""
    checks = [
        _check_speed_is_not_free_in_calm_conditions(),
        _check_speed_is_not_universally_better(),
        _check_speed_credit_requires_actual_movement(),
        _check_planned_reorder_helps_under_stockout_risk(),
        _check_planned_reorder_cost_is_not_free(),
        _check_planned_reorder_hurts_under_overstock(),
        _check_safety_stock_helps_under_demand_volatility(),
        _check_ppo_inventory_neglect_is_penalized_under_forecast_risk(),
        _check_ppo_neglect_penalties_skip_no_risk_inactivity(),
        _check_ppo_capacity_neglect_requires_capacity_pressure(),
        _check_low_speed_penalty_requires_flow_pressure(),
        _check_low_speed_is_penalized_under_lateness_pressure(),
        _check_capacity_buffer_neglect_requires_flow_pressure(),
        _check_capacity_buffer_is_rewarded_under_pending_work_pressure(),
        _check_order_flow_metrics_split_backlog_from_in_transit(),
        _check_order_flow_telemetry_preserves_raw_pressure_while_actionable_pressure_drops(),
        _check_ppo_dispatch_intensity_sets_budget_not_direct_dispatch(),
        _check_global_flow_credit_requires_real_pressure_reduction(),
        _check_ppo_flow_enablement_credit_requires_actionable_pressure(),
        _check_dqn_dispatch_progress_credit_requires_successful_dispatch(),
        _check_repeated_infeasible_dispatch_penalty_targets_spam(),
        _check_emergency_reorder_hurts_under_low_inventory_risk(),
        _check_emergency_gate_blocks_low_risk_credit(),
        _check_medium_risk_emergency_is_still_costly(),
        _check_emergency_credit_requires_useful_override_units(),
        _check_dqn_negative_local_is_not_masked_by_global_reward(),
        _check_hold_has_no_penalty_without_pre_crisis_pressure(),
        _check_hold_penalty_rises_with_pre_crisis_pressure(),
        _check_delay_abuse_penalty_reduces_late_emergency_credit(),
        _check_global_reward_caps_no_delivery_under_pending_pressure(),
        _check_global_reward_does_not_penalize_calm_no_work(),
        _check_hold_route_labels_are_neutral(),
        _check_action8_hold_low_congestion_has_no_route_credit_or_inflight_delivery_lift(),
        _check_high_resilience_route_is_not_free_in_calm_dispatch_conditions(),
        _check_action_observation_contract_is_v5_route_candidate_visibility(),
        _check_dispatch_feasibility_observation_features_are_bounded(),
        _check_real_world_stress_observation_features_are_bounded(),
        _check_route_candidate_visibility_observation_features_are_bounded(),
        _check_observations_are_finite_and_normalized(),
    ]
    return checks


def assert_all_checks_pass() -> None:
    failures = [result for result in run_all_checks() if not result.passed]
    if failures:
        formatted = "\n".join(f"- {failure.name}: {failure.details}" for failure in failures)
        raise AssertionError(f"pre-training MDP sanity check failed:\n{formatted}")


def main() -> int:
    results = run_all_checks()
    print("# Pre-Training MDP Sanity Check")
    print("| Check | Status | Details |")
    print("|---|---:|---|")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        details = ", ".join(f"{key}={value:.6f}" for key, value in result.details.items())
        print(f"| {result.name} | {status} | {details} |")
    if any(not result.passed for result in results):
        return 1
    return 0


def _check_speed_is_not_free_in_calm_conditions() -> SanityCheckResult:
    env = _env()
    try:
        _set_calm_inventory(env)
        snapshot = env.simulation.snapshot()
        nominal = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
            speed_cost_delta=0.0,
            distance_cost_delta=100.0,
        )["ppo_local"]
        max_speed = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.35, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
            speed_cost_delta=50.0,
            distance_cost_delta=100.0,
        )["ppo_local"]
        return SanityCheckResult(
            name="max_speed_costs_more_than_nominal_in_calm_conditions",
            passed=max_speed < nominal,
            details={"nominal": nominal, "max_speed": max_speed},
        )
    finally:
        env.close()


def _check_speed_is_not_universally_better() -> SanityCheckResult:
    env = _env()
    try:
        _set_overstock_low_demand(env)
        snapshot = env.simulation.snapshot()
        nominal = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
            speed_cost_delta=0.0,
            distance_cost_delta=100.0,
        )["ppo_local"]
        max_speed = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.35, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
            speed_cost_delta=50.0,
            distance_cost_delta=100.0,
        )["ppo_local"]
        return SanityCheckResult(
            name="max_speed_is_not_always_better_after_reward_normalization",
            passed=max_speed < nominal,
            details={"nominal": nominal, "max_speed": max_speed},
        )
    finally:
        env.close()


def _check_speed_credit_requires_actual_movement() -> SanityCheckResult:
    env = _env()
    try:
        _set_stockout_risk(env)
        _make_orders_urgent(env)
        snapshot = env.simulation.snapshot()
        idle_speed = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.35, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
            speed_cost_delta=0.0,
            distance_cost_delta=0.0,
        )
        moving_speed = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.35, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
            speed_cost_delta=50.0,
            distance_cost_delta=100.0,
        )
        return SanityCheckResult(
            name="speed_credit_requires_actual_route_movement",
            passed=idle_speed["speed_justification_credit"] == 0.0
            and idle_speed["no_movement_speed_penalty"] > 0.0
            and moving_speed["speed_justification_credit"] > idle_speed["speed_justification_credit"],
            details={
                "idle_credit": idle_speed["speed_justification_credit"],
                "idle_no_movement_penalty": idle_speed["no_movement_speed_penalty"],
                "moving_credit": moving_speed["speed_justification_credit"],
            },
        )
    finally:
        env.close()


def _check_planned_reorder_helps_under_stockout_risk() -> SanityCheckResult:
    env = _env()
    try:
        _set_stockout_risk(env)
        snapshot = env.simulation.snapshot()
        no_reorder = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
        )["ppo_local"]
        planned_reorder = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.8, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
            planned_units=100.0,
            useful_units=100.0,
            planned_cost=5.0,
        )["ppo_local"]
        return SanityCheckResult(
            name="planned_reorder_helps_under_stockout_risk",
            passed=planned_reorder > no_reorder,
            details={"no_reorder": no_reorder, "planned_reorder": planned_reorder},
        )
    finally:
        env.close()


def _check_planned_reorder_cost_is_not_free() -> SanityCheckResult:
    env = _env()
    try:
        _set_stockout_risk(env)
        snapshot = env.simulation.snapshot()
        cheap_reorder = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.8, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
            planned_units=100.0,
            useful_units=100.0,
            planned_cost=5.0,
        )["ppo_local"]
        costly_reorder = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.8, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
            planned_units=100.0,
            useful_units=100.0,
            planned_cost=80.0,
        )["ppo_local"]
        return SanityCheckResult(
            name="planned_reorder_cost_is_not_free",
            passed=costly_reorder < cheap_reorder,
            details={"cheap_reorder": cheap_reorder, "costly_reorder": costly_reorder},
        )
    finally:
        env.close()


def _check_planned_reorder_hurts_under_overstock() -> SanityCheckResult:
    env = _env()
    try:
        _set_overstock_low_demand(env)
        snapshot = env.simulation.snapshot()
        no_reorder = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
        )["ppo_local"]
        planned_reorder = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.8, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
            planned_units=100.0,
            useful_units=0.0,
            planned_cost=5.0,
        )["ppo_local"]
        return SanityCheckResult(
            name="planned_reorder_hurts_under_overstock",
            passed=planned_reorder < no_reorder,
            details={"no_reorder": no_reorder, "planned_reorder": planned_reorder},
        )
    finally:
        env.close()


def _check_safety_stock_helps_under_demand_volatility() -> SanityCheckResult:
    env = _env()
    try:
        _set_stockout_risk(env)
        for position in env.simulation.inventory_network.positions.values():
            position.demand_variance = max(position.demand_variance, position.demand_mean * 8.0)
            position.safety_stock_units = 0.0
        snapshot = env.simulation.snapshot()
        base_stock = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
        )["ppo_local"]
        high_stock = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.0, "safety_stock_multiplier": 2.0, "capacity_buffer_fraction": 0.0},
        )["ppo_local"]
        return SanityCheckResult(
            name="safety_stock_helps_under_demand_volatility",
            passed=high_stock > base_stock,
            details={"base_stock": base_stock, "high_stock": high_stock},
        )
    finally:
        env.close()


def _check_ppo_inventory_neglect_is_penalized_under_forecast_risk() -> SanityCheckResult:
    env = _env()
    try:
        _set_stockout_risk(env)
        snapshot = env.simulation.snapshot()
        negligent = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
        )
        proactive = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 1.0, "safety_stock_multiplier": 2.0, "capacity_buffer_fraction": 0.0},
            planned_units=100.0,
            useful_units=100.0,
            planned_cost=5.0,
        )
        return SanityCheckResult(
            name="ppo_inventory_neglect_is_penalized_under_forecast_risk",
            passed=negligent["forecast_inventory_risk"] > 0.0
            and negligent["reorder_neglect_penalty"] > 0.0
            and negligent["safety_stock_neglect_penalty"] > 0.0
            and proactive["reorder_neglect_penalty"] == 0.0
            and proactive["safety_stock_neglect_penalty"] == 0.0
            and proactive["ppo_local"] > negligent["ppo_local"],
            details={
                "forecast_inventory_risk": negligent["forecast_inventory_risk"],
                "reorder_neglect_penalty": negligent["reorder_neglect_penalty"],
                "safety_stock_neglect_penalty": negligent["safety_stock_neglect_penalty"],
            },
        )
    finally:
        env.close()


def _check_ppo_neglect_penalties_skip_no_risk_inactivity() -> SanityCheckResult:
    env = _env()
    try:
        _set_no_forecast_risk(env)
        _make_orders_delivered(env)
        snapshot = env.simulation.snapshot()
        components = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
        )
        return SanityCheckResult(
            name="ppo_neglect_penalties_skip_no_risk_inactivity",
            passed=components["forecast_inventory_risk"] == 0.0
            and components["capacity_pressure"] == 0.0
            and components["reorder_neglect_penalty"] == 0.0
            and components["safety_stock_neglect_penalty"] == 0.0
            and components["capacity_neglect_penalty"] == 0.0
            and components["low_speed_lateness_penalty"] == 0.0
            and components["flow_capacity_neglect_penalty"] == 0.0,
            details={
                "forecast_inventory_risk": components["forecast_inventory_risk"],
                "capacity_pressure": components["capacity_pressure"],
                "flow_pressure": components["flow_pressure"],
                "reorder_neglect_penalty": components["reorder_neglect_penalty"],
                "safety_stock_neglect_penalty": components["safety_stock_neglect_penalty"],
                "capacity_neglect_penalty": components["capacity_neglect_penalty"],
                "low_speed_lateness_penalty": components["low_speed_lateness_penalty"],
                "flow_capacity_neglect_penalty": components["flow_capacity_neglect_penalty"],
            },
        )
    finally:
        env.close()


def _check_ppo_capacity_neglect_requires_capacity_pressure() -> SanityCheckResult:
    env = _env()
    try:
        for state in env.simulation.capacity_states.values():
            state.reduce(0.75)
        snapshot = env.simulation.snapshot()
        low_buffer = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
        )
        max_buffer = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.5},
        )
        return SanityCheckResult(
            name="ppo_capacity_neglect_requires_capacity_pressure",
            passed=low_buffer["capacity_pressure"] > 0.0
            and low_buffer["capacity_neglect_penalty"] > 0.0
            and max_buffer["capacity_neglect_penalty"] == 0.0,
            details={
                "capacity_pressure": low_buffer["capacity_pressure"],
                "low_buffer_penalty": low_buffer["capacity_neglect_penalty"],
                "max_buffer_penalty": max_buffer["capacity_neglect_penalty"],
            },
        )
    finally:
        env.close()


def _check_low_speed_penalty_requires_flow_pressure() -> SanityCheckResult:
    env = _env()
    try:
        _set_no_forecast_risk(env)
        _make_orders_delivered(env)
        snapshot = env.simulation.snapshot()
        components = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 0.65, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
        )
        return SanityCheckResult(
            name="low_speed_penalty_requires_flow_pressure",
            passed=components["flow_pressure"] == 0.0 and components["low_speed_lateness_penalty"] == 0.0,
            details={
                "flow_pressure": components["flow_pressure"],
                "low_speed_lateness_penalty": components["low_speed_lateness_penalty"],
            },
        )
    finally:
        env.close()


def _check_low_speed_is_penalized_under_lateness_pressure() -> SanityCheckResult:
    env = _env()
    try:
        _set_no_forecast_risk(env)
        _make_orders_urgent_and_aged(env)
        snapshot = env.simulation.snapshot()
        low_speed = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 0.65, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
        )
        nominal_speed = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
        )
        return SanityCheckResult(
            name="low_speed_is_penalized_under_lateness_pressure",
            passed=low_speed["flow_pressure"] > 0.0
            and low_speed["low_speed_lateness_penalty"] > 0.0
            and nominal_speed["low_speed_lateness_penalty"] == 0.0
            and nominal_speed["ppo_local"] > low_speed["ppo_local"],
            details={
                "flow_pressure": low_speed["flow_pressure"],
                "low_speed_penalty": low_speed["low_speed_lateness_penalty"],
                "nominal_speed_penalty": nominal_speed["low_speed_lateness_penalty"],
            },
        )
    finally:
        env.close()


def _check_capacity_buffer_neglect_requires_flow_pressure() -> SanityCheckResult:
    env = _env()
    try:
        _set_no_forecast_risk(env)
        _make_orders_delivered(env)
        snapshot = env.simulation.snapshot()
        components = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
        )
        return SanityCheckResult(
            name="capacity_buffer_neglect_requires_flow_pressure",
            passed=components["flow_pressure"] == 0.0 and components["flow_capacity_neglect_penalty"] == 0.0,
            details={
                "flow_pressure": components["flow_pressure"],
                "flow_capacity_neglect_penalty": components["flow_capacity_neglect_penalty"],
            },
        )
    finally:
        env.close()


def _check_capacity_buffer_is_rewarded_under_pending_work_pressure() -> SanityCheckResult:
    env = _env()
    try:
        _set_no_forecast_risk(env)
        _make_orders_urgent_and_aged(env)
        snapshot = env.simulation.snapshot()
        low_buffer = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.0},
        )
        high_buffer = _ppo_components(
            env,
            snapshot,
            continuous_action={"speed_multiplier": 1.0, "reorder_fraction": 0.0, "safety_stock_multiplier": 1.0, "capacity_buffer_fraction": 0.5},
        )
        return SanityCheckResult(
            name="capacity_buffer_is_rewarded_under_pending_work_pressure",
            passed=low_buffer["flow_pressure"] > 0.0
            and low_buffer["flow_capacity_neglect_penalty"] > 0.0
            and high_buffer["flow_capacity_neglect_penalty"] == 0.0
            and high_buffer["ppo_local"] > low_buffer["ppo_local"],
            details={
                "flow_pressure": low_buffer["flow_pressure"],
                "low_buffer_penalty": low_buffer["flow_capacity_neglect_penalty"],
                "max_buffer_penalty": high_buffer["flow_capacity_neglect_penalty"],
            },
        )
    finally:
        env.close()


def _check_order_flow_metrics_split_backlog_from_in_transit() -> SanityCheckResult:
    env = _env()
    try:
        snapshot = env.simulation.snapshot()
        now = float(snapshot["time"])
        for index, order in enumerate(snapshot["orders"].values()):
            if isinstance(order, dict):
                order["status"] = "in_transit" if index == 0 else "created"
                order["delivery_time"] = None
                order["assigned_vehicle_id"] = "vehicle-1" if index == 0 else None
                order["due_time"] = now + 86_400.0
        metrics = env._order_flow_metrics(snapshot, masked_order_ids=set())
        return SanityCheckResult(
            name="order_flow_metrics_split_backlog_from_in_transit",
            passed=metrics.unassigned_backlog_pressure > 0.0
            and metrics.in_transit_pressure > 0.0
            and metrics.healthy_in_transit_ratio > 0.0,
            details={
                "unassigned_backlog_pressure": metrics.unassigned_backlog_pressure,
                "in_transit_pressure": metrics.in_transit_pressure,
                "healthy_in_transit_ratio": metrics.healthy_in_transit_ratio,
            },
        )
    finally:
        env.close()


def _check_order_flow_telemetry_preserves_raw_pressure_while_actionable_pressure_drops() -> SanityCheckResult:
    env = _env()
    try:
        snapshot = env.simulation.snapshot()
        now = float(snapshot["time"])
        for order in snapshot["orders"].values():
            if isinstance(order, dict):
                order["status"] = "in_transit"
                order["delivery_time"] = None
                order["assigned_vehicle_id"] = "vehicle-1"
                order["due_time"] = now + 60.0
        metrics = env._order_flow_metrics(snapshot, masked_order_ids=set())
        actionable_pending_work_pressure = env._pending_work_pressure(snapshot, masked_order_ids=set())
        raw_pending_work_pressure = env._raw_pending_work_pressure(snapshot, masked_order_ids=set())
        raw_lateness_risk = env._lateness_risk(snapshot)
        return SanityCheckResult(
            name="order_flow_telemetry_preserves_raw_pressure_while_actionable_pressure_drops",
            passed=metrics.unassigned_backlog_pressure == 0.0
            and metrics.in_transit_pressure == 1.0
            and actionable_pending_work_pressure == 0.0
            and raw_pending_work_pressure == 1.0
            and raw_lateness_risk > 0.90,
            details={
                "unassigned_backlog_pressure": metrics.unassigned_backlog_pressure,
                "in_transit_pressure": metrics.in_transit_pressure,
                "actionable_pending_work_pressure": actionable_pending_work_pressure,
                "raw_pending_work_pressure": raw_pending_work_pressure,
                "raw_lateness_risk": raw_lateness_risk,
            },
        )
    finally:
        env.close()


def _check_ppo_dispatch_intensity_sets_budget_not_direct_dispatch() -> SanityCheckResult:
    env = _env()
    try:
        hold_action = DiscreteLogisticsAction(
            dispatch=DispatchDecision.HOLD,
            route=RouteDecision.SHORTEST,
            mode=ModeDecision.SECONDARY_FLEET,
            reorder=ReorderDecision.NONE,
        )
        info = env._apply_action(
            {
                "continuous": [-1.0, 1.0, 0.0, -1.0, -1.0],
                "discrete": env.discrete_mapper.encode(hold_action),
            }
        )
        assigned_orders = sum(1 for order in env.simulation.orders.values() if order.assigned_vehicle_id is not None)
        return SanityCheckResult(
            name="ppo_dispatch_intensity_sets_budget_not_direct_dispatch",
            passed=info["macro_dispatch_release"] == 1.0
            and info["raw_macro_dispatch_budget"] == 5.0
            and 1.0 <= info["macro_dispatch_budget"] <= info["raw_macro_dispatch_budget"]
            and info["dqn_dispatch_requested"] == 0.0
            and info["dqn_dispatched_orders"] == 0.0
            and assigned_orders == 0,
            details={
                "macro_dispatch_release": info["macro_dispatch_release"],
                "raw_macro_dispatch_budget": info["raw_macro_dispatch_budget"],
                "feasible_macro_dispatch_cap": info["feasible_macro_dispatch_cap"],
                "macro_dispatch_budget": info["macro_dispatch_budget"],
                "dqn_dispatch_requested": info["dqn_dispatch_requested"],
                "dqn_dispatched_orders": info["dqn_dispatched_orders"],
                "assigned_orders": float(assigned_orders),
            },
        )
    finally:
        env.close()


def _check_global_flow_credit_requires_real_pressure_reduction() -> SanityCheckResult:
    env = _env()
    try:
        previous = env.simulation.snapshot()
        current = {**previous, "orders": {order_id: dict(order) for order_id, order in previous["orders"].items()}}
        now = float(previous["time"])
        for order in previous["orders"].values():
            order["status"] = "created"
            order["delivery_time"] = None
            order["assigned_vehicle_id"] = None
            order["due_time"] = now + 86_400.0
        for order in current["orders"].values():
            order["status"] = "in_transit"
            order["delivery_time"] = None
            order["assigned_vehicle_id"] = "vehicle-1"
            order["due_time"] = now + 86_400.0
        backlog_reduction, lateness_reduction, credit = env._global_flow_progress_credit(
            env._order_flow_metrics(previous, masked_order_ids=set()),
            env._order_flow_metrics(current, masked_order_ids=set()),
        )
        return SanityCheckResult(
            name="global_flow_credit_requires_real_pressure_reduction",
            passed=backlog_reduction > 0.0 and lateness_reduction == 0.0 and credit > 0.0,
            details={
                "backlog_reduction": backlog_reduction,
                "lateness_reduction": lateness_reduction,
                "global_flow_credit": credit,
            },
        )
    finally:
        env.close()


def _check_ppo_flow_enablement_credit_requires_actionable_pressure() -> SanityCheckResult:
    env = _env()
    try:
        _set_calm_inventory(env)
        _make_orders_delivered(env)
        calm = env.simulation.snapshot()
        calm_components = _ppo_components(
            env,
            calm,
            continuous_action={
                "speed_multiplier": 1.35,
                "reorder_fraction": 0.0,
                "dispatch_intensity": 0.0,
                "safety_stock_multiplier": 1.0,
                "capacity_buffer_fraction": 0.5,
            },
        )
        _make_orders_urgent_and_aged(env)
        pressured = env.simulation.snapshot()
        pressured_components = _ppo_components(
            env,
            pressured,
            continuous_action={
                "speed_multiplier": 1.35,
                "reorder_fraction": 0.0,
                "dispatch_intensity": 0.0,
                "safety_stock_multiplier": 1.0,
                "capacity_buffer_fraction": 0.5,
            },
        )
        return SanityCheckResult(
            name="ppo_flow_enablement_credit_requires_actionable_pressure",
            passed=calm_components["ppo_flow_enablement_credit"] == 0.0
            and pressured_components["ppo_flow_enablement_credit"] > 0.0,
            details={
                "calm_flow_credit": calm_components["ppo_flow_enablement_credit"],
                "pressured_flow_credit": pressured_components["ppo_flow_enablement_credit"],
                "pressured_flow": pressured_components["flow_pressure"],
            },
        )
    finally:
        env.close()


def _check_dqn_dispatch_progress_credit_requires_successful_dispatch() -> SanityCheckResult:
    env = _env()
    try:
        _make_orders_urgent_and_aged(env)
        snapshot = env.simulation.snapshot()
        failed = _dqn_components(
            env,
            snapshot,
            discrete_action={"dispatch": "dispatch", "route": "shortest", "mode": "secondary_fleet", "reorder": "none"},
            dispatched_orders=0.0,
        )
        successful = _dqn_components(
            env,
            snapshot,
            discrete_action={"dispatch": "dispatch", "route": "shortest", "mode": "secondary_fleet", "reorder": "none"},
            dispatched_orders=2.0,
            macro_dispatch_budget=2.0,
        )
        return SanityCheckResult(
            name="dqn_dispatch_progress_credit_requires_successful_dispatch",
            passed=failed["dispatch_progress_credit"] == 0.0
            and successful["successful_dispatch_count_capped"] > 0.0
            and successful["dispatch_progress_credit"] > 0.0,
            details={
                "failed_credit": failed["dispatch_progress_credit"],
                "successful_credit": successful["dispatch_progress_credit"],
                "successful_dispatch_count_capped": successful["successful_dispatch_count_capped"],
            },
        )
    finally:
        env.close()


def _check_repeated_infeasible_dispatch_penalty_targets_spam() -> SanityCheckResult:
    env = _env()
    try:
        _make_orders_urgent_and_aged(env)
        for vehicle in env.simulation.vehicles.values():
            vehicle.active = False
        action = DiscreteLogisticsAction(
            dispatch=DispatchDecision.DISPATCH,
            route=RouteDecision.SHORTEST,
            mode=ModeDecision.SECONDARY_FLEET,
            reorder=ReorderDecision.NONE,
        )
        snapshot = env.simulation.snapshot()
        first_info = env._apply_discrete_action(action)
        first_info["action"] = action.as_dict()
        first = env._dqn_local_reward_components(
            snapshot,
            first_info,
            delivered_delta=0,
            cost_delta=0.0,
            masked_order_ids=set(),
        )
        repeated_info = env._apply_discrete_action(action)
        repeated_info["action"] = action.as_dict()
        repeated = env._dqn_local_reward_components(
            snapshot,
            repeated_info,
            delivered_delta=0,
            cost_delta=0.0,
            masked_order_ids=set(),
        )
        return SanityCheckResult(
            name="repeated_infeasible_dispatch_penalty_targets_spam",
            passed=first["infeasible_dispatch_penalty"] == 0.0
            and repeated["infeasible_dispatch_penalty"] > first["infeasible_dispatch_penalty"]
            and repeated["dispatch_progress_credit"] == 0.0,
            details={
                "first_penalty": first["infeasible_dispatch_penalty"],
                "repeated_penalty": repeated["infeasible_dispatch_penalty"],
                "repeated_dispatch_progress_credit": repeated["dispatch_progress_credit"],
                "repeated_attempt_rate": repeated_info["repeated_infeasible_dispatch_attempt_rate"],
            },
        )
    finally:
        env.close()


def _check_emergency_reorder_hurts_under_low_inventory_risk() -> SanityCheckResult:
    env = _env()
    try:
        _set_overstock_low_demand(env)
        snapshot = env.simulation.snapshot()
        no_reorder = _dqn_components(
            env,
            snapshot,
            discrete_action={"dispatch": "hold", "route": "shortest", "mode": "secondary_fleet", "reorder": "none"},
        )["dqn_local"]
        emergency = _dqn_components(
            env,
            snapshot,
            discrete_action={"dispatch": "hold", "route": "shortest", "mode": "secondary_fleet", "reorder": "emergency"},
            override_units=100.0,
            override_useful_units=100.0,
            override_cost=50.0,
        )["dqn_local"]
        return SanityCheckResult(
            name="emergency_reorder_hurts_under_low_inventory_risk",
            passed=emergency < no_reorder,
            details={"no_reorder": no_reorder, "emergency": emergency},
        )
    finally:
        env.close()


def _check_emergency_gate_blocks_low_risk_credit() -> SanityCheckResult:
    env = _env()
    try:
        _set_overstock_low_demand(env)
        snapshot = env.simulation.snapshot()
        components = _dqn_components(
            env,
            snapshot,
            discrete_action={"dispatch": "hold", "route": "shortest", "mode": "secondary_fleet", "reorder": "emergency"},
            override_units=100.0,
            override_useful_units=100.0,
            override_cost=5.0,
        )
        return SanityCheckResult(
            name="emergency_gate_blocks_low_risk_credit",
            passed=components["emergency_gate"] == 0.0
            and components["emergency_procurement_justification_credit"] == 0.0
            and components["emergency_procurement_penalty"] > 0.0,
            details={
                "gate": components["emergency_gate"],
                "credit": components["emergency_procurement_justification_credit"],
                "penalty": components["emergency_procurement_penalty"],
            },
        )
    finally:
        env.close()


def _check_medium_risk_emergency_is_still_costly() -> SanityCheckResult:
    env = _env()
    try:
        _set_medium_inventory_risk(env)
        snapshot = env.simulation.snapshot()
        components = _dqn_components(
            env,
            snapshot,
            discrete_action={"dispatch": "hold", "route": "shortest", "mode": "secondary_fleet", "reorder": "emergency"},
            override_units=100.0,
            override_useful_units=100.0,
            override_cost=30.0,
        )
        return SanityCheckResult(
            name="medium_risk_emergency_is_still_costly",
            passed=components["emergency_gate"] > 0.0
            and components["emergency_gate"] < 1.0
            and components["emergency_procurement_penalty"]
            > components["emergency_procurement_justification_credit"],
            details={
                "gate": components["emergency_gate"],
                "credit": components["emergency_procurement_justification_credit"],
                "penalty": components["emergency_procurement_penalty"],
            },
        )
    finally:
        env.close()


def _check_emergency_credit_requires_useful_override_units() -> SanityCheckResult:
    env = _env()
    try:
        _set_stockout_risk(env)
        _make_orders_urgent(env)
        snapshot = env.simulation.snapshot()
        useful = _dqn_components(
            env,
            snapshot,
            discrete_action={"dispatch": "hold", "route": "shortest", "mode": "secondary_fleet", "reorder": "emergency"},
            override_units=100.0,
            override_useful_units=100.0,
            override_cost=5.0,
        )
        zero_useful = _dqn_components(
            env,
            snapshot,
            discrete_action={"dispatch": "hold", "route": "shortest", "mode": "secondary_fleet", "reorder": "emergency"},
            override_units=100.0,
            override_useful_units=0.0,
            override_cost=5.0,
        )
        return SanityCheckResult(
            name="emergency_credit_requires_useful_override_units",
            passed=useful["emergency_gate"] > 0.90
            and useful["emergency_procurement_justification_credit"] > 0.0
            and zero_useful["emergency_procurement_justification_credit"] == 0.0
            and useful["dqn_local"] > zero_useful["dqn_local"],
            details={
                "gate": useful["emergency_gate"],
                "useful_credit": useful["emergency_procurement_justification_credit"],
                "zero_useful_credit": zero_useful["emergency_procurement_justification_credit"],
            },
        )
    finally:
        env.close()


def _check_dqn_negative_local_is_not_masked_by_global_reward() -> SanityCheckResult:
    reward_global = 0.80
    reward_dqn_local = -0.50
    training_reward = dqn_training_reward(
        reward_global=reward_global,
        reward_dqn_local=reward_dqn_local,
    )
    return SanityCheckResult(
        name="dqn_negative_local_is_not_masked_by_global_reward",
        passed=training_reward < 0.0 and training_reward == 0.90 * reward_dqn_local,
        details={
            "global": reward_global,
            "dqn_local": reward_dqn_local,
            "dqn_train": training_reward,
        },
    )


def _check_hold_has_no_penalty_without_pre_crisis_pressure() -> SanityCheckResult:
    env = _env()
    try:
        _set_overstock_low_demand(env)
        _make_orders_delivered(env)
        snapshot = env.simulation.snapshot()
        components = _dqn_components(
            env,
            snapshot,
            discrete_action={"dispatch": "hold", "route": "shortest", "mode": "secondary_fleet", "reorder": "none"},
        )
        return SanityCheckResult(
            name="hold_has_no_penalty_without_pre_crisis_pressure",
            passed=components["pre_crisis_pressure"] == 0.0
            and components["hold_gate"] == 0.0
            and components["hold_penalty"] == 0.0,
            details={
                "pre_crisis_pressure": components["pre_crisis_pressure"],
                "hold_gate": components["hold_gate"],
                "hold_penalty": components["hold_penalty"],
            },
        )
    finally:
        env.close()


def _check_hold_penalty_rises_with_pre_crisis_pressure() -> SanityCheckResult:
    env = _env()
    try:
        _make_orders_urgent_and_aged(env)
        snapshot = env.simulation.snapshot()
        hold = _dqn_components(
            env,
            snapshot,
            discrete_action={"dispatch": "hold", "route": "shortest", "mode": "secondary_fleet", "reorder": "none"},
        )
        dispatch = _dqn_components(
            env,
            snapshot,
            discrete_action={"dispatch": "dispatch", "route": "shortest", "mode": "secondary_fleet", "reorder": "none"},
            dispatched_orders=1.0,
        )
        return SanityCheckResult(
            name="hold_penalty_rises_with_pre_crisis_pressure",
            passed=hold["pre_crisis_pressure"] > 0.25
            and hold["hold_penalty"] > 0.0
            and dispatch["hold_penalty"] == 0.0,
            details={
                "pre_crisis_pressure": hold["pre_crisis_pressure"],
                "hold_penalty": hold["hold_penalty"],
                "dispatch_hold_penalty": dispatch["hold_penalty"],
            },
        )
    finally:
        env.close()


def _check_delay_abuse_penalty_reduces_late_emergency_credit() -> SanityCheckResult:
    env = _env()
    try:
        _set_stockout_risk(env)
        _make_orders_urgent_and_aged(env)
        snapshot = env.simulation.snapshot()
        env._previous_inaction_pressure = 0.0
        clean = _dqn_components(
            env,
            snapshot,
            discrete_action={"dispatch": "hold", "route": "shortest", "mode": "secondary_fleet", "reorder": "emergency"},
            override_units=100.0,
            override_useful_units=100.0,
            override_cost=5.0,
        )
        env._previous_inaction_pressure = 1.0
        delayed = _dqn_components(
            env,
            snapshot,
            discrete_action={"dispatch": "hold", "route": "shortest", "mode": "secondary_fleet", "reorder": "emergency"},
            override_units=100.0,
            override_useful_units=100.0,
            override_cost=5.0,
        )
        return SanityCheckResult(
            name="delay_abuse_penalty_reduces_late_emergency_credit",
            passed=delayed["delay_abuse_penalty"] > 0.0 and delayed["dqn_local"] < clean["dqn_local"],
            details={
                "clean": clean["dqn_local"],
                "delayed": delayed["dqn_local"],
                "delay_abuse_penalty": delayed["delay_abuse_penalty"],
            },
        )
    finally:
        env.close()


def _check_global_reward_caps_no_delivery_under_pending_pressure() -> SanityCheckResult:
    env = _env()
    try:
        _make_orders_urgent_and_aged(env)
        snapshot = env.simulation.snapshot()
        pending_work_pressure = env._pending_work_pressure(snapshot, masked_order_ids=set())
        global_reward = env._global_reward(
            snapshot,
            delivered_delta=0,
            service_gap=0.0,
            masked_order_ids=set(),
            cost_delta=0.0,
        )
        return SanityCheckResult(
            name="global_reward_caps_no_delivery_under_pending_pressure",
            passed=pending_work_pressure > 0.60 and global_reward <= 0.45,
            details={
                "pending_work_pressure": pending_work_pressure,
                "global_reward": global_reward,
            },
        )
    finally:
        env.close()


def _check_global_reward_does_not_penalize_calm_no_work() -> SanityCheckResult:
    env = _env()
    try:
        _set_no_forecast_risk(env)
        _make_orders_delivered(env)
        snapshot = env.simulation.snapshot()
        adjusted = env._global_reward(
            snapshot,
            delivered_delta=0,
            service_gap=0.0,
            masked_order_ids=set(),
            cost_delta=0.0,
        )
        raw = env._global_reward_raw(
            snapshot,
            delivered_delta=0,
            service_gap=0.0,
            masked_order_ids=set(),
            cost_delta=0.0,
        )
        pending_work_pressure = env._pending_work_pressure(snapshot, masked_order_ids=set())
        return SanityCheckResult(
            name="global_reward_does_not_penalize_calm_no_work",
            passed=pending_work_pressure == 0.0 and adjusted == raw,
            details={
                "pending_work_pressure": pending_work_pressure,
                "adjusted": adjusted,
                "raw": raw,
            },
        )
    finally:
        env.close()


def _check_hold_route_labels_are_neutral() -> SanityCheckResult:
    env = _env()
    try:
        _set_calm_inventory(env)
        snapshot = env.simulation.snapshot()
        shortest = _dqn_components(
            env,
            snapshot,
            discrete_action={"dispatch": "hold", "route": "shortest", "mode": "secondary_fleet", "reorder": "none"},
        )
        low_congestion = _dqn_components(
            env,
            snapshot,
            discrete_action={
                "dispatch": "hold",
                "route": "low_congestion",
                "mode": "secondary_fleet",
                "reorder": "none",
            },
        )
        high_resilience = _dqn_components(
            env,
            snapshot,
            discrete_action={"dispatch": "hold", "route": "high_resilience", "mode": "secondary_fleet", "reorder": "none"},
        )
        locals_equal = (
            abs(shortest["dqn_local"] - low_congestion["dqn_local"]) <= 1e-9
            and abs(shortest["dqn_local"] - high_resilience["dqn_local"]) <= 1e-9
        )
        route_rewards_blocked = all(
            components["hold_route_neutralized"] == 1.0
            and components["route_label_ignored_for_hold"] == 1.0
            and components["selected_route_effective_for_reward"] == -1.0
            and components["route_credit_blocked_hold"] == 1.0
            and components["candidate_alignment_blocked_hold"] == 1.0
            and components["selected_route_candidate_score"] == 0.0
            and components["selected_route_score_gap"] == 0.0
            and components["route_candidate_alignment_credit"] == 0.0
            and components["route_candidate_mismatch_penalty"] == 0.0
            and components["route_adaptation_reward_or_credit"] == 0.0
            and components["shortest_moderate_pressure_relief"] == 0.0
            for components in (shortest, low_congestion, high_resilience)
        )
        return SanityCheckResult(
            name="hold_route_labels_are_neutral",
            passed=locals_equal and route_rewards_blocked,
            details={
                "hold_shortest": shortest["dqn_local"],
                "hold_low_congestion": low_congestion["dqn_local"],
                "hold_high_resilience": high_resilience["dqn_local"],
                "selected_route_effective_for_reward": shortest["selected_route_effective_for_reward"],
                "route_candidate_alignment_credit": low_congestion["route_candidate_alignment_credit"],
                "route_adaptation_reward_or_credit": high_resilience["route_adaptation_reward_or_credit"],
            },
        )
    finally:
        env.close()


def _check_action8_hold_low_congestion_has_no_route_credit_or_inflight_delivery_lift() -> SanityCheckResult:
    env = _env()
    try:
        env.scenario_stress_state.update(
            {
                "route_disruption_probability": 0.25,
                "congestion_multiplier": 2.3125,
                "shortest_route_risk_multiplier": 1.5,
                "traversal_cost_multiplier": 1.3,
                "disruption_risk_multiplier": 1.5,
            }
        )
        _set_calm_inventory(env)
        snapshot = env.simulation.snapshot()
        action8 = {
            "dispatch": "hold",
            "route": "low_congestion",
            "mode": "secondary_fleet",
            "reorder": "none",
        }
        no_delivery = _dqn_components(env, snapshot, discrete_action=action8, delivered_delta=0)
        inflight_completion = _dqn_components(env, snapshot, discrete_action=action8, delivered_delta=2)
        route_credit_blocked = all(
            components["hold_route_neutralized"] == 1.0
            and components["selected_route_effective_for_reward"] == -1.0
            and components["selected_route_candidate_score"] == 0.0
            and components["route_candidate_alignment_credit"] == 0.0
            and components["low_congestion_adaptation_credit"] == 0.0
            and components["route_adaptation_reward_or_credit"] == 0.0
            and components["route_candidate_useful_work_factor"] == 0.0
            and components["dqn_delivery_credit"] == 0.0
            for components in (no_delivery, inflight_completion)
        )
        return SanityCheckResult(
            name="action8_hold_low_congestion_has_no_route_credit_or_inflight_delivery_lift",
            passed=route_credit_blocked
            and no_delivery["hold_inflight_completion_credit_blocked_for_route"] == 0.0
            and inflight_completion["hold_inflight_completion_credit_blocked_for_route"] == 1.0
            and inflight_completion["dqn_local"] <= no_delivery["dqn_local"] + 1e-9,
            details={
                "no_delivery_dqn_local": no_delivery["dqn_local"],
                "inflight_completion_dqn_local": inflight_completion["dqn_local"],
                "route_candidate_alignment_credit": inflight_completion["route_candidate_alignment_credit"],
                "low_congestion_adaptation_credit": inflight_completion["low_congestion_adaptation_credit"],
                "route_candidate_useful_work_factor": inflight_completion["route_candidate_useful_work_factor"],
                "dqn_delivery_credit": inflight_completion["dqn_delivery_credit"],
                "hold_inflight_completion_blocked": inflight_completion[
                    "hold_inflight_completion_credit_blocked_for_route"
                ],
            },
        )
    finally:
        env.close()


def _check_high_resilience_route_is_not_free_in_calm_dispatch_conditions() -> SanityCheckResult:
    env = _env()
    try:
        _set_calm_inventory(env)
        snapshot = env.simulation.snapshot()
        shortest = _dqn_components(
            env,
            snapshot,
            discrete_action={
                "dispatch": "dispatch",
                "route": "shortest",
                "mode": "secondary_fleet",
                "reorder": "none",
            },
            dispatched_orders=1.0,
            delivered_delta=1,
        )
        high_resilience = _dqn_components(
            env,
            snapshot,
            discrete_action={
                "dispatch": "dispatch",
                "route": "high_resilience",
                "mode": "secondary_fleet",
                "reorder": "none",
            },
            dispatched_orders=1.0,
            delivered_delta=1,
        )
        return SanityCheckResult(
            name="high_resilience_route_is_not_free_in_calm_dispatch_conditions",
            passed=high_resilience["dqn_local"] < shortest["dqn_local"]
            and high_resilience["route_premium_penalty"] > 0.0
            and high_resilience["route_adaptation_reward_or_credit"] == 0.0
            and high_resilience["route_candidate_alignment_credit"] == 0.0
            and high_resilience["hold_route_neutralized"] == 0.0,
            details={
                "shortest": shortest["dqn_local"],
                "high_resilience": high_resilience["dqn_local"],
                "route_premium_penalty": high_resilience["route_premium_penalty"],
                "route_adaptation_reward_or_credit": high_resilience["route_adaptation_reward_or_credit"],
                "route_candidate_alignment_credit": high_resilience["route_candidate_alignment_credit"],
            },
        )
    finally:
        env.close()


def _check_action_observation_contract_is_v5_route_candidate_visibility() -> SanityCheckResult:
    env = _env()
    try:
        return SanityCheckResult(
            name="action_observation_contract_is_v5_route_candidate_visibility",
            passed=OBSERVATION_DIM == 73
            and DISCRETE_ACTION_COUNT == 48
            and env.discrete_mapper.action_count == 48
            and MDP_CONTRACT_VERSION == "physical_reality_v5_route_candidate_visibility",
            details={
                "observation_dim": float(OBSERVATION_DIM),
                "discrete_action_count": float(DISCRETE_ACTION_COUNT),
                "mapper_action_count": float(env.discrete_mapper.action_count),
            },
        )
    finally:
        env.close()


def _check_observations_are_finite_and_normalized() -> SanityCheckResult:
    env = _env()
    try:
        scenarios = (_set_calm_inventory, _set_stockout_risk, _set_overstock_low_demand)
        min_value = 1.0
        max_value = 0.0
        for scenario in scenarios:
            scenario(env)
            observation = env.observation_builder.build(env.simulation).vector
            if not np.isfinite(observation).all():
                return SanityCheckResult(
                    name="observation_vectors_are_finite_and_normalized",
                    passed=False,
                    details={"finite": 0.0, "min": float(np.nanmin(observation)), "max": float(np.nanmax(observation))},
                )
            min_value = min(min_value, float(observation.min()))
            max_value = max(max_value, float(observation.max()))
            if bool((observation < 0.0).any()) or bool((observation > 1.0).any()):
                return SanityCheckResult(
                    name="observation_vectors_are_finite_and_normalized",
                    passed=False,
                    details={"finite": 1.0, "min": min_value, "max": max_value},
                )
        return SanityCheckResult(
            name="observation_vectors_are_finite_and_normalized",
            passed=True,
            details={"finite": 1.0, "min": min_value, "max": max_value},
        )
    finally:
        env.close()


def _check_dispatch_feasibility_observation_features_are_bounded() -> SanityCheckResult:
    env = _env()
    try:
        for vehicle in env.simulation.vehicles.values():
            vehicle.active = False
        _make_orders_urgent(env)
        no_vehicle_observation = env.observation_builder.build(env.simulation).vector
        for vehicle in env.simulation.vehicles.values():
            vehicle.active = True
        feasible_observation = env.observation_builder.build(env.simulation).vector

        feature_names = (
            "normalized_available_vehicle_count",
            "normalized_dispatchable_order_count",
            "feasible_dispatch_opportunity",
            "feasible_dispatch_ratio",
            "vehicle_availability_pressure",
        )
        no_vehicle_values = {
            name: float(no_vehicle_observation[OBSERVATION_FEATURE_INDEX[name]])
            for name in feature_names
        }
        feasible_values = {
            name: float(feasible_observation[OBSERVATION_FEATURE_INDEX[name]])
            for name in feature_names
        }
        all_values = tuple(no_vehicle_values.values()) + tuple(feasible_values.values())
        return SanityCheckResult(
            name="dispatch_feasibility_observation_features_are_bounded",
            passed=OBSERVATION_DIM == 73
            and all(0.0 <= value <= 1.0 for value in all_values)
            and no_vehicle_values["feasible_dispatch_opportunity"] == 0.0
            and no_vehicle_values["vehicle_availability_pressure"] == 1.0
            and feasible_values["feasible_dispatch_opportunity"] == 1.0,
            details={
                "observation_dim": float(OBSERVATION_DIM),
                "no_vehicle_feasible_dispatch_opportunity": no_vehicle_values["feasible_dispatch_opportunity"],
                "no_vehicle_availability_pressure": no_vehicle_values["vehicle_availability_pressure"],
                "feasible_dispatch_opportunity": feasible_values["feasible_dispatch_opportunity"],
                "feasible_dispatch_ratio": feasible_values["feasible_dispatch_ratio"],
            },
        )
    finally:
        env.close()


def _check_real_world_stress_observation_features_are_bounded() -> SanityCheckResult:
    env = _env()
    try:
        env.scenario_stress_state.update(
            {
                "holding_cost_multiplier": 2.25,
                "excess_inventory_penalty_multiplier": 2.0,
                "planned_replenishment_cost_multiplier": 2.0,
                "stockout_penalty_multiplier": 1.6,
                "inventory_shortfall_penalty_multiplier": 1.5,
                "vehicle_availability_multiplier": 0.5,
                "fleet_capacity_multiplier": 0.55,
                "capacity_shock_severity": 0.45,
                "route_disruption_probability": 0.6,
                "congestion_multiplier": 3.0,
                "traversal_cost_multiplier": 1.5,
                "premium_sla_ratio": 0.55,
                "urgent_due_window_multiplier": 0.65,
                "premium_lateness_penalty_multiplier": 1.8,
                "supplier_delay_probability": 0.45,
                "lead_time_mean_multiplier": 1.8,
                "lead_time_variance_multiplier": 3.0,
            }
        )
        observation = env.observation_builder.build(
            env.simulation,
            scenario_stress_state=env.scenario_stress_state,
        ).vector
        feature_names = (
            "normalized_holding_cost_pressure",
            "normalized_stockout_penalty_pressure",
            "capacity_shock_pressure",
            "scenario_route_disruption_pressure",
            "route_cost_pressure",
            "premium_sla_pressure",
            "supplier_delay_pressure",
            "scenario_lead_time_volatility_pressure",
        )
        values = {name: float(observation[OBSERVATION_FEATURE_INDEX[name]]) for name in feature_names}
        return SanityCheckResult(
            name="real_world_stress_observation_features_are_bounded",
            passed=OBSERVATION_DIM == 73
            and all(0.0 <= value <= 1.0 for value in values.values())
            and values["normalized_holding_cost_pressure"] > 0.0
            and values["capacity_shock_pressure"] > 0.0
            and values["scenario_route_disruption_pressure"] > 0.0
            and values["premium_sla_pressure"] > 0.0
            and values["supplier_delay_pressure"] > 0.0,
            details={
                "observation_dim": float(OBSERVATION_DIM),
                **values,
            },
        )
    finally:
        env.close()


def _check_route_candidate_visibility_observation_features_are_bounded() -> SanityCheckResult:
    feature_names = (
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
    missing = [name for name in feature_names if name not in OBSERVATION_FEATURE_INDEX]
    if missing:
        return SanityCheckResult(
            name="route_candidate_visibility_observation_features_are_bounded",
            passed=False,
            details={"missing_feature_count": float(len(missing)), "observation_dim": float(OBSERVATION_DIM)},
        )

    env = _env()
    try:
        _make_orders_urgent(env)
        for vehicle in env.simulation.vehicles.values():
            vehicle.active = True
        env.scenario_stress_state.update(
            {
                "route_disruption_probability": 0.25,
                "congestion_multiplier": 2.3125,
                "shortest_route_risk_multiplier": 1.5,
                "traversal_cost_multiplier": 1.3,
                "disruption_risk_multiplier": 1.5,
            }
        )
        safe_values = _observation_feature_values(env, feature_names)

        env.scenario_stress_state.clear()
        env.scenario_stress_state.update(
            {
                "route_disruption_probability": 0.05,
                "congestion_multiplier": 2.8,
                "shortest_route_risk_multiplier": 1.05,
                "traversal_cost_multiplier": 2.0,
                "disruption_risk_multiplier": 1.05,
            }
        )
        congestion_values = _observation_feature_values(env, feature_names)

        env.scenario_stress_state.clear()
        env.scenario_stress_state.update(
            {
                "route_disruption_probability": 0.50,
                "congestion_multiplier": 1.3,
                "shortest_route_risk_multiplier": 2.0,
                "traversal_cost_multiplier": 1.1,
                "disruption_risk_multiplier": 2.0,
            }
        )
        reliability_values = _observation_feature_values(env, feature_names)

        all_values = tuple(safe_values.values()) + tuple(congestion_values.values()) + tuple(reliability_values.values())
        best_safe_score = max(
            safe_values["shortest_candidate_score"],
            safe_values["low_congestion_candidate_score"],
            safe_values["high_resilience_candidate_score"],
        )
        shortest_gap_matches = (
            abs(
                safe_values["shortest_candidate_score_gap"]
                - max(best_safe_score - safe_values["shortest_candidate_score"], 0.0)
            )
            <= 1e-6
        )
        return SanityCheckResult(
            name="route_candidate_visibility_observation_features_are_bounded",
            passed=OBSERVATION_DIM == 73
            and all(0.0 <= value <= 1.0 for value in all_values)
            and safe_values["shortest_candidate_score"] > 0.0
            and safe_values["shortest_candidate_near_best"] == 1.0
            and safe_values["shortest_secondary_safe_context"] > 0.0
            and shortest_gap_matches
            and congestion_values["low_congestion_candidate_score"] > congestion_values["high_resilience_candidate_score"]
            and reliability_values["high_resilience_candidate_score"] > reliability_values["low_congestion_candidate_score"],
            details={
                "observation_dim": float(OBSERVATION_DIM),
                "safe_shortest_candidate_score": safe_values["shortest_candidate_score"],
                "safe_shortest_candidate_score_gap": safe_values["shortest_candidate_score_gap"],
                "safe_shortest_secondary_safe_context": safe_values["shortest_secondary_safe_context"],
                "congestion_low_congestion_candidate_score": congestion_values["low_congestion_candidate_score"],
                "reliability_high_resilience_candidate_score": reliability_values["high_resilience_candidate_score"],
            },
        )
    finally:
        env.close()


def _observation_feature_values(env: FivePLDigitalTwinEnv, feature_names: tuple[str, ...]) -> dict[str, float]:
    observation = env.observation_builder.build(
        env.simulation,
        scenario_stress_state=env.scenario_stress_state,
    ).vector
    return {
        name: float(observation[OBSERVATION_FEATURE_INDEX[name]])
        for name in feature_names
    }


def _env() -> FivePLDigitalTwinEnv:
    return FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False, random_seed=42))


def _ppo_components(
    env: FivePLDigitalTwinEnv,
    snapshot: dict[str, Any],
    *,
    continuous_action: dict[str, float],
    planned_units: float = 0.0,
    useful_units: float | None = None,
    planned_cost: float = 0.0,
    speed_cost_delta: float = 0.0,
    distance_cost_delta: float = 0.0,
) -> dict[str, float]:
    return env._ppo_local_reward_components(
        snapshot,
        {
            "action": continuous_action,
            "planned_replenishment_units": planned_units,
            "planned_replenishment_useful_units": planned_units if useful_units is None else useful_units,
            "planned_replenishment_cost": planned_cost,
        },
        cost_delta=max(speed_cost_delta, planned_cost),
        speed_cost_delta=speed_cost_delta,
        distance_cost_delta=distance_cost_delta,
    )


def _dqn_components(
    env: FivePLDigitalTwinEnv,
    snapshot: dict[str, Any],
    *,
    discrete_action: dict[str, str],
    override_units: float = 0.0,
    override_useful_units: float | None = None,
    override_cost: float = 0.0,
    dispatched_orders: float = 0.0,
    macro_dispatch_budget: float = 1.0,
    delivered_delta: int = 0,
) -> dict[str, float]:
    return env._dqn_local_reward_components(
        snapshot,
        {
            "action": discrete_action,
            "dqn_replenishment_override_units": override_units,
            "dqn_replenishment_override_useful_units": (
                override_units if override_useful_units is None else override_useful_units
            ),
            "dqn_replenishment_override_cost": override_cost,
            "dqn_dispatched_orders": dispatched_orders,
            "macro_dispatch_budget": macro_dispatch_budget,
        },
        delivered_delta=delivered_delta,
        cost_delta=0.0,
        masked_order_ids=set(),
    )


def _set_calm_inventory(env: FivePLDigitalTwinEnv) -> None:
    for position in env.simulation.inventory_network.positions.values():
        position.on_hand_units = max(position.on_hand_units, 1_000.0)
        position.reserved_units = 0.0
        position.in_transit_units = 0.0
        position.backlog_units = 0.0
        position.demand_mean = min(position.demand_mean, 25.0)
        position.demand_variance = 0.0
        position.lead_time_variance = 0.0
        position.safety_stock_units = max(position.safety_stock_units, position.target_safety_stock())
    _make_orders_nonurgent(env)


def _set_stockout_risk(env: FivePLDigitalTwinEnv) -> None:
    for position in env.simulation.inventory_network.positions.values():
        position.on_hand_units = 5.0
        position.reserved_units = 20.0
        position.in_transit_units = 0.0
        position.backlog_units = max(position.demand_mean * 3.0, 100.0)
        position.demand_mean = max(position.demand_mean, 100.0)
        position.demand_variance = max(position.demand_variance, 50.0)
        position.safety_stock_units = 0.0


def _set_medium_inventory_risk(env: FivePLDigitalTwinEnv) -> None:
    for position in env.simulation.inventory_network.positions.values():
        position.on_hand_units = 95.0
        position.reserved_units = 0.0
        position.in_transit_units = 0.0
        position.backlog_units = 5.0
        position.demand_mean = 100.0
        position.demand_variance = 10.0
        position.lead_time_mean = 2.0
        position.lead_time_variance = 1.0
        position.safety_stock_units = position.target_safety_stock() * 0.50


def _set_overstock_low_demand(env: FivePLDigitalTwinEnv) -> None:
    for position in env.simulation.inventory_network.positions.values():
        position.on_hand_units = 5_000.0
        position.reserved_units = 0.0
        position.in_transit_units = 0.0
        position.backlog_units = 0.0
        position.demand_mean = 1.0
        position.demand_variance = 0.0
        position.lead_time_mean = 1.0
        position.lead_time_variance = 0.0
        position.safety_stock_units = max(1.0, position.target_safety_stock())
    _make_orders_nonurgent(env)


def _set_no_forecast_risk(env: FivePLDigitalTwinEnv) -> None:
    _set_overstock_low_demand(env)
    for position in env.simulation.inventory_network.positions.values():
        position.demand_variance = 0.0
        position.lead_time_variance = 0.0
        position.safety_stock_units = max(position.safety_stock_units, position.target_safety_stock())


def _make_orders_nonurgent(env: FivePLDigitalTwinEnv) -> None:
    now = env.simulation.env.now
    for order in env.simulation.orders.values():
        order.release_time = now
        order.due_time = now + 48 * 3_600.0


def _make_orders_urgent(env: FivePLDigitalTwinEnv) -> None:
    now = env.simulation.env.now
    for order in env.simulation.orders.values():
        order.status = ShipmentStatus.CREATED
        order.release_time = now
        order.due_time = now + 60.0
        order.delivery_time = None
        order.assigned_vehicle_id = None


def _make_orders_urgent_and_aged(env: FivePLDigitalTwinEnv) -> None:
    now = env.simulation.env.now
    for order in env.simulation.orders.values():
        order.status = ShipmentStatus.CREATED
        order.release_time = now - 28_800.0
        order.due_time = now + 60.0
        order.delivery_time = None
        order.assigned_vehicle_id = None


def _make_orders_delivered(env: FivePLDigitalTwinEnv) -> None:
    now = env.simulation.env.now
    for order in env.simulation.orders.values():
        order.mark_delivered(now)


if __name__ == "__main__":
    raise SystemExit(main())
