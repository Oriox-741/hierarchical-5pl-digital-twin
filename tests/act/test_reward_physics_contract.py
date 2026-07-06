from __future__ import annotations

import copy
import unittest
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from src.act.discrete_action_mapper import (
    DISCRETE_ACTION_COUNT,
    DispatchDecision,
    DiscreteLogisticsAction,
    ModeDecision,
    ReorderDecision,
    RouteDecision,
)
from src.act.entities import Order, OrderLine, Vehicle, VehicleTier
from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv
from src.act.observation_builder import OBSERVATION_DIM, OBSERVATION_FEATURE_INDEX
from src.shared.types import AssetKind, ShipmentStatus


class RewardPhysicsContractTest(unittest.TestCase):
    def test_max_speed_in_calm_conditions_is_worse_than_nominal_speed(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            calm = _calm_snapshot(env)
            nominal = env._ppo_local_reward(calm, _continuous_info(speed=1.0), cost_delta=0.0)
            max_speed = env._ppo_local_reward(calm, _continuous_info(speed=1.35), cost_delta=0.0)

            self.assertLess(max_speed, nominal)
        finally:
            env.close()

    def test_max_speed_under_urgent_backlog_has_credit_but_still_carries_cost(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_high_risk(env)
            stressed = env.simulation.snapshot()
            now = float(stressed["time"])
            for order in stressed["orders"].values():
                if isinstance(order, dict):
                    order["status"] = "pending"
                    order["due_time"] = now + 60.0
            components = env._ppo_local_reward_components(
                stressed,
                _continuous_info(speed=1.35),
                cost_delta=60.0,
                speed_cost_delta=45.0,
                distance_cost_delta=120.0,
            )

            self.assertGreater(components["speed_justification_credit"], 0.0)
            self.assertGreater(components["speed_cost_penalty"], 0.0)
        finally:
            env.close()

    def test_speed_justification_credit_requires_actual_movement(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_high_risk(env)
            stressed = env.simulation.snapshot()
            now = float(stressed["time"])
            for order in stressed["orders"].values():
                if isinstance(order, dict):
                    order["status"] = "pending"
                    order["due_time"] = now + 60.0
            no_movement = env._ppo_local_reward_components(
                stressed,
                _continuous_info(speed=1.35),
                cost_delta=0.0,
                speed_cost_delta=0.0,
                distance_cost_delta=0.0,
            )
            actual_movement = env._ppo_local_reward_components(
                stressed,
                _continuous_info(speed=1.35),
                cost_delta=60.0,
                speed_cost_delta=45.0,
                distance_cost_delta=120.0,
            )

            self.assertEqual(no_movement["speed_justification_credit"], 0.0)
            self.assertGreater(no_movement["no_movement_speed_penalty"], 0.0)
            self.assertGreater(actual_movement["speed_justification_credit"], 0.0)
            self.assertGreater(actual_movement["speed_cost_penalty"], 0.0)
        finally:
            env.close()

    def test_planned_reorder_is_rewarded_under_stockout_risk_and_penalized_under_low_risk(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_high_risk(env)
            high_risk = env.simulation.snapshot()
            no_reorder_high = env._ppo_local_reward(high_risk, _continuous_info(reorder=0.0), cost_delta=0.0)
            reorder_high = env._ppo_local_reward(high_risk, _continuous_info(reorder=1.0), cost_delta=0.0)

            _make_inventory_low_risk(env)
            low_risk = env.simulation.snapshot()
            no_reorder_low = env._ppo_local_reward(low_risk, _continuous_info(reorder=0.0), cost_delta=0.0)
            reorder_low = env._ppo_local_reward(low_risk, _continuous_info(reorder=1.0), cost_delta=0.0)

            self.assertGreater(reorder_high, no_reorder_high)
            self.assertLess(reorder_low, no_reorder_low)
        finally:
            env.close()

    def test_planned_replenishment_cost_reduces_ppo_reward_for_same_units_and_risk(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_high_risk(env)
            high_risk = env.simulation.snapshot()
            low_cost = env._ppo_local_reward_components(
                high_risk,
                _continuous_info(reorder=1.0, planned_units=100.0, useful_units=100.0, planned_cost=5.0),
                cost_delta=5.0,
                speed_cost_delta=0.0,
            )
            high_cost = env._ppo_local_reward_components(
                high_risk,
                _continuous_info(reorder=1.0, planned_units=100.0, useful_units=100.0, planned_cost=80.0),
                cost_delta=80.0,
                speed_cost_delta=0.0,
            )

            self.assertGreater(high_cost["planned_replenishment_cost_penalty"], low_cost["planned_replenishment_cost_penalty"])
            self.assertLess(high_cost["ppo_local"], low_cost["ppo_local"])
        finally:
            env.close()

    def test_safety_stock_multiplier_is_rewarded_under_volatility_and_penalized_under_overstock(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_high_risk(env)
            high_risk = env.simulation.snapshot()
            normal_high = env._ppo_local_reward(high_risk, _continuous_info(safety_stock=1.0), cost_delta=0.0)
            buffer_high = env._ppo_local_reward(high_risk, _continuous_info(safety_stock=2.0), cost_delta=0.0)

            _make_inventory_low_risk(env)
            low_risk = env.simulation.snapshot()
            normal_low = env._ppo_local_reward(low_risk, _continuous_info(safety_stock=1.0), cost_delta=0.0)
            buffer_low = env._ppo_local_reward(low_risk, _continuous_info(safety_stock=2.0), cost_delta=0.0)

            self.assertGreater(buffer_high, normal_high)
            self.assertLess(buffer_low, normal_low)
        finally:
            env.close()

    def test_high_holding_cost_penalizes_unnecessary_max_inventory_posture(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_no_forecast_risk(env)
            env.scenario_stress_state.update(
                {
                    "holding_cost_multiplier": 2.25,
                    "excess_inventory_penalty_multiplier": 2.0,
                    "planned_replenishment_cost_multiplier": 2.0,
                    "stockout_penalty_multiplier": 1.0,
                    "inventory_shortfall_penalty_multiplier": 1.0,
                    "supplier_delay_probability": 0.0,
                    "lead_time_mean_multiplier": 1.0,
                    "lead_time_variance_multiplier": 1.0,
                }
            )
            low_risk = env.simulation.snapshot()
            moderate = env._ppo_local_reward_components(
                low_risk,
                _continuous_info(reorder=0.35, safety_stock=1.25, planned_cost=2.0),
                cost_delta=2.0,
                speed_cost_delta=0.0,
            )
            near_max = env._ppo_local_reward_components(
                low_risk,
                _continuous_info(reorder=1.0, safety_stock=2.0, planned_cost=5.0),
                cost_delta=5.0,
                speed_cost_delta=0.0,
            )

            self.assertGreater(near_max["holding_cost_pressure"], 0.0)
            self.assertEqual(near_max["stockout_risk"], 0.0)
            self.assertEqual(near_max["inventory_shortfall"], 0.0)
            self.assertGreater(near_max["cost_sensitive_inventory_penalty"], moderate["cost_sensitive_inventory_penalty"])
            self.assertLess(near_max["ppo_local"], moderate["ppo_local"])
        finally:
            env.close()

    def test_high_stockout_or_lead_time_risk_can_justify_safety_stock(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_high_risk(env)
            env.scenario_stress_state.update(
                {
                    "holding_cost_multiplier": 2.25,
                    "stockout_penalty_multiplier": 1.6,
                    "inventory_shortfall_penalty_multiplier": 1.5,
                    "supplier_delay_probability": 0.45,
                    "lead_time_mean_multiplier": 1.8,
                    "lead_time_variance_multiplier": 3.0,
                }
            )
            high_risk = env.simulation.snapshot()
            negligent = env._ppo_local_reward_components(
                high_risk,
                _continuous_info(reorder=0.0, safety_stock=1.0, planned_cost=0.0),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )
            proactive = env._ppo_local_reward_components(
                high_risk,
                _continuous_info(reorder=1.0, safety_stock=2.0, planned_cost=5.0),
                cost_delta=5.0,
                speed_cost_delta=0.0,
            )

            self.assertGreater(proactive["inventory_risk_justification"], 0.0)
            self.assertLess(proactive["cost_sensitive_inventory_penalty"], 0.05)
            self.assertGreater(proactive["planned_replenishment_credit"], 0.0)
            self.assertGreater(proactive["safety_stock_credit"], 0.0)
            self.assertGreater(proactive["ppo_local"], negligent["ppo_local"])
        finally:
            env.close()

    def test_default_holding_cost_pressure_does_not_penalize_normal_inventory_posture(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_no_forecast_risk(env)
            calm = env.simulation.snapshot()
            components = env._ppo_local_reward_components(
                calm,
                _continuous_info(reorder=0.35, safety_stock=1.25, planned_cost=2.0),
                cost_delta=2.0,
                speed_cost_delta=0.0,
            )

            self.assertEqual(components["holding_cost_pressure"], 0.0)
            self.assertEqual(components["cost_sensitive_inventory_penalty"], 0.0)
        finally:
            env.close()

    def test_ppo_inventory_neglect_penalties_apply_under_forecast_risk(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_high_risk(env)
            high_risk = env.simulation.snapshot()
            negligent = env._ppo_local_reward_components(
                high_risk,
                _continuous_info(reorder=0.0, safety_stock=1.0, capacity_buffer=0.0),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )
            proactive = env._ppo_local_reward_components(
                high_risk,
                _continuous_info(reorder=1.0, safety_stock=2.0, capacity_buffer=0.0, planned_cost=5.0),
                cost_delta=5.0,
                speed_cost_delta=0.0,
            )

            self.assertGreater(negligent["forecast_inventory_risk"], 0.0)
            self.assertGreater(negligent["reorder_neglect_penalty"], 0.0)
            self.assertGreater(negligent["safety_stock_neglect_penalty"], 0.0)
            self.assertEqual(proactive["reorder_neglect_penalty"], 0.0)
            self.assertEqual(proactive["safety_stock_neglect_penalty"], 0.0)
            self.assertGreater(proactive["ppo_local"], negligent["ppo_local"])
        finally:
            env.close()

    def test_ppo_neglect_penalties_do_not_apply_without_forecast_or_capacity_pressure(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_no_forecast_risk(env)
            calm = copy.deepcopy(env.simulation.snapshot())
            calm["disruption_score"] = 0.0
            calm["network_safety_potential"] = 0.05
            for order in calm["orders"].values():
                if isinstance(order, dict):
                    order["status"] = "delivered"
            components = env._ppo_local_reward_components(
                calm,
                _continuous_info(reorder=0.0, safety_stock=1.0, capacity_buffer=0.0),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )

            self.assertEqual(components["forecast_inventory_risk"], 0.0)
            self.assertEqual(components["capacity_pressure"], 0.0)
            self.assertEqual(components["reorder_neglect_penalty"], 0.0)
            self.assertEqual(components["safety_stock_neglect_penalty"], 0.0)
            self.assertEqual(components["capacity_neglect_penalty"], 0.0)
        finally:
            env.close()

    def test_ppo_capacity_neglect_penalty_requires_capacity_pressure(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_capacity_pressure(env)
            snapshot = env.simulation.snapshot()
            low_buffer = env._ppo_local_reward_components(
                snapshot,
                _continuous_info(reorder=0.0, safety_stock=1.0, capacity_buffer=0.0),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )
            max_buffer = env._ppo_local_reward_components(
                snapshot,
                _continuous_info(reorder=0.0, safety_stock=1.0, capacity_buffer=0.5),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )

            self.assertGreater(low_buffer["capacity_pressure"], 0.0)
            self.assertGreater(low_buffer["capacity_neglect_penalty"], 0.0)
            self.assertEqual(max_buffer["capacity_neglect_penalty"], 0.0)
        finally:
            env.close()

    def test_low_speed_penalty_requires_flow_pressure(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            calm = _calm_snapshot(env)
            components = env._ppo_local_reward_components(
                calm,
                _continuous_info(speed=0.65),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )

            self.assertEqual(components["flow_pressure"], 0.0)
            self.assertEqual(components["low_speed_lateness_penalty"], 0.0)
        finally:
            env.close()

    def test_low_speed_is_penalized_under_lateness_pressure(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_no_forecast_risk(env)
            _make_orders_urgent_and_aged(env)
            snapshot = env.simulation.snapshot()
            low_speed = env._ppo_local_reward_components(
                snapshot,
                _continuous_info(speed=0.65),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )
            nominal_speed = env._ppo_local_reward_components(
                snapshot,
                _continuous_info(speed=1.0),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )

            self.assertGreater(low_speed["flow_pressure"], 0.0)
            self.assertGreater(low_speed["low_speed_lateness_penalty"], 0.0)
            self.assertEqual(nominal_speed["low_speed_lateness_penalty"], 0.0)
            self.assertGreater(nominal_speed["ppo_local"], low_speed["ppo_local"])
        finally:
            env.close()

    def test_capacity_buffer_neglect_penalty_requires_flow_pressure(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            calm = _calm_snapshot(env)
            components = env._ppo_local_reward_components(
                calm,
                _continuous_info(capacity_buffer=0.0),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )

            self.assertEqual(components["flow_pressure"], 0.0)
            self.assertEqual(components["flow_capacity_neglect_penalty"], 0.0)
        finally:
            env.close()

    def test_capacity_buffer_is_rewarded_under_pending_work_pressure(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_no_forecast_risk(env)
            _make_orders_urgent_and_aged(env)
            snapshot = env.simulation.snapshot()
            low_buffer = env._ppo_local_reward_components(
                snapshot,
                _continuous_info(capacity_buffer=0.0),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )
            high_buffer = env._ppo_local_reward_components(
                snapshot,
                _continuous_info(capacity_buffer=0.5),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )

            self.assertGreater(low_buffer["flow_pressure"], 0.0)
            self.assertGreater(low_buffer["flow_capacity_neglect_penalty"], 0.0)
            self.assertEqual(high_buffer["flow_capacity_neglect_penalty"], 0.0)
            self.assertGreater(high_buffer["ppo_local"], low_buffer["ppo_local"])
        finally:
            env.close()

    def test_vehicle_scarcity_rewards_nonzero_capacity_buffer_when_pressure_exists(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_no_forecast_risk(env)
            env.scenario_stress_state.update(
                {
                    "vehicle_availability_multiplier": 0.35,
                    "fleet_capacity_multiplier": 0.50,
                    "capacity_shock_severity": 0.65,
                }
            )
            snapshot = env.simulation.snapshot()
            near_zero_buffer = env._ppo_local_reward_components(
                snapshot,
                _continuous_info(capacity_buffer=0.0),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )
            modest_buffer = env._ppo_local_reward_components(
                snapshot,
                _continuous_info(capacity_buffer=0.15),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )

            self.assertGreater(near_zero_buffer["capacity_shock_pressure"], 0.0)
            self.assertGreater(near_zero_buffer["capacity_scarcity_pressure"], 0.0)
            self.assertGreater(near_zero_buffer["capacity_scarcity_buffer_penalty"], 0.0)
            self.assertLess(
                modest_buffer["capacity_scarcity_buffer_penalty"],
                near_zero_buffer["capacity_scarcity_buffer_penalty"],
            )
            self.assertGreater(modest_buffer["ppo_local"], near_zero_buffer["ppo_local"])
        finally:
            env.close()

    def test_unnecessary_capacity_buffer_remains_costly_in_calm_conditions(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            calm = _calm_snapshot(env)
            no_buffer = env._ppo_local_reward_components(
                calm,
                _continuous_info(capacity_buffer=0.0),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )
            high_buffer = env._ppo_local_reward_components(
                calm,
                _continuous_info(capacity_buffer=0.5),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )

            self.assertEqual(no_buffer["capacity_scarcity_pressure"], 0.0)
            self.assertEqual(no_buffer["capacity_scarcity_buffer_penalty"], 0.0)
            self.assertEqual(high_buffer["capacity_scarcity_buffer_penalty"], 0.0)
            self.assertGreater(high_buffer["capacity_opportunity_cost"], 0.0)
            self.assertLess(high_buffer["ppo_local"], no_buffer["ppo_local"])
        finally:
            env.close()

    def test_capacity_buffer_repair_preserves_dispatch_reward_protections(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "vehicle_availability_multiplier": 0.35,
                    "fleet_capacity_multiplier": 0.50,
                    "capacity_shock_severity": 0.65,
                }
            )
            _make_orders_unassigned(env)
            for order in env.simulation.orders.values():
                for line in order.lines:
                    line.quantity_units = 220.0
            action = DiscreteLogisticsAction(
                dispatch=DispatchDecision.DISPATCH,
                route=RouteDecision.SHORTEST,
                mode=ModeDecision.PRIMARY_FLEET,
                reorder=ReorderDecision.NONE,
            )

            projection_info = {
                "action": action.as_dict(),
                **env._apply_discrete_action(action, macro_dispatch_release=1.0, macro_dispatch_budget=5),
            }
            components = env._dqn_local_reward_components(
                env.simulation.snapshot(),
                projection_info,
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(projection_info["dqn_dispatched_orders"], 0.0)
            self.assertEqual(projection_info["dispatch_success_count"], 0.0)
            self.assertGreater(projection_info["dispatch_route_failure"], 0.0)
            self.assertEqual(components["dispatch_progress_credit"], 0.0)
            self.assertEqual(components["dispatch_feasibility_credit"], 0.0)
            self.assertEqual(components["successful_dispatch_count_capped"], 0.0)
        finally:
            env.close()

    def test_order_flow_metrics_ignore_delivered_orders(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            snapshot = env.simulation.snapshot()
            for order in snapshot["orders"].values():
                if isinstance(order, dict):
                    order["status"] = "delivered"
                    order["delivery_time"] = float(snapshot["time"])
                    order["assigned_vehicle_id"] = "vehicle-1"

            metrics = env._order_flow_metrics(snapshot, masked_order_ids=set())

            self.assertEqual(metrics.delivered_orders, metrics.eligible_orders)
            self.assertEqual(metrics.pending_orders, 0)
            self.assertEqual(metrics.unassigned_backlog_pressure, 0.0)
            self.assertEqual(metrics.in_transit_pressure, 0.0)
            self.assertEqual(metrics.due_soon_pressure, 0.0)
            self.assertEqual(metrics.true_lateness_pressure, 0.0)
        finally:
            env.close()

    def test_order_flow_metrics_split_unassigned_and_in_transit_pressure(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
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

            self.assertGreater(metrics.unassigned_backlog_pressure, 0.0)
            self.assertGreater(metrics.in_transit_pressure, 0.0)
            self.assertGreater(metrics.healthy_in_transit_ratio, 0.0)
            self.assertNotEqual(metrics.unassigned_backlog_pressure, metrics.in_transit_pressure)
        finally:
            env.close()

    def test_order_flow_metrics_detect_due_soon_and_late_pending_orders(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            due_soon = env.simulation.snapshot()
            now = float(due_soon["time"])
            for order in due_soon["orders"].values():
                if isinstance(order, dict):
                    order["status"] = "created"
                    order["delivery_time"] = None
                    order["assigned_vehicle_id"] = None
                    order["due_time"] = now + 60.0

            late = copy.deepcopy(due_soon)
            for order in late["orders"].values():
                if isinstance(order, dict):
                    order["due_time"] = now - 3_600.0

            due_soon_metrics = env._order_flow_metrics(due_soon, masked_order_ids=set())
            late_metrics = env._order_flow_metrics(late, masked_order_ids=set())

            self.assertGreater(due_soon_metrics.due_soon_pressure, 0.90)
            self.assertEqual(due_soon_metrics.true_lateness_pressure, 0.0)
            self.assertEqual(late_metrics.due_soon_pressure, 0.0)
            self.assertGreater(late_metrics.true_lateness_pressure, 0.0)
        finally:
            env.close()

    def test_healthy_in_transit_no_longer_pins_reward_facing_pending_pressure(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
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

            self.assertEqual(metrics.unassigned_backlog_pressure, 0.0)
            self.assertEqual(metrics.in_transit_pressure, 1.0)
            self.assertGreater(metrics.healthy_in_transit_ratio, 0.0)
            self.assertEqual(env._pending_work_pressure(snapshot, masked_order_ids=set()), 0.0)
            self.assertEqual(env._raw_pending_work_pressure(snapshot, masked_order_ids=set()), 1.0)
            self.assertGreater(env._lateness_risk(snapshot), 0.90)
        finally:
            env.close()

    def test_unassigned_backlog_still_creates_reward_facing_pending_pressure(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            snapshot = env.simulation.snapshot()
            now = float(snapshot["time"])
            for order in snapshot["orders"].values():
                if isinstance(order, dict):
                    order["status"] = "created"
                    order["delivery_time"] = None
                    order["assigned_vehicle_id"] = None
                    order["due_time"] = now + 86_400.0

            metrics = env._order_flow_metrics(snapshot, masked_order_ids=set())

            self.assertEqual(metrics.unassigned_backlog_pressure, 1.0)
            self.assertEqual(metrics.in_transit_pressure, 0.0)
            self.assertEqual(env._pending_work_pressure(snapshot, masked_order_ids=set()), 1.0)
        finally:
            env.close()

    def test_ppo_flow_penalties_are_not_driven_by_healthy_in_transit_work(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_no_forecast_risk(env)
            snapshot = env.simulation.snapshot()
            now = float(snapshot["time"])
            for order in snapshot["orders"].values():
                if isinstance(order, dict):
                    order["status"] = "in_transit"
                    order["delivery_time"] = None
                    order["assigned_vehicle_id"] = "vehicle-1"
                    order["due_time"] = now + 60.0

            components = env._ppo_local_reward_components(
                snapshot,
                _continuous_info(speed=0.65, capacity_buffer=0.0),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )

            self.assertEqual(components["in_transit_pressure"], 1.0)
            self.assertEqual(components["pending_work_pressure"], 0.0)
            self.assertEqual(components["flow_pressure"], 0.0)
            self.assertEqual(components["low_speed_lateness_penalty"], 0.0)
            self.assertEqual(components["flow_capacity_neglect_penalty"], 0.0)
        finally:
            env.close()

    def test_dispatch_diagnostic_telemetry_reports_no_unassigned_orders_without_changing_dispatch(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            for order in env.simulation.orders.values():
                order.mark_delivered(env.simulation.env.now)

            result = env._apply_discrete_action(env.discrete_mapper.map(24))

            self.assertEqual(result["dqn_dispatched_orders"], 0.0)
            self.assertEqual(result["dispatch_attempted"], 1.0)
            self.assertEqual(result["dispatch_success_count"], 0.0)
            self.assertEqual(result["dispatch_no_unassigned_orders"], 1.0)
        finally:
            env.close()

    def test_joint_ppo_dispatch_intensity_sets_macro_dispatch_budget(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            self.assertEqual(env._macro_dispatch_budget(0.0), 1)
            self.assertEqual(env._macro_dispatch_budget(0.49), 2)
            self.assertEqual(env._macro_dispatch_budget(1.0), 5)
        finally:
            env.close()

    def test_joint_ppo_high_dispatch_intensity_does_not_dispatch_when_dqn_holds(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            hold_action = DiscreteLogisticsAction(
                dispatch=DispatchDecision.HOLD,
                route=RouteDecision.SHORTEST,
                mode=ModeDecision.SECONDARY_FLEET,
                reorder=ReorderDecision.NONE,
            )
            projection_info = env._apply_action(
                {
                    "continuous": [-1.0, 1.0, 0.0, -1.0, -1.0],
                    "discrete": env.discrete_mapper.encode(hold_action),
                }
            )

            self.assertEqual(projection_info["macro_dispatch_release"], 1.0)
            self.assertEqual(projection_info["raw_macro_dispatch_budget"], 5.0)
            self.assertEqual(projection_info["dqn_dispatch_requested"], 0.0)
            self.assertEqual(projection_info["dqn_dispatched_orders"], 0.0)
            self.assertTrue(all(order.assigned_vehicle_id is None for order in env.simulation.orders.values()))
        finally:
            env.close()

    def test_ppo_high_dispatch_intensity_without_feasible_opportunity_caps_macro_budget(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_unassigned(env)
            for vehicle in env.simulation.vehicles.values():
                vehicle.active = False
            action = DiscreteLogisticsAction(
                dispatch=DispatchDecision.DISPATCH,
                route=RouteDecision.SHORTEST,
                mode=ModeDecision.SECONDARY_FLEET,
                reorder=ReorderDecision.NONE,
            )

            result = env._apply_discrete_action(action, macro_dispatch_release=1.0, macro_dispatch_budget=5)

            self.assertEqual(result["raw_macro_dispatch_budget"], 5.0)
            self.assertEqual(result["feasible_dispatch_ratio"], 0.0)
            self.assertEqual(result["feasible_macro_dispatch_cap"], 1.0)
            self.assertEqual(result["macro_dispatch_budget"], 1.0)
            self.assertEqual(result["dqn_dispatch_requested"], 1.0)
            self.assertEqual(result["dqn_dispatched_orders"], 0.0)
        finally:
            env.close()

    def test_ppo_high_dispatch_intensity_with_feasible_opportunity_allows_macro_budget_above_baseline(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_unassigned(env)
            _add_secondary_vehicles(env, count=5)
            action = DiscreteLogisticsAction(
                dispatch=DispatchDecision.DISPATCH,
                route=RouteDecision.SHORTEST,
                mode=ModeDecision.SECONDARY_FLEET,
                reorder=ReorderDecision.NONE,
            )

            result = env._apply_discrete_action(action, macro_dispatch_release=1.0, macro_dispatch_budget=5)

            self.assertEqual(result["raw_macro_dispatch_budget"], 5.0)
            self.assertGreater(result["feasible_dispatch_ratio"], 0.0)
            self.assertGreater(result["feasible_macro_dispatch_cap"], 1.0)
            self.assertGreater(result["macro_dispatch_budget"], 1.0)
            self.assertGreater(result["dqn_dispatched_orders"], 1.0)
            self.assertEqual(result["dqn_dispatched_orders"], result["dispatch_success_count"])
        finally:
            env.close()

    def test_demand_surge_backlog_does_not_collapse_macro_cap_to_one_when_vehicle_capacity_exists(self) -> None:
        env = FivePLDigitalTwinEnv(
            EnvironmentConfig(
                rolling_demand_enabled=False,
                rolling_demand_probability=0.70,
                rolling_demand_max_orders_per_step=4,
                rolling_demand_min_units=12.0,
                rolling_demand_max_units=97.5,
                urgent_order_probability=0.425,
            )
        )
        try:
            _make_orders_unassigned(env)
            _add_backlog_orders(env, count=24, quantity_units=20.0)
            action = DiscreteLogisticsAction(
                dispatch=DispatchDecision.DISPATCH,
                route=RouteDecision.SHORTEST,
                mode=ModeDecision.SECONDARY_FLEET,
                reorder=ReorderDecision.NONE,
            )

            result = env._apply_discrete_action(action, macro_dispatch_release=1.0, macro_dispatch_budget=5)

            self.assertEqual(result["feasible_macro_dispatch_cap_before_surge"], 1.0)
            self.assertGreater(result["demand_surge_release_active"], 0.0)
            self.assertGreater(result["demand_surge_macro_floor"], 1.0)
            self.assertGreater(result["feasible_macro_dispatch_cap_after_surge"], 1.0)
            self.assertGreater(result["macro_dispatch_budget_after_surge"], 1.0)
            self.assertGreater(result["dqn_dispatched_orders"], 1.0)
            self.assertEqual(result["dqn_dispatched_orders"], result["dispatch_success_count"])
        finally:
            env.close()

    def test_demand_surge_release_is_inactive_in_calm_normal_backlog(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_unassigned(env)
            action = DiscreteLogisticsAction(
                dispatch=DispatchDecision.DISPATCH,
                route=RouteDecision.SHORTEST,
                mode=ModeDecision.SECONDARY_FLEET,
                reorder=ReorderDecision.NONE,
            )

            result = env._apply_discrete_action(action, macro_dispatch_release=1.0, macro_dispatch_budget=5)

            self.assertEqual(result["demand_surge_release_active"], 0.0)
            self.assertEqual(
                result["feasible_macro_dispatch_cap_after_surge"],
                result["feasible_macro_dispatch_cap_before_surge"],
            )
            self.assertEqual(result["macro_dispatch_budget_after_surge"], result["macro_dispatch_budget_before_surge"])
        finally:
            env.close()

    def test_demand_surge_release_does_not_override_no_vehicle_availability(self) -> None:
        env = FivePLDigitalTwinEnv(
            EnvironmentConfig(
                rolling_demand_enabled=False,
                rolling_demand_probability=0.70,
                rolling_demand_max_orders_per_step=4,
                rolling_demand_min_units=12.0,
                rolling_demand_max_units=97.5,
                urgent_order_probability=0.425,
            )
        )
        try:
            _make_orders_unassigned(env)
            _add_backlog_orders(env, count=24, quantity_units=20.0)
            for vehicle in env.simulation.vehicles.values():
                vehicle.active = False
            action = DiscreteLogisticsAction(
                dispatch=DispatchDecision.DISPATCH,
                route=RouteDecision.SHORTEST,
                mode=ModeDecision.SECONDARY_FLEET,
                reorder=ReorderDecision.NONE,
            )

            result = env._apply_discrete_action(action, macro_dispatch_release=1.0, macro_dispatch_budget=5)

            self.assertEqual(result["available_vehicle_count_all"], 0.0)
            self.assertEqual(result["demand_surge_release_active"], 0.0)
            self.assertEqual(result["macro_dispatch_budget_after_surge"], 1.0)
            self.assertEqual(result["dqn_dispatched_orders"], 0.0)
            self.assertGreater(result["dispatch_no_vehicle_available"], 0.0)
        finally:
            env.close()

    def test_route_feasibility_failure_does_not_leak_vehicle_capacity_or_assignment(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            order = next(iter(env.simulation.orders.values()))
            vehicle = next(vehicle for vehicle in env.simulation.vehicles.values() if vehicle.tier == VehicleTier.SECONDARY)
            for line in order.lines:
                line.quantity_units = vehicle.capacity_units + 1.0
            path = env.simulation.route_network.shortest_path(order.origin_node_id, order.destination_node_id)
            initial_load_units = vehicle.load_units
            initial_remaining_units = vehicle.remaining_units

            env.simulation.dispatch_order(order.order_id, vehicle.vehicle_id, path)
            env.simulation.run_until_next_decision()

            self.assertEqual(vehicle.load_units, initial_load_units)
            self.assertEqual(vehicle.remaining_units, initial_remaining_units)
            self.assertIsNone(order.assigned_vehicle_id)
            self.assertIsNone(order.pickup_time)
            self.assertIsNone(order.delivery_time)
            self.assertEqual(order.status, ShipmentStatus.DELAYED)
            self.assertEqual(env.simulation.state.failed_orders, 1)
        finally:
            env.close()

    def test_customer_can_receive_multiple_distinct_orders_over_time(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            first_order = next(iter(env.simulation.orders.values()))
            vehicle = next(vehicle for vehicle in env.simulation.vehicles.values() if vehicle.tier == VehicleTier.SECONDARY)
            env.simulation.customers[first_order.customer_id].time_window_end = float("inf")
            path = env.simulation.route_network.shortest_path(first_order.origin_node_id, first_order.destination_node_id)

            env.simulation.dispatch_order(first_order.order_id, vehicle.vehicle_id, path)
            env.simulation.run(until=env.simulation.env.now + 86_400.0)
            self.assertEqual(first_order.status, ShipmentStatus.DELIVERED)
            self.assertTrue(env.simulation.customers[first_order.customer_id].visited)

            second_order = _add_order_for_same_customer(env, first_order)
            second_path = env.simulation.route_network.shortest_path(
                second_order.origin_node_id,
                second_order.destination_node_id,
            )
            feasibility = env.simulation.evaluate_order_route(
                second_order,
                vehicle,
                second_path,
                start_time=env.simulation.env.now,
            )

            self.assertTrue(feasibility.feasible)
            self.assertNotIn(f"customer_revisited:{second_order.customer_id}", feasibility.violations)
            env.simulation.dispatch_order(second_order.order_id, vehicle.vehicle_id, second_path)
            env.simulation.run(until=env.simulation.env.now + 86_400.0)

            self.assertEqual(second_order.status, ShipmentStatus.DELIVERED)
            self.assertEqual(env.simulation.state.delivered_orders, 2)
            self.assertEqual(vehicle.load_units, 0.0)
        finally:
            env.close()

    def test_recurring_customer_demand_remains_dispatchable_after_prior_delivery(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            first_order = next(iter(env.simulation.orders.values()))
            vehicle = next(vehicle for vehicle in env.simulation.vehicles.values() if vehicle.tier == VehicleTier.SECONDARY)
            env.simulation.customers[first_order.customer_id].time_window_end = float("inf")
            path = env.simulation.route_network.shortest_path(first_order.origin_node_id, first_order.destination_node_id)
            env.simulation.dispatch_order(first_order.order_id, vehicle.vehicle_id, path)
            env.simulation.run(until=env.simulation.env.now + 86_400.0)
            for order in env.simulation.orders.values():
                if order is not first_order and order.delivery_time is None:
                    order.mark_delivered(env.simulation.env.now)
            second_order = _add_order_for_same_customer(env, first_order)
            action = DiscreteLogisticsAction(
                dispatch=DispatchDecision.DISPATCH,
                route=RouteDecision.SHORTEST,
                mode=ModeDecision.SECONDARY_FLEET,
                reorder=ReorderDecision.NONE,
            )

            feasibility = env._dispatch_feasibility_metrics(action)
            result = env._apply_discrete_action(action, macro_dispatch_release=0.0, macro_dispatch_budget=1)

            self.assertEqual(feasibility["dispatchable_order_count"], 1.0)
            self.assertEqual(feasibility["feasible_dispatch_opportunity"], 1.0)
            self.assertEqual(result["dispatch_route_failure"], 0.0)
            self.assertEqual(result["customer_revisit_blocked_count"], 0.0)
            self.assertEqual(result["dqn_dispatched_orders"], 1.0)
            env.simulation.run_until_next_decision()
            self.assertEqual(second_order.assigned_vehicle_id, vehicle.vehicle_id)
        finally:
            env.close()

    def test_successful_feasible_dispatch_still_loads_assigns_and_delivers(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            order = next(iter(env.simulation.orders.values()))
            vehicle = next(vehicle for vehicle in env.simulation.vehicles.values() if vehicle.tier == VehicleTier.SECONDARY)
            path = env.simulation.route_network.shortest_path(order.origin_node_id, order.destination_node_id)

            env.simulation.dispatch_order(order.order_id, vehicle.vehicle_id, path)
            env.simulation.run(until=env.simulation.env.now + 86_400.0)

            self.assertEqual(order.status, ShipmentStatus.DELIVERED)
            self.assertEqual(order.assigned_vehicle_id, vehicle.vehicle_id)
            self.assertIsNotNone(order.pickup_time)
            self.assertIsNotNone(order.delivery_time)
            self.assertEqual(vehicle.load_units, 0.0)
            self.assertEqual(env.simulation.state.delivered_orders, 1)
        finally:
            env.close()

    def test_route_feasibility_failure_does_not_create_dispatch_progress_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_unassigned(env)
            for order in env.simulation.orders.values():
                for line in order.lines:
                    line.quantity_units = 220.0
            action = DiscreteLogisticsAction(
                dispatch=DispatchDecision.DISPATCH,
                route=RouteDecision.SHORTEST,
                mode=ModeDecision.PRIMARY_FLEET,
                reorder=ReorderDecision.NONE,
            )

            projection_info = {
                "action": action.as_dict(),
                **env._apply_discrete_action(action, macro_dispatch_release=1.0, macro_dispatch_budget=5),
            }
            components = env._dqn_local_reward_components(
                env.simulation.snapshot(),
                projection_info,
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(projection_info["dqn_dispatched_orders"], 0.0)
            self.assertEqual(projection_info["dispatch_success_count"], 0.0)
            self.assertGreater(projection_info["dispatch_route_failure"], 0.0)
            self.assertEqual(components["dispatch_progress_credit"], 0.0)
            self.assertEqual(components["dispatch_feasibility_credit"], 0.0)
        finally:
            env.close()

    def test_feasibility_macro_budget_gating_uses_current_observation_dimension(self) -> None:
        env = FivePLDigitalTwinEnv(
            EnvironmentConfig(rolling_demand_enabled=False, feasibility_gated_macro_budget=True)
        )
        try:
            self.assertEqual(OBSERVATION_DIM, 73)
            self.assertEqual(env.observation_space.shape, (OBSERVATION_DIM,))
            observation, _ = env.reset()
            self.assertEqual(observation.shape, (OBSERVATION_DIM,))
        finally:
            env.close()

    def test_dqn_dispatch_uses_ppo_macro_dispatch_budget_for_actual_assignments(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _add_secondary_vehicles(env, 3)
            _make_orders_unassigned(env)
            action = DiscreteLogisticsAction(
                dispatch=DispatchDecision.DISPATCH,
                route=RouteDecision.SHORTEST,
                mode=ModeDecision.SECONDARY_FLEET,
                reorder=ReorderDecision.NONE,
            )

            low = env._apply_discrete_action(action, macro_dispatch_release=0.0, macro_dispatch_budget=1)
            self.assertEqual(low["macro_dispatch_budget"], 1.0)
            self.assertEqual(low["dqn_dispatch_requested"], 1.0)
            self.assertEqual(low["dqn_dispatched_orders"], 1.0)

            _make_orders_unassigned(env)
            high = env._apply_discrete_action(action, macro_dispatch_release=1.0, macro_dispatch_budget=5)
            self.assertEqual(high["macro_dispatch_release"], 1.0)
            self.assertEqual(high["macro_dispatch_budget"], 5.0)
            self.assertGreater(high["dqn_dispatched_orders"], low["dqn_dispatched_orders"])
            self.assertEqual(high["dqn_dispatched_orders"], high["dispatch_success_count"])
        finally:
            env.close()

    def test_already_assigned_orders_are_not_double_dispatched_by_macro_budget(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _add_secondary_vehicles(env, 3)
            _make_orders_unassigned(env)
            first_vehicle = next(iter(env.simulation.vehicles.values()))
            first_order = next(iter(env.simulation.orders.values()))
            first_order.assigned_vehicle_id = first_vehicle.vehicle_id
            action = DiscreteLogisticsAction(
                dispatch=DispatchDecision.DISPATCH,
                route=RouteDecision.SHORTEST,
                mode=ModeDecision.SECONDARY_FLEET,
                reorder=ReorderDecision.NONE,
            )

            result = env._apply_discrete_action(action, macro_dispatch_release=1.0, macro_dispatch_budget=5)

            self.assertEqual(first_order.assigned_vehicle_id, first_vehicle.vehicle_id)
            self.assertGreater(result["dispatch_already_assigned_count"], 0.0)
            self.assertLessEqual(result["dqn_dispatched_orders"], len(env.simulation.orders) - 1)
        finally:
            env.close()

    def test_inventory_safety_incentives_survive_flow_rebalance(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_high_risk(env)
            snapshot = env.simulation.snapshot()
            negligent = env._ppo_local_reward_components(
                snapshot,
                _continuous_info(reorder=0.0, safety_stock=1.0, capacity_buffer=0.0),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )
            proactive = env._ppo_local_reward_components(
                snapshot,
                _continuous_info(reorder=1.0, safety_stock=2.0, capacity_buffer=0.0, planned_cost=5.0),
                cost_delta=5.0,
                speed_cost_delta=0.0,
            )

            self.assertGreater(proactive["planned_replenishment_credit"], 0.0)
            self.assertGreater(proactive["safety_stock_credit"], 0.0)
            self.assertEqual(proactive["reorder_neglect_penalty"], 0.0)
            self.assertEqual(proactive["safety_stock_neglect_penalty"], 0.0)
            self.assertGreater(proactive["ppo_local"], negligent["ppo_local"])
        finally:
            env.close()

    def test_global_reward_no_delivery_high_pressure_is_capped(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_urgent_and_aged(env)
            current = env.simulation.snapshot()
            global_reward = env._global_reward(
                current,
                delivered_delta=0,
                service_gap=0.0,
                masked_order_ids=set(),
                cost_delta=0.0,
            )

            self.assertGreater(env._pending_work_pressure(current, masked_order_ids=set()), 0.60)
            self.assertLessEqual(global_reward, 0.45)
        finally:
            env.close()

    def test_global_reward_no_delivery_calm_state_is_not_idle_penalized(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            calm = _calm_snapshot(env)
            adjusted = env._global_reward(
                calm,
                delivered_delta=0,
                service_gap=0.0,
                masked_order_ids=set(),
                cost_delta=0.0,
            )
            raw = env._global_reward_raw(
                calm,
                delivered_delta=0,
                service_gap=0.0,
                masked_order_ids=set(),
                cost_delta=0.0,
            )

            self.assertEqual(env._pending_work_pressure(calm, masked_order_ids=set()), 0.0)
            self.assertEqual(adjusted, raw)
        finally:
            env.close()

    def test_global_no_work_cap_does_not_punish_healthy_in_transit_work(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_low_risk(env)
            current = env.simulation.snapshot()
            now = float(current["time"])
            for order in current["orders"].values():
                if isinstance(order, dict):
                    order["status"] = "in_transit"
                    order["delivery_time"] = None
                    order["assigned_vehicle_id"] = "vehicle-1"
                    order["due_time"] = now + 60.0

            adjusted = env._global_reward(
                current,
                delivered_delta=0,
                service_gap=0.0,
                masked_order_ids=set(),
                cost_delta=0.0,
            )
            raw = env._global_reward_raw(
                current,
                delivered_delta=0,
                service_gap=0.0,
                masked_order_ids=set(),
                cost_delta=0.0,
            )

            self.assertEqual(env._pending_work_pressure(current, masked_order_ids=set()), 0.0)
            self.assertEqual(env._raw_pending_work_pressure(current, masked_order_ids=set()), 1.0)
            self.assertEqual(adjusted, raw)
        finally:
            env.close()

    def test_global_idle_adjustment_does_not_change_local_rewards(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_high_risk(env)
            _make_orders_urgent_and_aged(env)
            current = env.simulation.snapshot()
            projection_info = _joint_info(
                continuous=_continuous_info(reorder=0.0, safety_stock=1.0, capacity_buffer=0.0)["action"],
                discrete=_discrete_info(dispatch="hold", reorder="none")["action"],
            )
            ppo_before = env._ppo_local_reward_components(
                current,
                projection_info,
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )["ppo_local"]
            dqn_before = env._dqn_local_reward_components(
                current,
                projection_info,
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )["dqn_local"]
            _, components = env._reward(current, current, projection_info)

            self.assertEqual(components["ppo_local"], ppo_before)
            self.assertEqual(components["dqn_local"], dqn_before)
            self.assertLess(components["global"], components["global_raw"])
        finally:
            env.close()

    def test_global_flow_credit_rewards_true_unassigned_backlog_reduction(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            previous = env.simulation.snapshot()
            current = copy.deepcopy(previous)
            now = float(previous["time"])
            for order in previous["orders"].values():
                if isinstance(order, dict):
                    order["status"] = "created"
                    order["delivery_time"] = None
                    order["assigned_vehicle_id"] = None
                    order["due_time"] = now + 86_400.0
            for order in current["orders"].values():
                if isinstance(order, dict):
                    order["status"] = "in_transit"
                    order["delivery_time"] = None
                    order["assigned_vehicle_id"] = "vehicle-1"
                    order["due_time"] = now + 86_400.0

            _, _, credit = env._global_flow_progress_credit(
                env._order_flow_metrics(previous, masked_order_ids=set()),
                env._order_flow_metrics(current, masked_order_ids=set()),
            )
            without_credit = env._global_reward_raw(
                current,
                delivered_delta=0,
                service_gap=0.0,
                masked_order_ids=set(),
                cost_delta=0.0,
            )
            with_credit = env._global_reward_raw(
                current,
                delivered_delta=0,
                service_gap=0.0,
                masked_order_ids=set(),
                cost_delta=0.0,
                global_flow_credit=credit,
            )

            self.assertGreater(credit, 0.0)
            self.assertGreater(with_credit, without_credit)
        finally:
            env.close()

    def test_global_flow_credit_rewards_true_lateness_reduction(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            previous = env.simulation.snapshot()
            current = copy.deepcopy(previous)
            now = float(previous["time"])
            for order in previous["orders"].values():
                if isinstance(order, dict):
                    order["status"] = "created"
                    order["delivery_time"] = None
                    order["assigned_vehicle_id"] = None
                    order["due_time"] = now - 28_800.0
            for order in current["orders"].values():
                if isinstance(order, dict):
                    order["status"] = "delivered"
                    order["delivery_time"] = now
                    order["assigned_vehicle_id"] = None

            _, lateness_reduction, credit = env._global_flow_progress_credit(
                env._order_flow_metrics(previous, masked_order_ids=set()),
                env._order_flow_metrics(current, masked_order_ids=set()),
            )

            self.assertGreater(lateness_reduction, 0.0)
            self.assertGreater(credit, 0.0)
        finally:
            env.close()

    def test_global_flow_credit_is_zero_without_real_backlog_or_lateness_reduction(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            previous = env.simulation.snapshot()
            current = copy.deepcopy(previous)
            now = float(previous["time"])
            for snapshot in (previous, current):
                for order in snapshot["orders"].values():
                    if isinstance(order, dict):
                        order["status"] = "in_transit"
                        order["delivery_time"] = None
                        order["assigned_vehicle_id"] = "vehicle-1"
                        order["due_time"] = now + 60.0

            backlog_reduction, lateness_reduction, credit = env._global_flow_progress_credit(
                env._order_flow_metrics(previous, masked_order_ids=set()),
                env._order_flow_metrics(current, masked_order_ids=set()),
            )

            self.assertEqual(backlog_reduction, 0.0)
            self.assertEqual(lateness_reduction, 0.0)
            self.assertEqual(credit, 0.0)
        finally:
            env.close()

    def test_ppo_flow_enablement_credit_requires_actionable_pressure(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            calm = _calm_snapshot(env)
            components = env._ppo_local_reward_components(
                calm,
                _continuous_info(speed=1.35, capacity_buffer=0.5),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )

            self.assertEqual(components["flow_pressure"], 0.0)
            self.assertEqual(components["ppo_flow_enablement_credit"], 0.0)
        finally:
            env.close()

    def test_ppo_flow_enablement_prefers_speed_and_capacity_under_actionable_pressure(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_no_forecast_risk(env)
            _make_orders_urgent_and_aged(env)
            snapshot = env.simulation.snapshot()
            low_enablement = env._ppo_local_reward_components(
                snapshot,
                _continuous_info(speed=0.65, capacity_buffer=0.0),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )
            high_enablement = env._ppo_local_reward_components(
                snapshot,
                _continuous_info(speed=1.35, capacity_buffer=0.5),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )

            self.assertGreater(high_enablement["flow_pressure"], 0.0)
            self.assertEqual(low_enablement["ppo_flow_enablement_credit"], 0.0)
            self.assertGreater(high_enablement["ppo_flow_enablement_credit"], 0.0)
            self.assertGreater(high_enablement["ppo_local"], low_enablement["ppo_local"])
        finally:
            env.close()

    def test_dqn_dispatch_progress_credit_requires_successful_dispatch(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_urgent_and_aged(env)
            snapshot = env.simulation.snapshot()
            failed = env._dqn_local_reward_components(
                snapshot,
                _discrete_info(dispatch="dispatch", dispatched_orders=0.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            successful = env._dqn_local_reward_components(
                snapshot,
                {
                    **_discrete_info(dispatch="dispatch", dispatched_orders=2.0),
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(failed["dispatch_progress_credit"], 0.0)
            self.assertEqual(failed["successful_dispatch_count_capped"], 0.0)
            self.assertGreater(successful["successful_dispatch_count_capped"], 0.0)
            self.assertGreater(successful["dispatch_progress_credit"], 0.0)
        finally:
            env.close()

    def test_dqn_no_current_or_no_unassigned_dispatch_guard_preserves_valid_dispatch(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_urgent_and_aged(env)
            snapshot = env.simulation.snapshot()
            no_work = env._dqn_local_reward_components(
                snapshot,
                {
                    **_discrete_info(
                        dispatch="dispatch",
                        route="shortest",
                        mode="secondary_fleet",
                        reorder="conservative",
                        dispatched_orders=0.0,
                    ),
                    "dispatch_success_count": 0.0,
                    "dispatch_no_unassigned_orders": 1.0,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            valid_dispatch = env._dqn_local_reward_components(
                snapshot,
                {
                    **_discrete_info(
                        dispatch="dispatch",
                        route="shortest",
                        mode="primary_fleet",
                        reorder="none",
                        dispatched_orders=1.0,
                    ),
                    "dispatch_success_count": 1.0,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(no_work["no_unassigned_dispatch_exposure"], 0.0)
            self.assertGreater(no_work["no_current_or_unassigned_dispatch_penalty"], 0.0)
            self.assertLessEqual(no_work["dqn_local"], -0.14)
            self.assertEqual(no_work["dispatch_feasibility_credit"], 0.0)
            self.assertEqual(no_work["dispatch_progress_credit"], 0.0)
            self.assertEqual(no_work["dqn_delivery_credit"], 0.0)
            self.assertEqual(no_work["route_candidate_alignment_credit"], 0.0)
            self.assertEqual(no_work["premium_fleet_credit_allowed"], 0.0)

            self.assertGreater(valid_dispatch["dispatch_feasibility_credit"], 0.0)
            self.assertGreater(valid_dispatch["dispatch_progress_credit"], 0.0)
            self.assertEqual(valid_dispatch["no_unassigned_dispatch_exposure"], 0.0)
            self.assertEqual(valid_dispatch["no_current_or_unassigned_dispatch_penalty"], 0.0)
            self.assertGreater(valid_dispatch["dqn_local"], no_work["dqn_local"])
            self.assertLess(valid_dispatch["dqn_local"], 1.0)
        finally:
            env.close()

    def test_dqn_dispatch_progress_credit_is_zero_for_duplicate_or_failed_attempts(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_unassigned(env)
            vehicle_id = next(iter(env.simulation.vehicles))
            for order in env.simulation.orders.values():
                order.assigned_vehicle_id = vehicle_id
                order.status = ShipmentStatus.IN_TRANSIT
            action = DiscreteLogisticsAction(
                dispatch=DispatchDecision.DISPATCH,
                route=RouteDecision.SHORTEST,
                mode=ModeDecision.SECONDARY_FLEET,
                reorder=ReorderDecision.NONE,
            )
            projection_info = env._apply_discrete_action(action, macro_dispatch_release=1.0, macro_dispatch_budget=5)
            projection_info["action"] = action.as_dict()
            snapshot = env.simulation.snapshot()
            components = env._dqn_local_reward_components(
                snapshot,
                projection_info,
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(projection_info["dqn_dispatched_orders"], 0.0)
            self.assertGreater(projection_info["dispatch_already_assigned_count"], 0.0)
            self.assertEqual(components["dispatch_progress_credit"], 0.0)
        finally:
            env.close()

    def test_dqn_route_resilience_is_penalized_in_calm_state_and_justified_by_disruption(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            calm = _calm_snapshot(env)
            disrupted = copy.deepcopy(calm)
            disrupted["disruption_score"] = 0.90

            calm_reward = env._dqn_local_reward(
                calm,
                _discrete_info(dispatch="dispatch", route="high_resilience", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            disrupted_reward = env._dqn_local_reward(
                disrupted,
                _discrete_info(dispatch="dispatch", route="high_resilience", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertLess(calm_reward, disrupted_reward)
        finally:
            env.close()

    def test_dqn_primary_fleet_penalty_is_higher_when_urgency_is_low(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            calm = _calm_snapshot(env)
            urgent = copy.deepcopy(calm)
            now = float(urgent["time"])
            for order in urgent["orders"].values():
                if isinstance(order, dict):
                    order["status"] = "pending"
                    order["due_time"] = now + 60.0

            calm_components = env._dqn_local_reward_components(
                calm,
                _discrete_info(mode="primary_fleet"),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            urgent_components = env._dqn_local_reward_components(
                urgent,
                _discrete_info(mode="primary_fleet"),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(
                calm_components["primary_fleet_cost_penalty"],
                urgent_components["primary_fleet_cost_penalty"],
            )
        finally:
            env.close()

    def test_action_24_and_25_decode_to_shortest_secondary_dispatch(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            self.assertEqual(DISCRETE_ACTION_COUNT, 48)
            self.assertEqual(env.discrete_mapper.action_count, 48)
            self.assertEqual(env.action_space.spaces["discrete"].n, 48)
            action_24 = env.discrete_mapper.map(24)
            action_25 = env.discrete_mapper.map(25)

            self.assertEqual(action_24.dispatch, DispatchDecision.DISPATCH)
            self.assertEqual(action_24.route, RouteDecision.SHORTEST)
            self.assertEqual(action_24.mode, ModeDecision.SECONDARY_FLEET)
            self.assertEqual(action_24.reorder, ReorderDecision.NONE)
            self.assertEqual(action_25.dispatch, DispatchDecision.DISPATCH)
            self.assertEqual(action_25.route, RouteDecision.SHORTEST)
            self.assertEqual(action_25.mode, ModeDecision.SECONDARY_FLEET)
            self.assertEqual(action_25.reorder, ReorderDecision.CONSERVATIVE)
        finally:
            env.close()

    def test_shortest_route_loses_advantage_when_disruption_risk_is_high(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "route_disruption_probability": 0.75,
                    "congestion_multiplier": 3.0,
                    "shortest_route_risk_multiplier": 2.5,
                    "traversal_cost_multiplier": 1.8,
                    "disruption_risk_multiplier": 2.0,
                }
            )
            stressed = _calm_snapshot(env)
            shortest = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="shortest", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            resilient = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="high_resilience", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(shortest["scenario_route_disruption_pressure"], 0.0)
            self.assertGreater(shortest["shortest_route_disruption_risk_penalty"], 0.0)
            self.assertGreater(shortest["route_adaptation_pressure"], 0.0)
            self.assertGreater(shortest["route_narrowness_penalty"], 0.0)
            self.assertGreater(resilient["route_resilience_adaptation_credit"], 0.0)
            self.assertGreater(resilient["route_adaptation_reward_or_credit"], 0.0)
            self.assertGreater(resilient["dqn_local"], shortest["dqn_local"])
        finally:
            env.close()

    def test_fixed_medium_route_disruption_creates_meaningful_alternative_route_gradient(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "route_disruption_probability": 0.25,
                    "congestion_multiplier": 1.75,
                    "shortest_route_risk_multiplier": 1.5,
                    "traversal_cost_multiplier": 1.3,
                    "disruption_risk_multiplier": 1.5,
                }
            )
            stressed = _calm_snapshot(env)
            shortest = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="shortest", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            low_congestion = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="low_congestion", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            high_resilience = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="high_resilience", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(shortest["route_adaptation_pressure"], 0.0)
            self.assertGreater(shortest["shortest_route_disruption_risk_penalty"], 0.02)
            self.assertGreater(low_congestion["route_resilience_adaptation_credit"], 0.0)
            self.assertGreater(high_resilience["route_resilience_adaptation_credit"], 0.0)
            self.assertGreater(low_congestion["dqn_local"], shortest["dqn_local"])
            self.assertGreater(high_resilience["dqn_local"], shortest["dqn_local"])
        finally:
            env.close()

    def test_fixed_medium_route_disruption_keeps_low_congestion_competitive(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "route_disruption_probability": 0.25,
                    "congestion_multiplier": 1.75,
                    "shortest_route_risk_multiplier": 1.5,
                    "traversal_cost_multiplier": 1.3,
                    "disruption_risk_multiplier": 1.5,
                }
            )
            stressed = _calm_snapshot(env)
            shortest = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="shortest", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            low_congestion = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="low_congestion", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            high_resilience = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="high_resilience", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(shortest["shortest_route_disruption_risk_penalty"], 0.02)
            self.assertGreater(low_congestion["dqn_local"], shortest["dqn_local"])
            self.assertGreater(high_resilience["dqn_local"], shortest["dqn_local"])
            self.assertGreater(low_congestion["low_congestion_adaptation_credit"], 0.02)
            self.assertLessEqual(
                high_resilience["route_adaptation_reward_or_credit"],
                low_congestion["route_adaptation_reward_or_credit"] + 0.03,
            )
        finally:
            env.close()

    def test_fixed_medium_route_disruption_does_not_overweight_one_alternative_route(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "route_disruption_probability": 0.25,
                    "congestion_multiplier": 1.75,
                    "shortest_route_risk_multiplier": 1.5,
                    "traversal_cost_multiplier": 1.3,
                    "disruption_risk_multiplier": 1.5,
                }
            )
            stressed = _calm_snapshot(env)
            low_congestion = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="low_congestion", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            high_resilience = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="high_resilience", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(low_congestion["route_congestion_cost_pressure"], 0.0)
            self.assertGreater(high_resilience["route_disruption_reliability_pressure"], 0.0)
            self.assertLessEqual(
                abs(low_congestion["dqn_local"] - high_resilience["dqn_local"]),
                0.02,
            )
            self.assertLessEqual(
                low_congestion["low_congestion_adaptation_credit"],
                high_resilience["route_adaptation_reward_or_credit"] + 0.015,
            )
        finally:
            env.close()

    def test_observed_mixed_route_pressure_dampens_low_congestion_edge(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
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
            stressed = _calm_snapshot(env)
            low_congestion = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="low_congestion", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            high_resilience = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="high_resilience", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertAlmostEqual(low_congestion["route_congestion_cost_pressure"], 0.4375)
            self.assertAlmostEqual(low_congestion["route_disruption_reliability_pressure"], 0.25)
            self.assertGreater(low_congestion["route_pressure_congestion_share"], 0.60)
            self.assertGreater(low_congestion["route_pressure_balance"], 0.45)
            self.assertGreater(low_congestion["route_alt_pressure_imbalance"], 0.0)
            self.assertGreater(low_congestion["low_congestion_balance_dampening"], 0.0)
            self.assertLessEqual(low_congestion["low_congestion_adaptation_credit"], 0.033)
            self.assertLessEqual(
                low_congestion["dqn_local"],
                high_resilience["dqn_local"] + 0.006,
            )
        finally:
            env.close()

    def test_observed_mixed_route_pressure_supports_high_resilience_viability(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
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
            stressed = _calm_snapshot(env)
            low_congestion = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="low_congestion", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            high_resilience = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="high_resilience", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(high_resilience["route_pressure_balance"], 0.45)
            self.assertGreater(high_resilience["high_resilience_balance_support"], 0.0)
            self.assertGreater(high_resilience["high_resilience_adaptation_credit"], 0.025)
            self.assertGreater(high_resilience["dqn_local"], low_congestion["dqn_local"] - 0.006)
            self.assertLessEqual(
                high_resilience["route_adaptation_reward_or_credit"],
                low_congestion["route_adaptation_reward_or_credit"] + 0.02,
            )
        finally:
            env.close()

    def test_low_congestion_hold_route_label_is_neutralized_without_route_advantage(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
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
            healthy = _calm_snapshot(env)
            passive = env._dqn_local_reward_components(
                healthy,
                _discrete_info(dispatch="hold", route="low_congestion", dispatched_orders=0.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            useful_dispatch = env._dqn_local_reward_components(
                healthy,
                {
                    **_discrete_info(dispatch="dispatch", route="low_congestion", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "feasible_dispatch_ratio": 0.80,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(passive["route_adaptation_pressure"], 0.0)
            self.assertEqual(passive["hold_route_neutralized"], 1.0)
            self.assertEqual(passive["route_label_ignored_for_hold"], 1.0)
            self.assertEqual(passive["selected_route_effective_for_reward"], -1.0)
            self.assertEqual(passive["route_credit_blocked_hold"], 1.0)
            self.assertEqual(passive["candidate_alignment_blocked_hold"], 1.0)
            self.assertEqual(passive["low_congestion_adaptation_credit"], 0.0)
            self.assertEqual(passive["route_adaptation_reward_or_credit"], 0.0)
            self.assertEqual(passive["selected_route_candidate_score"], 0.0)
            self.assertEqual(passive["selected_route_score_gap"], 0.0)
            self.assertEqual(passive["route_candidate_alignment_credit"], 0.0)
            self.assertEqual(passive["route_candidate_mismatch_penalty"], 0.0)
            self.assertEqual(passive["route_candidate_useful_work_factor"], 0.0)
            self.assertEqual(useful_dispatch["hold_route_neutralized"], 0.0)
            self.assertEqual(useful_dispatch["route_label_ignored_for_hold"], 0.0)
            self.assertEqual(useful_dispatch["selected_route_effective_for_reward"], 1.0)
            self.assertGreater(useful_dispatch["low_congestion_adaptation_credit"], 0.0)
            self.assertGreater(useful_dispatch["selected_route_candidate_score"], 0.0)
            self.assertGreater(useful_dispatch["route_candidate_alignment_credit"], 0.0)
            self.assertEqual(useful_dispatch["passive_route_credit_dampening"], 0.0)
            self.assertAlmostEqual(useful_dispatch["route_credit_useful_work_factor"], 1.0)
        finally:
            env.close()

    def test_low_congestion_passive_route_credit_cannot_make_hold_locally_attractive(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
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
            healthy = _calm_snapshot(env)
            passive = env._dqn_local_reward_components(
                healthy,
                _discrete_info(dispatch="hold", route="low_congestion", dispatched_orders=0.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            useful_dispatch = env._dqn_local_reward_components(
                healthy,
                {
                    **_discrete_info(dispatch="dispatch", route="low_congestion", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "feasible_dispatch_ratio": 0.80,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertLessEqual(passive["dqn_local"], 0.0)
            self.assertEqual(passive["hold_route_neutralized"], 1.0)
            self.assertEqual(passive["low_congestion_adaptation_credit"], 0.0)
            self.assertEqual(passive["route_candidate_alignment_credit"], 0.0)
            self.assertGreater(useful_dispatch["dqn_local"], passive["dqn_local"])
        finally:
            env.close()

    def test_high_resilience_hold_route_label_is_neutralized_without_resilience_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
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
            healthy = _calm_snapshot(env)
            passive = env._dqn_local_reward_components(
                healthy,
                _discrete_info(dispatch="hold", route="high_resilience", dispatched_orders=0.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            useful_dispatch = env._dqn_local_reward_components(
                healthy,
                {
                    **_discrete_info(dispatch="dispatch", route="high_resilience", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "feasible_dispatch_ratio": 0.80,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(passive["route_adaptation_pressure"], 0.0)
            self.assertEqual(passive["hold_route_neutralized"], 1.0)
            self.assertEqual(passive["route_label_ignored_for_hold"], 1.0)
            self.assertEqual(passive["selected_route_effective_for_reward"], -1.0)
            self.assertEqual(passive["route_credit_blocked_hold"], 1.0)
            self.assertEqual(passive["candidate_alignment_blocked_hold"], 1.0)
            self.assertEqual(passive["route_resilience_credit"], 0.0)
            self.assertEqual(passive["high_resilience_adaptation_credit"], 0.0)
            self.assertEqual(passive["route_resilience_adaptation_credit"], 0.0)
            self.assertEqual(passive["route_adaptation_reward_or_credit"], 0.0)
            self.assertEqual(passive["route_candidate_alignment_credit"], 0.0)
            self.assertEqual(passive["selected_route_candidate_score"], 0.0)
            self.assertLessEqual(passive["dqn_local"], 0.0)
            self.assertGreater(useful_dispatch["dqn_local"], passive["dqn_local"])
            self.assertEqual(useful_dispatch["hold_route_neutralized"], 0.0)
            self.assertGreater(useful_dispatch["route_resilience_credit"], 0.0)
            self.assertGreater(useful_dispatch["high_resilience_adaptation_credit"], 0.0)
            self.assertEqual(useful_dispatch["high_resilience_hold_credit_dampening"], 0.0)
        finally:
            env.close()

    def test_moderate_pressure_shortest_relief_is_meaningful_but_bounded_for_safe_useful_dispatch(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "route_disruption_probability": 0.25,
                    "congestion_multiplier": 1.75,
                    "shortest_route_risk_multiplier": 1.5,
                    "traversal_cost_multiplier": 1.3,
                    "disruption_risk_multiplier": 1.5,
                }
            )
            healthy = _calm_snapshot(env)
            shortest = env._dqn_local_reward_components(
                healthy,
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", mode="primary_fleet", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            no_useful_work = env._dqn_local_reward_components(
                healthy,
                _discrete_info(dispatch="hold", route="shortest", dispatched_orders=0.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            action_24_like = env._dqn_local_reward_components(
                healthy,
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", mode="secondary_fleet", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(shortest["shortest_route_disruption_risk_penalty"], 0.0)
            self.assertEqual(shortest["shortest_relief_eligible"], 1.0)
            self.assertGreater(shortest["shortest_moderate_pressure_relief"], 0.020)
            self.assertLess(
                shortest["shortest_moderate_pressure_relief"],
                shortest["shortest_route_disruption_risk_penalty"] * 0.65,
            )
            self.assertLess(
                shortest["route_narrowness_penalty"],
                shortest["shortest_route_disruption_risk_penalty"],
            )
            self.assertGreater(shortest["route_narrowness_penalty"], 0.015)
            self.assertEqual(no_useful_work["hold_route_neutralized"], 1.0)
            self.assertEqual(no_useful_work["route_label_ignored_for_hold"], 1.0)
            self.assertEqual(no_useful_work["selected_route_effective_for_reward"], -1.0)
            self.assertEqual(no_useful_work["route_credit_blocked_hold"], 1.0)
            self.assertEqual(no_useful_work["candidate_alignment_blocked_hold"], 1.0)
            self.assertEqual(no_useful_work["shortest_route_disruption_risk_penalty"], 0.0)
            self.assertEqual(no_useful_work["shortest_moderate_pressure_relief"], 0.0)
            self.assertEqual(no_useful_work["shortest_relief_no_useful_work_blocked"], 0.0)
            self.assertEqual(no_useful_work["shortest_route_candidate_score"], 0.0)
            self.assertEqual(no_useful_work["selected_route_candidate_score"], 0.0)
            self.assertEqual(no_useful_work["route_candidate_alignment_credit"], 0.0)
            self.assertEqual(no_useful_work["selected_route_score_gap"], 0.0)
            _assert_shortest_secondary_telemetry(self, action_24_like)
            self.assertEqual(action_24_like["shortest_secondary_safe_useful_exception"], 1.0)
            self.assertEqual(action_24_like["shortest_secondary_guard_active"], 0.0)
            self.assertEqual(action_24_like["shortest_relief_brittle_secondary_blocked"], 0.0)
            self.assertGreater(action_24_like["shortest_moderate_pressure_relief"], 0.0)
            self.assertLess(
                action_24_like["shortest_moderate_pressure_relief"]
                + action_24_like["route_candidate_alignment_credit"],
                action_24_like["shortest_route_disruption_risk_penalty"],
            )
        finally:
            env.close()

    def test_observed_fixed_medium_candidate_scores_keep_safe_shortest_viable_with_bounded_24_25_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
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
            healthy = _calm_snapshot(env)
            safe_shortest = env._dqn_local_reward_components(
                healthy,
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", mode="primary_fleet", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "feasible_dispatch_ratio": 0.80,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            action_24_like = env._dqn_local_reward_components(
                healthy,
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", mode="secondary_fleet", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "feasible_dispatch_ratio": 0.80,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            low_congestion = env._dqn_local_reward_components(
                healthy,
                {
                    **_discrete_info(dispatch="dispatch", route="low_congestion", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "feasible_dispatch_ratio": 0.80,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            high_resilience = env._dqn_local_reward_components(
                healthy,
                {
                    **_discrete_info(dispatch="dispatch", route="high_resilience", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "feasible_dispatch_ratio": 0.80,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(safe_shortest["shortest_route_candidate_score"], 0.45)
            self.assertGreater(low_congestion["low_congestion_route_candidate_score"], 0.45)
            self.assertGreater(high_resilience["high_resilience_route_candidate_score"], 0.20)
            self.assertGreater(safe_shortest["route_candidate_alignment_credit"], 0.06)
            self.assertLessEqual(safe_shortest["selected_route_candidate_score_gap"], 0.30)
            _assert_shortest_secondary_telemetry(self, action_24_like)
            self.assertEqual(action_24_like["shortest_secondary_safe_useful_exception"], 1.0)
            self.assertEqual(action_24_like["selected_route_candidate_action_gate"], 1.0)
            self.assertEqual(action_24_like["shortest_relief_brittle_secondary_blocked"], 0.0)
            self.assertGreater(action_24_like["route_candidate_alignment_credit"], 0.0)
            self.assertLess(
                action_24_like["route_candidate_alignment_credit"],
                safe_shortest["route_candidate_alignment_credit"],
            )
            self.assertLess(
                action_24_like["shortest_moderate_pressure_relief"]
                + action_24_like["route_candidate_alignment_credit"],
                action_24_like["shortest_route_disruption_risk_penalty"],
            )
        finally:
            env.close()

    def test_observation_exposes_safe_shortest_secondary_context_before_reward_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _apply_observed_fixed_medium_route_stress(env)
            _make_orders_unassigned(env)
            _add_secondary_vehicles(env, count=4)

            observation_features = _observation_features(env)
            action_24_like = _shortest_secondary_components(env, reorder="none")

            self.assertGreater(observation_features["shortest_candidate_score"], 0.0)
            self.assertLessEqual(observation_features["shortest_candidate_score_gap"], 0.12)
            self.assertEqual(observation_features["shortest_candidate_near_best"], 1.0)
            self.assertGreater(observation_features["shortest_secondary_safe_context"], 0.0)
            self.assertLess(observation_features["shortest_secondary_brittle_risk"], 0.5)
            self.assertEqual(action_24_like["shortest_secondary_safe_useful_exception"], 1.0)
            self.assertEqual(action_24_like["shortest_secondary_guard_active"], 0.0)
            self.assertGreater(action_24_like["route_candidate_alignment_credit"], 0.0)
        finally:
            env.close()

    def test_observation_exposes_brittle_shortest_secondary_context_when_reward_guard_suppresses_24_25(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "route_disruption_probability": 0.70,
                    "congestion_multiplier": 3.5,
                    "shortest_route_risk_multiplier": 2.5,
                    "traversal_cost_multiplier": 2.0,
                    "disruption_risk_multiplier": 2.5,
                }
            )
            _make_orders_unassigned(env)
            _add_secondary_vehicles(env, count=4)

            observation_features = _observation_features(env)
            action_24_like = _shortest_secondary_components(env, reorder="none")

            self.assertGreater(observation_features["shortest_secondary_brittle_risk"], 0.5)
            self.assertEqual(observation_features["shortest_secondary_safe_context"], 0.0)
            self.assertEqual(action_24_like["shortest_secondary_guard_active"], 1.0)
            self.assertEqual(action_24_like["route_candidate_alignment_credit"], 0.0)
        finally:
            env.close()

    def test_candidate_alignment_closes_safe_shortest_gap_with_bounded_24_25_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
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
            healthy = _calm_snapshot(env)
            safe_shortest = env._dqn_local_reward_components(
                healthy,
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", mode="primary_fleet", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "feasible_dispatch_ratio": 0.80,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            low_congestion = env._dqn_local_reward_components(
                healthy,
                {
                    **_discrete_info(dispatch="dispatch", route="low_congestion", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "feasible_dispatch_ratio": 0.80,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            action_24_like = env._dqn_local_reward_components(
                healthy,
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", mode="secondary_fleet", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "feasible_dispatch_ratio": 0.80,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(safe_shortest["shortest_route_candidate_score"], low_congestion["low_congestion_route_candidate_score"])
            self.assertEqual(safe_shortest["candidate_alignment_eligible"], 1.0)
            self.assertGreater(safe_shortest["route_candidate_alignment_credit"], 0.105)
            self.assertGreaterEqual(safe_shortest["dqn_local"], low_congestion["dqn_local"] - 0.05)
            _assert_shortest_secondary_telemetry(self, action_24_like)
            self.assertEqual(action_24_like["candidate_alignment_blocked_brittle_secondary"], 0.0)
            self.assertEqual(action_24_like["action24_25_candidate_credit_allowed"], 1.0)
            self.assertGreater(action_24_like["route_candidate_alignment_credit"], 0.0)
            self.assertLess(
                action_24_like["route_candidate_alignment_credit"],
                safe_shortest["route_candidate_alignment_credit"],
            )
        finally:
            env.close()

    def test_best_low_and_high_routes_receive_bounded_candidate_alignment_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            low_pressure = {
                "route_disruption_probability": 0.05,
                "congestion_multiplier": 2.8,
                "shortest_route_risk_multiplier": 1.05,
                "traversal_cost_multiplier": 2.0,
                "disruption_risk_multiplier": 1.05,
            }
            high_pressure = {
                "route_disruption_probability": 0.50,
                "congestion_multiplier": 1.3,
                "shortest_route_risk_multiplier": 2.0,
                "traversal_cost_multiplier": 1.1,
                "disruption_risk_multiplier": 2.0,
            }
            env.scenario_stress_state.update(low_pressure)
            low_congestion = env._dqn_local_reward_components(
                _calm_snapshot(env),
                {
                    **_discrete_info(dispatch="dispatch", route="low_congestion", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "feasible_dispatch_ratio": 0.80,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            env.scenario_stress_state.clear()
            env.scenario_stress_state.update(high_pressure)
            high_resilience = env._dqn_local_reward_components(
                _calm_snapshot(env),
                {
                    **_discrete_info(dispatch="dispatch", route="high_resilience", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "feasible_dispatch_ratio": 0.80,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(low_congestion["candidate_alignment_eligible"], 1.0)
            self.assertGreater(low_congestion["route_candidate_alignment_credit"], 0.0)
            self.assertEqual(low_congestion["route_candidate_mismatch_penalty"], 0.0)
            self.assertEqual(high_resilience["candidate_alignment_eligible"], 1.0)
            self.assertGreater(high_resilience["route_candidate_alignment_credit"], 0.0)
            self.assertEqual(high_resilience["route_candidate_mismatch_penalty"], 0.0)
        finally:
            env.close()

    def test_passive_low_congestion_action8_ignores_route_label_and_inflight_completion(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
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
            healthy = _calm_snapshot(env)
            action_8_no_delivery = env._dqn_local_reward_components(
                healthy,
                _discrete_info(
                    dispatch="hold",
                    route="low_congestion",
                    mode="secondary_fleet",
                    reorder="none",
                    dispatched_orders=0.0,
                ),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            action_8_inflight_completion = env._dqn_local_reward_components(
                healthy,
                _discrete_info(
                    dispatch="hold",
                    route="low_congestion",
                    mode="secondary_fleet",
                    reorder="none",
                    dispatched_orders=0.0,
                ),
                delivered_delta=2,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            for components in (action_8_no_delivery, action_8_inflight_completion):
                self.assertEqual(components["hold_route_neutralized"], 1.0)
                self.assertEqual(components["route_label_ignored_for_hold"], 1.0)
                self.assertEqual(components["selected_route_effective_for_reward"], -1.0)
                self.assertEqual(components["route_credit_blocked_hold"], 1.0)
                self.assertEqual(components["candidate_alignment_blocked_hold"], 1.0)
                self.assertEqual(components["candidate_alignment_eligible"], 0.0)
                self.assertEqual(components["route_candidate_mismatch_penalty"], 0.0)
                self.assertEqual(components["route_candidate_alignment_credit"], 0.0)
                self.assertEqual(components["selected_route_candidate_score"], 0.0)
                self.assertEqual(components["route_candidate_useful_work_factor"], 0.0)
                self.assertEqual(components["low_congestion_adaptation_credit"], 0.0)
                self.assertEqual(components["dqn_delivery_credit"], 0.0)
                self.assertLessEqual(components["dqn_local"], 0.0)
            self.assertEqual(action_8_no_delivery["hold_inflight_completion_credit_blocked_for_route"], 0.0)
            self.assertEqual(action_8_inflight_completion["hold_inflight_completion_credit_blocked_for_route"], 1.0)
            self.assertLessEqual(
                action_8_inflight_completion["dqn_local"],
                action_8_no_delivery["dqn_local"] + 1e-9,
            )
        finally:
            env.close()

    def test_dispatch_without_current_work_does_not_get_inflight_delivery_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            no_current_work = env._dqn_local_reward_components(
                _calm_snapshot(env),
                {
                    **_discrete_info(
                        dispatch="dispatch",
                        route="shortest",
                        mode="secondary_fleet",
                        reorder="none",
                        dispatched_orders=0.0,
                    ),
                    "dispatch_success_count": 0.0,
                    "dispatch_success_rate": 0.0,
                    "dispatch_no_unassigned_orders": 1.0,
                    "dispatch_already_assigned_count": 2.0,
                    "already_assigned_rate": 1.0,
                    "dispatchable_order_count": 0.0,
                    "feasible_dispatch_ratio": 0.0,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=2,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(no_current_work["dispatch_feasibility_credit"], 0.0)
            self.assertEqual(no_current_work["dispatch_progress_credit"], 0.0)
            self.assertEqual(no_current_work["route_candidate_useful_work_factor"], 0.0)
            self.assertEqual(no_current_work["shortest_relief_useful_work_factor"], 0.0)
            self.assertEqual(no_current_work["dqn_delivery_credit"], 0.0)
            self.assertLessEqual(no_current_work["dqn_local"], 0.0)
        finally:
            env.close()

    def test_valid_dispatch_success_still_gets_delivery_and_progress_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_urgent_and_aged(env)
            valid_dispatch = env._dqn_local_reward_components(
                env.simulation.snapshot(),
                {
                    **_discrete_info(
                        dispatch="dispatch",
                        route="low_congestion",
                        mode="secondary_fleet",
                        reorder="none",
                        dispatched_orders=2.0,
                    ),
                    "dispatch_success_count": 2.0,
                    "dispatch_success_rate": 1.0,
                    "feasible_dispatch_ratio": 0.90,
                    "already_assigned_rate": 0.0,
                    "dispatch_already_assigned_count": 0.0,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=2,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(valid_dispatch["dispatch_feasibility_credit"], 0.0)
            self.assertGreater(valid_dispatch["dispatch_progress_credit"], 0.0)
            self.assertGreater(valid_dispatch["route_candidate_useful_work_factor"], 0.0)
            self.assertGreater(valid_dispatch["dqn_delivery_credit"], 0.0)
        finally:
            env.close()

    def test_observed_fixed_medium_shortest_relief_is_meaningful_but_partial(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
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
            healthy = _calm_snapshot(env)
            safe_shortest = env._dqn_local_reward_components(
                healthy,
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", mode="primary_fleet", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            action_24_like = env._dqn_local_reward_components(
                healthy,
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", mode="secondary_fleet", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            blocked_shortest = env._dqn_local_reward_components(
                healthy,
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", mode="primary_fleet", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "macro_dispatch_budget": 1.0,
                    "blocked": True,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertAlmostEqual(safe_shortest["route_congestion_cost_pressure"], 0.4375)
            self.assertAlmostEqual(safe_shortest["route_disruption_reliability_pressure"], 0.25)
            self.assertGreater(safe_shortest["shortest_route_disruption_risk_penalty"], 0.0)
            self.assertEqual(safe_shortest["shortest_relief_eligible"], 1.0)
            self.assertIn("shortest_relief_observed_profile_support", safe_shortest)
            self.assertGreater(safe_shortest["shortest_relief_observed_profile_support"], 0.30)
            self.assertGreater(safe_shortest["shortest_relief_pressure_factor"], 0.30)
            self.assertGreater(
                safe_shortest["shortest_moderate_pressure_relief"],
                safe_shortest["shortest_route_disruption_risk_penalty"] * 0.18,
            )
            self.assertLess(
                safe_shortest["shortest_moderate_pressure_relief"],
                safe_shortest["shortest_route_disruption_risk_penalty"] * 0.45,
            )
            self.assertGreater(
                safe_shortest["route_narrowness_penalty"],
                safe_shortest["shortest_route_disruption_risk_penalty"] * 0.50,
            )
            _assert_shortest_secondary_telemetry(self, action_24_like)
            self.assertEqual(action_24_like["shortest_secondary_safe_useful_exception"], 1.0)
            self.assertEqual(action_24_like["shortest_secondary_guard_active"], 0.0)
            self.assertGreater(action_24_like["shortest_moderate_pressure_relief"], 0.0)
            self.assertLess(
                action_24_like["shortest_moderate_pressure_relief"]
                + action_24_like["route_candidate_alignment_credit"],
                action_24_like["shortest_route_disruption_risk_penalty"],
            )
            self.assertEqual(blocked_shortest["shortest_moderate_pressure_relief"], 0.0)
            self.assertEqual(blocked_shortest["shortest_relief_impossible_dispatch_blocked"], 1.0)
            self.assertGreater(action_24_like["route_candidate_alignment_credit"], 0.0)
            self.assertEqual(blocked_shortest["route_candidate_alignment_credit"], 0.0)
        finally:
            env.close()

    def test_safe_fixed_medium_action_24_receives_bounded_shortest_secondary_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _apply_observed_fixed_medium_route_stress(env)
            components = _shortest_secondary_components(env, reorder="none")
            _assert_shortest_secondary_telemetry(self, components)

            self.assertGreater(components["shortest_route_candidate_score"], 0.45)
            self.assertLessEqual(components["selected_route_candidate_score_gap"], 0.30)
            self.assertEqual(components["shortest_secondary_safe_useful_exception"], 1.0)
            self.assertEqual(components["shortest_secondary_brittle_risk"], 0.0)
            self.assertEqual(components["shortest_secondary_guard_active"], 0.0)
            self.assertEqual(components["shortest_secondary_guard_relaxed_safe"], 1.0)
            self.assertEqual(components["action24_25_candidate_credit_allowed"], 1.0)
            self.assertEqual(components["action24_25_candidate_credit_blocked_reason"], 0.0)
            self.assertEqual(components["candidate_alignment_eligible"], 1.0)
            self.assertGreater(components["route_candidate_alignment_credit"], 0.0)
            self.assertEqual(components["shortest_relief_eligible"], 1.0)
            self.assertGreater(components["shortest_moderate_pressure_relief"], 0.0)
            self.assertLess(
                components["shortest_moderate_pressure_relief"]
                + components["route_candidate_alignment_credit"],
                components["shortest_route_disruption_risk_penalty"],
            )
            self.assertGreater(components["route_narrowness_penalty"], 0.0)
        finally:
            env.close()

    def test_safe_fixed_medium_action_25_receives_bounded_shortest_secondary_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _apply_observed_fixed_medium_route_stress(env)
            components = _shortest_secondary_components(env, reorder="conservative")
            _assert_shortest_secondary_telemetry(self, components)

            self.assertGreater(components["shortest_route_candidate_score"], 0.45)
            self.assertLessEqual(components["selected_route_candidate_score_gap"], 0.30)
            self.assertEqual(components["shortest_secondary_safe_useful_exception"], 1.0)
            self.assertEqual(components["shortest_secondary_guard_active"], 0.0)
            self.assertEqual(components["action24_25_candidate_credit_allowed"], 1.0)
            self.assertEqual(components["candidate_alignment_eligible"], 1.0)
            self.assertGreater(components["route_candidate_alignment_credit"], 0.0)
            self.assertGreater(components["shortest_moderate_pressure_relief"], 0.0)
            self.assertLess(
                components["shortest_moderate_pressure_relief"]
                + components["route_candidate_alignment_credit"],
                components["shortest_route_disruption_risk_penalty"],
            )
        finally:
            env.close()

    def test_action_24_25_shortest_secondary_credit_remains_blocked_under_brittle_conditions(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            cases = []

            env.scenario_stress_state.update(
                {
                    "route_disruption_probability": 0.75,
                    "congestion_multiplier": 3.0,
                    "shortest_route_risk_multiplier": 2.5,
                    "traversal_cost_multiplier": 1.8,
                    "disruption_risk_multiplier": 2.0,
                }
            )
            cases.append(("severe", _shortest_secondary_components(env, reorder="none")))

            env.scenario_stress_state.clear()
            _apply_observed_fixed_medium_route_stress(env)
            cases.extend(
                [
                    (
                        "route_failure",
                        _shortest_secondary_components(
                            env,
                            reorder="none",
                            projection_overrides={
                                "dispatch_route_failure": 1.0,
                                "dispatch_success_count": 0.0,
                            },
                            delivered_delta=0,
                        ),
                    ),
                    (
                        "customer_revisit",
                        _shortest_secondary_components(
                            env,
                            reorder="none",
                            projection_overrides={"customer_revisit_blocked_count": 1.0},
                        ),
                    ),
                    (
                        "no_useful_work",
                        _shortest_secondary_components(
                            env,
                            reorder="none",
                            projection_overrides={
                                "dispatch_success_count": 0.0,
                                "dispatch_success_rate": 0.0,
                                "feasible_dispatch_ratio": 0.0,
                            },
                            dispatched_orders=0.0,
                            delivered_delta=0,
                        ),
                    ),
                    (
                        "blocked",
                        _shortest_secondary_components(
                            env,
                            reorder="conservative",
                            projection_overrides={"blocked": True, "discrete_blocked": True},
                        ),
                    ),
                    (
                        "no_vehicle",
                        _shortest_secondary_components(
                            env,
                            reorder="none",
                            projection_overrides={
                                "dispatch_no_vehicle_available": 1.0,
                                "no_vehicle_available_rate": 1.0,
                            },
                        ),
                    ),
                    (
                        "already_assigned_saturation",
                        _shortest_secondary_components(
                            env,
                            reorder="conservative",
                            projection_overrides={
                                "already_assigned_rate": 0.90,
                                "dispatch_already_assigned_count": 8.0,
                            },
                        ),
                    ),
                ]
            )

            for name, components in cases:
                with self.subTest(name=name):
                    _assert_shortest_secondary_telemetry(self, components)
                    self.assertEqual(components["shortest_secondary_safe_useful_exception"], 0.0)
                    self.assertEqual(components["shortest_secondary_brittle_risk"], 1.0)
                    self.assertEqual(components["shortest_secondary_guard_active"], 1.0)
                    self.assertEqual(components["shortest_secondary_guard_relaxed_safe"], 0.0)
                    self.assertEqual(components["action24_25_candidate_credit_allowed"], 0.0)
                    self.assertGreater(components["action24_25_candidate_credit_blocked_reason"], 0.0)
                    self.assertEqual(components["route_candidate_alignment_credit"], 0.0)
                    self.assertEqual(components["shortest_moderate_pressure_relief"], 0.0)
        finally:
            env.close()

    def test_shortest_relief_is_inactive_under_severe_route_risk_or_route_failure(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "route_disruption_probability": 0.75,
                    "congestion_multiplier": 3.0,
                    "shortest_route_risk_multiplier": 2.5,
                    "traversal_cost_multiplier": 1.8,
                    "disruption_risk_multiplier": 2.0,
                }
            )
            stressed = _calm_snapshot(env)
            severe_shortest = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="hold", route="shortest", dispatched_orders=0.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            failed_shortest = env._dqn_local_reward_components(
                stressed,
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", dispatched_orders=0.0),
                    "dispatch_route_failure": 1.0,
                    "dispatch_success_count": 0.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(severe_shortest["hold_route_neutralized"], 1.0)
            self.assertEqual(severe_shortest["route_label_ignored_for_hold"], 1.0)
            self.assertEqual(severe_shortest["selected_route_effective_for_reward"], -1.0)
            self.assertEqual(severe_shortest["route_credit_blocked_hold"], 1.0)
            self.assertEqual(severe_shortest["candidate_alignment_blocked_hold"], 1.0)
            self.assertEqual(severe_shortest["shortest_moderate_pressure_relief"], 0.0)
            self.assertEqual(severe_shortest["shortest_relief_severe_pressure_blocked"], 0.0)
            self.assertEqual(failed_shortest["shortest_moderate_pressure_relief"], 0.0)
            self.assertEqual(failed_shortest["shortest_relief_route_failure_blocked"], 1.0)
            self.assertEqual(severe_shortest["route_narrowness_penalty"], 0.0)
            self.assertEqual(severe_shortest["route_candidate_alignment_credit"], 0.0)
            self.assertEqual(failed_shortest["route_candidate_alignment_credit"], 0.0)
            self.assertEqual(failed_shortest["selected_route_candidate_action_gate"], 0.0)
        finally:
            env.close()

    def test_shortest_relief_is_blocked_on_customer_revisit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "route_disruption_probability": 0.25,
                    "congestion_multiplier": 1.75,
                    "shortest_route_risk_multiplier": 1.5,
                    "traversal_cost_multiplier": 1.3,
                    "disruption_risk_multiplier": 1.5,
                }
            )
            revisit_blocked = env._dqn_local_reward_components(
                _calm_snapshot(env),
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", mode="primary_fleet", dispatched_orders=1.0),
                    "customer_revisit_blocked_count": 1.0,
                    "dispatch_success_count": 1.0,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(revisit_blocked["shortest_moderate_pressure_relief"], 0.0)
            self.assertEqual(revisit_blocked["shortest_relief_customer_revisit_blocked"], 1.0)
            self.assertGreater(revisit_blocked["route_narrowness_penalty"], 0.0)
            self.assertEqual(revisit_blocked["route_candidate_alignment_credit"], 0.0)
        finally:
            env.close()

    def test_route_candidate_alignment_credit_is_suppressed_by_service_lateness_degradation(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
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
            degraded = _calm_snapshot(env)
            degraded["service_level"] = 0.60
            now = float(degraded["time"])
            for order in degraded["orders"].values():
                if isinstance(order, dict):
                    order["status"] = ShipmentStatus.CREATED.value
                    order["delivery_time"] = None
                    order["assigned_vehicle_id"] = None
                    order["due_time"] = now - 28_800.0

            components = env._dqn_local_reward_components(
                degraded,
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", mode="primary_fleet", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "feasible_dispatch_ratio": 0.80,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(components["demand_operational_degradation"], 0.0)
            self.assertLess(components["route_candidate_operational_health_factor"], 0.50)
            self.assertEqual(components["selected_route_candidate_action_gate"], 0.0)
            self.assertEqual(components["route_candidate_alignment_credit"], 0.0)
        finally:
            env.close()

    def test_congestion_dominant_route_pressure_prefers_low_congestion(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "route_disruption_probability": 0.05,
                    "congestion_multiplier": 2.8,
                    "shortest_route_risk_multiplier": 1.05,
                    "traversal_cost_multiplier": 2.0,
                    "disruption_risk_multiplier": 1.05,
                }
            )
            stressed = _calm_snapshot(env)
            low_congestion = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="low_congestion", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            high_resilience = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="high_resilience", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(
                low_congestion["route_congestion_cost_pressure"],
                high_resilience["route_disruption_reliability_pressure"],
            )
            self.assertGreater(
                low_congestion["low_congestion_adaptation_credit"],
                high_resilience["high_resilience_adaptation_credit"],
            )
            self.assertGreater(low_congestion["dqn_local"], high_resilience["dqn_local"])
        finally:
            env.close()

    def test_strong_route_disruption_still_justifies_high_resilience(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "route_disruption_probability": 0.75,
                    "congestion_multiplier": 3.0,
                    "shortest_route_risk_multiplier": 2.5,
                    "traversal_cost_multiplier": 1.8,
                    "disruption_risk_multiplier": 2.0,
                }
            )
            stressed = _calm_snapshot(env)
            low_congestion = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="low_congestion", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            high_resilience = env._dqn_local_reward_components(
                stressed,
                _discrete_info(dispatch="dispatch", route="high_resilience", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(high_resilience["high_resilience_adaptation_credit"], 0.0)
            self.assertGreater(
                high_resilience["route_adaptation_reward_or_credit"],
                low_congestion["route_adaptation_reward_or_credit"],
            )
            self.assertGreater(high_resilience["dqn_local"], low_congestion["dqn_local"])
        finally:
            env.close()

    def test_route_failure_suppresses_route_adaptation_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "route_disruption_probability": 0.25,
                    "congestion_multiplier": 1.75,
                    "shortest_route_risk_multiplier": 1.5,
                    "traversal_cost_multiplier": 1.3,
                    "disruption_risk_multiplier": 1.5,
                }
            )
            failed_dispatch = {
                **_discrete_info(dispatch="dispatch", route="high_resilience", dispatched_orders=0.0),
                "dispatch_route_failure": 1.0,
                "dispatch_success_count": 0.0,
            }

            components = env._dqn_local_reward_components(
                _calm_snapshot(env),
                failed_dispatch,
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(components["route_adaptation_reward_or_credit"], 0.0)
            self.assertEqual(components["route_failure_credit_suppression"], 1.0)
            self.assertEqual(components["dispatch_progress_credit"], 0.0)
            self.assertEqual(components["dispatch_feasibility_credit"], 0.0)
        finally:
            env.close()

    def test_no_work_high_resilience_dispatch_gets_no_route_resilience_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "route_disruption_probability": 0.75,
                    "congestion_multiplier": 3.0,
                    "shortest_route_risk_multiplier": 2.5,
                    "traversal_cost_multiplier": 1.8,
                    "disruption_risk_multiplier": 2.0,
                }
            )
            no_work_dispatch = {
                **_discrete_info(
                    dispatch="dispatch",
                    route="high_resilience",
                    mode="secondary_fleet",
                    reorder="conservative",
                    dispatched_orders=0.0,
                ),
                "dispatch_success_count": 0.0,
                "dispatch_no_unassigned_orders": 1.0,
                "macro_dispatch_budget": 2.0,
            }

            components = env._dqn_local_reward_components(
                _calm_snapshot(env),
                no_work_dispatch,
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(components["current_dispatch_work_signal"], 0.0)
            self.assertGreater(components["no_current_or_unassigned_dispatch_penalty"], 0.0)
            self.assertEqual(components["route_candidate_useful_work_factor"], 0.0)
            self.assertEqual(components["route_resilience_credit"], 0.0)
            self.assertEqual(components["high_resilience_adaptation_credit"], 0.0)
            self.assertEqual(components["route_resilience_adaptation_credit"], 0.0)
            self.assertEqual(components["route_adaptation_reward_or_credit"], 0.0)
            self.assertEqual(components["route_candidate_alignment_credit"], 0.0)
        finally:
            env.close()

    def test_route_failure_actions_41_and_45_block_no_work_resilience_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "route_disruption_probability": 0.75,
                    "congestion_multiplier": 3.0,
                    "shortest_route_risk_multiplier": 2.5,
                    "traversal_cost_multiplier": 1.8,
                    "disruption_risk_multiplier": 2.0,
                }
            )
            for action_id, expected_mode in (
                (41, ModeDecision.SECONDARY_FLEET),
                (45, ModeDecision.PRIMARY_FLEET),
            ):
                with self.subTest(action_id=action_id):
                    action = env.discrete_mapper.map(action_id)
                    self.assertEqual(action.dispatch, DispatchDecision.DISPATCH)
                    self.assertEqual(action.route, RouteDecision.HIGH_RESILIENCE)
                    self.assertEqual(action.mode, expected_mode)
                    self.assertEqual(action.reorder, ReorderDecision.CONSERVATIVE)

                    components = env._dqn_local_reward_components(
                        _calm_snapshot(env),
                        {
                            **_discrete_info(**action.as_dict(), dispatched_orders=0.0),
                            "dispatch_success_count": 0.0,
                            "dispatch_no_unassigned_orders": 1.0,
                            "macro_dispatch_budget": 2.0,
                        },
                        delivered_delta=0,
                        cost_delta=0.0,
                        masked_order_ids=set(),
                    )

                    self.assertEqual(components["current_dispatch_work_signal"], 0.0)
                    self.assertGreater(components["no_current_or_unassigned_dispatch_penalty"], 0.0)
                    self.assertEqual(components["route_candidate_useful_work_factor"], 0.0)
                    self.assertEqual(components["high_resilience_no_useful_work_credit_blocked"], 1.0)
                    self.assertEqual(components["route_resilience_credit"], 0.0)
                    self.assertEqual(components["high_resilience_adaptation_credit"], 0.0)
                    self.assertEqual(components["route_resilience_adaptation_credit"], 0.0)
                    self.assertEqual(components["route_adaptation_reward_or_credit"], 0.0)
                    self.assertEqual(components["route_candidate_alignment_credit"], 0.0)
        finally:
            env.close()

    def test_useful_route_actions_41_and_45_keep_high_resilience_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "route_disruption_probability": 0.75,
                    "congestion_multiplier": 3.0,
                    "shortest_route_risk_multiplier": 2.5,
                    "traversal_cost_multiplier": 1.8,
                    "disruption_risk_multiplier": 2.0,
                }
            )
            stressed = _calm_snapshot(env)
            for action_id in (41, 45):
                with self.subTest(action_id=action_id):
                    action = env.discrete_mapper.map(action_id)
                    components = env._dqn_local_reward_components(
                        stressed,
                        {
                            **_discrete_info(**action.as_dict(), dispatched_orders=1.0),
                            "dispatch_success_count": 1.0,
                            "dispatch_success_rate": 1.0,
                            "feasible_dispatch_ratio": 1.0,
                            "macro_dispatch_budget": 1.0,
                        },
                        delivered_delta=0,
                        cost_delta=0.0,
                        masked_order_ids=set(),
                    )

                    self.assertGreater(components["route_candidate_useful_work_factor"], 0.0)
                    self.assertEqual(components["high_resilience_no_useful_work_credit_blocked"], 0.0)
                    self.assertGreater(components["route_resilience_credit"], 0.0)
                    self.assertGreater(components["high_resilience_adaptation_credit"], 0.0)
                    self.assertGreater(components["route_resilience_adaptation_credit"], 0.0)
                    self.assertGreater(components["route_adaptation_reward_or_credit"], 0.0)
        finally:
            env.close()

    def test_no_work_action32_low_congestion_dispatch_blocks_route_adaptation_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _apply_observed_fixed_medium_route_stress(env)
            action = env.discrete_mapper.map(32)
            self.assertEqual(action.dispatch, DispatchDecision.DISPATCH)
            self.assertEqual(action.route, RouteDecision.LOW_CONGESTION)
            self.assertEqual(action.mode, ModeDecision.SECONDARY_FLEET)
            self.assertEqual(action.reorder, ReorderDecision.NONE)

            components = env._dqn_local_reward_components(
                _late_dispatch_pressure_snapshot(env),
                {
                    **_discrete_info(**action.as_dict(), dispatched_orders=0.0),
                    "dispatch_success_count": 0.0,
                    "dispatch_success_rate": 0.0,
                    "dispatch_no_unassigned_orders": 1.0,
                    "feasible_dispatch_ratio": 0.0,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(components["route_adaptation_pressure"], 0.0)
            self.assertEqual(components["current_dispatch_work_signal"], 0.0)
            self.assertGreater(components["no_current_or_unassigned_dispatch_penalty"], 0.0)
            self.assertEqual(components["route_candidate_useful_work_factor"], 0.0)
            self.assertEqual(components["candidate_alignment_blocked_no_useful_work"], 1.0)
            self.assertEqual(components["low_congestion_no_useful_work_credit_blocked"], 1.0)
            self.assertEqual(components["low_congestion_adaptation_credit"], 0.0)
            self.assertEqual(components["route_resilience_adaptation_credit"], 0.0)
            self.assertEqual(components["route_adaptation_reward_or_credit"], 0.0)
            self.assertEqual(components["route_candidate_alignment_credit"], 0.0)
        finally:
            env.close()

    def test_no_work_low_congestion_blocker_reports_when_route_failure_zeroes_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _apply_observed_fixed_medium_route_stress(env)
            action = env.discrete_mapper.map(32)

            components = env._dqn_local_reward_components(
                _late_dispatch_pressure_snapshot(env),
                {
                    **_discrete_info(**action.as_dict(), dispatched_orders=0.0),
                    "dispatch_success_count": 0.0,
                    "dispatch_success_rate": 0.0,
                    "dispatch_no_unassigned_orders": 1.0,
                    "dispatch_route_failure": 1.0,
                    "feasible_dispatch_ratio": 0.0,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(components["route_adaptation_pressure"], 0.0)
            self.assertEqual(components["route_failure_credit_suppression"], 1.0)
            self.assertEqual(components["route_candidate_useful_work_factor"], 0.0)
            self.assertEqual(components["low_congestion_no_useful_work_credit_blocked"], 1.0)
            self.assertEqual(components["low_congestion_adaptation_credit"], 0.0)
            self.assertEqual(components["route_resilience_adaptation_credit"], 0.0)
            self.assertEqual(components["route_adaptation_reward_or_credit"], 0.0)
        finally:
            env.close()

    def test_useful_action32_low_congestion_dispatch_keeps_route_adaptation_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _apply_observed_fixed_medium_route_stress(env)
            action = env.discrete_mapper.map(32)

            components = env._dqn_local_reward_components(
                _late_dispatch_pressure_snapshot(env),
                {
                    **_discrete_info(**action.as_dict(), dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "dispatch_success_rate": 1.0,
                    "feasible_dispatch_ratio": 1.0,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(components["current_dispatch_work_signal"], 1.0)
            self.assertGreater(components["route_candidate_useful_work_factor"], 0.0)
            self.assertEqual(components["low_congestion_no_useful_work_credit_blocked"], 0.0)
            self.assertGreater(components["low_congestion_adaptation_credit"], 0.0)
            self.assertGreater(components["route_resilience_adaptation_credit"], 0.0)
            self.assertGreater(components["route_adaptation_reward_or_credit"], 0.0)
        finally:
            env.close()

    def test_shortest_route_remains_reasonable_when_cheapest_and_low_risk(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            calm = _calm_snapshot(env)
            shortest = env._dqn_local_reward_components(
                calm,
                _discrete_info(dispatch="dispatch", route="shortest", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            resilient = env._dqn_local_reward_components(
                calm,
                _discrete_info(dispatch="dispatch", route="high_resilience", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            low_congestion = env._dqn_local_reward_components(
                calm,
                _discrete_info(dispatch="dispatch", route="low_congestion", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(shortest["shortest_route_disruption_risk_penalty"], 0.0)
            self.assertEqual(shortest["route_adaptation_pressure"], 0.0)
            self.assertEqual(resilient["route_adaptation_reward_or_credit"], 0.0)
            self.assertEqual(low_congestion["route_adaptation_reward_or_credit"], 0.0)
            self.assertEqual(low_congestion["low_congestion_balance_dampening"], 0.0)
            self.assertEqual(low_congestion["low_congestion_hold_credit_dampening"], 0.0)
            self.assertEqual(low_congestion["passive_route_credit_dampening"], 0.0)
            self.assertEqual(low_congestion["route_credit_useful_work_factor"], 1.0)
            self.assertEqual(shortest["shortest_moderate_pressure_relief"], 0.0)
            self.assertEqual(shortest["shortest_relief_eligible"], 0.0)
            self.assertEqual(shortest["shortest_relief_pressure_factor"], 0.0)
            self.assertEqual(resilient["high_resilience_balance_support"], 0.0)
            self.assertEqual(shortest["route_candidate_alignment_credit"], 0.0)
            self.assertEqual(shortest["shortest_route_candidate_score"], 0.0)
            self.assertEqual(resilient["high_resilience_hold_credit_dampening"], 0.0)
            self.assertGreater(shortest["dqn_local"], resilient["dqn_local"])
            self.assertGreater(shortest["dqn_local"], low_congestion["dqn_local"])
        finally:
            env.close()

    def test_calm_hold_route_neutralization_has_no_normal_v4_route_leakage(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            components = env._dqn_local_reward_components(
                _calm_snapshot(env),
                _discrete_info(dispatch="hold", route="high_resilience", dispatched_orders=0.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(components["route_adaptation_pressure"], 0.0)
            self.assertEqual(components["hold_route_neutralized"], 1.0)
            self.assertEqual(components["route_label_ignored_for_hold"], 1.0)
            self.assertEqual(components["selected_route_effective_for_reward"], -1.0)
            self.assertEqual(components["route_credit_blocked_hold"], 1.0)
            self.assertEqual(components["candidate_alignment_blocked_hold"], 1.0)
            self.assertEqual(components["route_adaptation_reward_or_credit"], 0.0)
            self.assertEqual(components["route_candidate_alignment_credit"], 0.0)
            self.assertEqual(components["selected_route_candidate_score"], 0.0)
            self.assertEqual(components["selected_route_score_gap"], 0.0)
        finally:
            env.close()

    def test_primary_or_premium_fleet_is_justified_only_under_urgent_premium_pressure(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            calm = _calm_snapshot(env)
            calm_secondary = env._dqn_local_reward_components(
                calm,
                _discrete_info(dispatch="dispatch", mode="secondary_fleet", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            calm_primary = env._dqn_local_reward_components(
                calm,
                _discrete_info(dispatch="dispatch", mode="primary_fleet", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            env.scenario_stress_state.update(
                {
                    "premium_sla_ratio": 0.80,
                    "service_target": 0.99,
                    "urgent_due_window_multiplier": 0.50,
                    "premium_lateness_penalty_multiplier": 2.5,
                }
            )
            _make_orders_urgent_and_aged(env)
            premium = env.simulation.snapshot()
            premium_secondary = env._dqn_local_reward_components(
                premium,
                _discrete_info(dispatch="dispatch", mode="secondary_fleet", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            premium_primary = env._dqn_local_reward_components(
                premium,
                _discrete_info(dispatch="dispatch", mode="primary_fleet", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(calm_primary["primary_fleet_cost_penalty"], 0.0)
            self.assertEqual(calm_primary["premium_sla_fleet_justification_credit"], 0.0)
            self.assertLess(calm_primary["dqn_local"], calm_secondary["dqn_local"])
            self.assertGreater(premium_primary["premium_sla_pressure"], 0.0)
            self.assertLess(premium_primary["primary_fleet_cost_penalty"], calm_primary["primary_fleet_cost_penalty"])
            self.assertGreater(premium_primary["premium_sla_fleet_justification_credit"], 0.0)
            self.assertGreater(premium_primary["premium_primary_adaptation_credit"], 0.0)
            self.assertGreater(premium_primary["dqn_local"], premium_secondary["dqn_local"])
        finally:
            env.close()

    def test_hold_primary_fleet_premium_credit_is_blocked_without_dispatch_work(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "premium_sla_ratio": 0.80,
                    "service_target": 0.99,
                    "urgent_due_window_multiplier": 0.50,
                    "premium_lateness_penalty_multiplier": 2.5,
                }
            )
            _make_orders_urgent_and_aged(env)
            premium = env.simulation.snapshot()

            hold_primary = env._dqn_local_reward_components(
                premium,
                _discrete_info(dispatch="hold", route="high_resilience", mode="primary_fleet", dispatched_orders=0.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(hold_primary["premium_sla_pressure"], 0.0)
            self.assertEqual(hold_primary["current_dispatch_work_count"], 0.0)
            self.assertEqual(hold_primary["current_dispatch_work_signal"], 0.0)
            self.assertEqual(hold_primary["premium_sla_fleet_justification_credit"], 0.0)
            self.assertEqual(hold_primary["premium_primary_adaptation_credit"], 0.0)
            self.assertEqual(hold_primary["premium_fleet_credit_allowed"], 0.0)
            self.assertEqual(hold_primary["premium_sla_fleet_credit_blocked_no_current_work"], 1.0)
            self.assertEqual(hold_primary["premium_primary_adaptation_credit_blocked_no_current_work"], 1.0)
            self.assertLessEqual(hold_primary["dqn_local"], 0.0)
        finally:
            env.close()

    def test_dispatch_primary_fleet_premium_credit_requires_real_dispatch_work(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "premium_sla_ratio": 0.80,
                    "service_target": 0.99,
                    "urgent_due_window_multiplier": 0.50,
                    "premium_lateness_penalty_multiplier": 2.5,
                }
            )
            _make_orders_urgent_and_aged(env)
            premium = env.simulation.snapshot()

            zero_work = env._dqn_local_reward_components(
                premium,
                {
                    **_discrete_info(dispatch="dispatch", mode="primary_fleet", dispatched_orders=0.0),
                    "dispatch_success_count": 0.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            useful_dispatch = env._dqn_local_reward_components(
                premium,
                {
                    **_discrete_info(dispatch="dispatch", mode="primary_fleet", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(zero_work["current_dispatch_work_count"], 0.0)
            self.assertEqual(zero_work["current_dispatch_work_signal"], 0.0)
            self.assertEqual(zero_work["premium_sla_fleet_justification_credit"], 0.0)
            self.assertEqual(zero_work["premium_primary_adaptation_credit"], 0.0)
            self.assertEqual(zero_work["premium_fleet_credit_allowed"], 0.0)
            self.assertEqual(zero_work["premium_sla_fleet_credit_blocked_no_current_work"], 1.0)
            self.assertEqual(zero_work["premium_primary_adaptation_credit_blocked_no_current_work"], 1.0)
            self.assertGreater(zero_work["no_current_or_unassigned_dispatch_penalty"], 0.0)

            self.assertGreater(useful_dispatch["current_dispatch_work_count"], 0.0)
            self.assertEqual(useful_dispatch["current_dispatch_work_signal"], 1.0)
            self.assertEqual(useful_dispatch["premium_fleet_credit_allowed"], 1.0)
            self.assertEqual(useful_dispatch["premium_sla_fleet_credit_blocked_no_current_work"], 0.0)
            self.assertEqual(useful_dispatch["premium_primary_adaptation_credit_blocked_no_current_work"], 0.0)
            self.assertEqual(useful_dispatch["no_current_or_unassigned_dispatch_penalty"], 0.0)
            self.assertGreater(useful_dispatch["premium_sla_fleet_justification_credit"], 0.0)
            self.assertGreater(useful_dispatch["premium_primary_adaptation_credit"], 0.0)
        finally:
            env.close()

    def test_useful_dispatch_under_lateness_pressure_gets_bounded_urgency_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            late = _late_dispatch_pressure_snapshot(env)
            useful_dispatch = env._dqn_local_reward_components(
                late,
                {
                    **_discrete_info(
                        dispatch="dispatch",
                        route="low_congestion",
                        mode="secondary_fleet",
                        dispatched_orders=2.0,
                    ),
                    "dispatch_success_count": 2.0,
                    "dispatch_success_rate": 1.0,
                    "feasible_dispatch_ratio": 0.90,
                    "already_assigned_rate": 0.0,
                    "dispatch_already_assigned_count": 0.0,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(useful_dispatch["true_lateness_pressure"], 0.0)
            self.assertEqual(useful_dispatch["current_dispatch_work_signal"], 1.0)
            self.assertEqual(useful_dispatch["no_current_or_unassigned_dispatch_penalty"], 0.0)
            self.assertIn("useful_dispatch_lateness_urgency_credit", useful_dispatch)
            self.assertGreater(useful_dispatch["useful_dispatch_lateness_urgency_credit"], 0.0)
            self.assertLessEqual(useful_dispatch["useful_dispatch_lateness_urgency_credit"], 0.12)
        finally:
            env.close()

    def test_no_work_dispatch_gets_no_urgency_credit_and_preserves_route_hard_gate(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _apply_observed_fixed_medium_route_stress(env)
            late = _late_dispatch_pressure_snapshot(env)
            no_work = env._dqn_local_reward_components(
                late,
                {
                    **_discrete_info(
                        dispatch="dispatch",
                        route="low_congestion",
                        mode="secondary_fleet",
                        dispatched_orders=0.0,
                    ),
                    "dispatch_success_count": 0.0,
                    "dispatch_success_rate": 0.0,
                    "dispatch_no_unassigned_orders": 1.0,
                    "feasible_dispatch_ratio": 0.0,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(no_work["route_adaptation_pressure"], 0.0)
            self.assertEqual(no_work["current_dispatch_work_signal"], 0.0)
            self.assertGreater(no_work["no_current_or_unassigned_dispatch_penalty"], 0.0)
            self.assertEqual(no_work["route_candidate_alignment_credit"], 0.0)
            self.assertEqual(no_work["candidate_alignment_blocked_no_useful_work"], 1.0)
            self.assertIn("useful_dispatch_lateness_urgency_credit", no_work)
            self.assertEqual(no_work["useful_dispatch_lateness_urgency_credit"], 0.0)
        finally:
            env.close()

    def test_premium_useful_primary_dispatch_gets_bounded_timing_justification(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "premium_sla_ratio": 0.80,
                    "service_target": 0.99,
                    "urgent_due_window_multiplier": 0.50,
                    "premium_lateness_penalty_multiplier": 2.5,
                }
            )
            late = _late_dispatch_pressure_snapshot(env)
            useful_primary = env._dqn_local_reward_components(
                late,
                {
                    **_discrete_info(
                        dispatch="dispatch",
                        route="low_congestion",
                        mode="primary_fleet",
                        dispatched_orders=2.0,
                    ),
                    "dispatch_success_count": 2.0,
                    "dispatch_success_rate": 1.0,
                    "feasible_dispatch_ratio": 0.90,
                    "already_assigned_rate": 0.0,
                    "dispatch_already_assigned_count": 0.0,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(useful_primary["premium_sla_pressure"], 0.0)
            self.assertEqual(useful_primary["current_dispatch_work_signal"], 1.0)
            self.assertEqual(useful_primary["premium_fleet_credit_allowed"], 1.0)
            self.assertIn("primary_fleet_timing_justification_credit", useful_primary)
            self.assertGreater(useful_primary["primary_fleet_timing_justification_credit"], 0.0)
            self.assertLessEqual(useful_primary["primary_fleet_timing_justification_credit"], 0.08)
        finally:
            env.close()

    def test_no_work_hold_primary_gets_no_urgency_or_timing_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "premium_sla_ratio": 0.80,
                    "service_target": 0.99,
                    "urgent_due_window_multiplier": 0.50,
                    "premium_lateness_penalty_multiplier": 2.5,
                }
            )
            late = _late_dispatch_pressure_snapshot(env)
            hold_primary = env._dqn_local_reward_components(
                late,
                _discrete_info(
                    dispatch="hold",
                    route="high_resilience",
                    mode="primary_fleet",
                    dispatched_orders=0.0,
                ),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(hold_primary["premium_sla_pressure"], 0.0)
            self.assertEqual(hold_primary["current_dispatch_work_signal"], 0.0)
            self.assertEqual(hold_primary["premium_sla_fleet_justification_credit"], 0.0)
            self.assertEqual(hold_primary["premium_primary_adaptation_credit"], 0.0)
            self.assertEqual(hold_primary["premium_fleet_credit_allowed"], 0.0)
            self.assertEqual(hold_primary["premium_sla_fleet_credit_blocked_no_current_work"], 1.0)
            self.assertIn("useful_dispatch_lateness_urgency_credit", hold_primary)
            self.assertIn("primary_fleet_timing_justification_credit", hold_primary)
            self.assertIn("hold_under_lateness_pressure", hold_primary)
            self.assertEqual(hold_primary["useful_dispatch_lateness_urgency_credit"], 0.0)
            self.assertEqual(hold_primary["primary_fleet_timing_justification_credit"], 0.0)
            self.assertGreater(hold_primary["hold_under_lateness_pressure"], 0.0)
        finally:
            env.close()

    def test_hold_under_lateness_pressure_gets_bounded_penalty_only_when_work_is_useful(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "premium_sla_ratio": 0.80,
                    "service_target": 0.99,
                    "urgent_due_window_multiplier": 0.50,
                    "premium_lateness_penalty_multiplier": 2.5,
                }
            )
            late = _late_dispatch_pressure_snapshot(env)
            pressure_hold = env._dqn_local_reward_components(
                late,
                _discrete_info(
                    dispatch="hold",
                    route="high_resilience",
                    mode="secondary_fleet",
                    dispatched_orders=0.0,
                ),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            calm_hold = env._dqn_local_reward_components(
                _calm_snapshot(env),
                _discrete_info(
                    dispatch="hold",
                    route="high_resilience",
                    mode="secondary_fleet",
                    dispatched_orders=0.0,
                ),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(pressure_hold["hold_under_lateness_pressure"], 0.0)
            self.assertIn("hold_under_lateness_pressure_penalty", pressure_hold)
            self.assertGreater(pressure_hold["hold_under_lateness_pressure_penalty"], 0.0)
            self.assertLessEqual(pressure_hold["hold_under_lateness_pressure_penalty"], 0.16)
            self.assertEqual(calm_hold["hold_under_lateness_pressure"], 0.0)
            self.assertEqual(calm_hold["hold_under_lateness_pressure_penalty"], 0.0)
        finally:
            env.close()

    def test_severe_timing_pressure_hold_penalty_exceeds_legacy_cap_without_affecting_calm_hold(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "premium_sla_ratio": 0.80,
                    "service_target": 0.99,
                    "urgent_due_window_multiplier": 0.50,
                    "premium_lateness_penalty_multiplier": 2.5,
                    "holding_cost_multiplier": 2.25,
                    "lead_time_mean_multiplier": 1.8,
                    "lead_time_variance_multiplier": 3.0,
                }
            )
            late = _late_dispatch_pressure_snapshot(env)
            pressure_hold = env._dqn_local_reward_components(
                late,
                _discrete_info(
                    dispatch="hold",
                    route="low_congestion",
                    mode="secondary_fleet",
                    dispatched_orders=0.0,
                ),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            calm_hold = env._dqn_local_reward_components(
                _calm_snapshot(env),
                _discrete_info(
                    dispatch="hold",
                    route="low_congestion",
                    mode="secondary_fleet",
                    dispatched_orders=0.0,
                ),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(pressure_hold["hold_under_lateness_pressure"], 0.0)
            self.assertGreater(pressure_hold["hold_under_lateness_pressure_penalty"], 0.10)
            self.assertLessEqual(pressure_hold["hold_under_lateness_pressure_penalty"], 0.16)
            self.assertEqual(calm_hold["hold_under_lateness_pressure"], 0.0)
            self.assertEqual(calm_hold["hold_under_lateness_pressure_penalty"], 0.0)
        finally:
            env.close()

    def test_secondary_fleet_premium_underuse_penalty_requires_useful_dispatch_work(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "premium_sla_ratio": 0.80,
                    "service_target": 0.99,
                    "urgent_due_window_multiplier": 0.50,
                    "premium_lateness_penalty_multiplier": 2.5,
                }
            )
            late = _late_dispatch_pressure_snapshot(env)
            hold_secondary = env._dqn_local_reward_components(
                late,
                _discrete_info(
                    dispatch="hold",
                    route="low_congestion",
                    mode="secondary_fleet",
                    dispatched_orders=0.0,
                ),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            no_work_dispatch = env._dqn_local_reward_components(
                late,
                {
                    **_discrete_info(
                        dispatch="dispatch",
                        route="low_congestion",
                        mode="secondary_fleet",
                        dispatched_orders=0.0,
                    ),
                    "dispatch_success_count": 0.0,
                    "dispatch_no_unassigned_orders": 1.0,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            useful_secondary = env._dqn_local_reward_components(
                late,
                {
                    **_discrete_info(
                        dispatch="dispatch",
                        route="low_congestion",
                        mode="secondary_fleet",
                        dispatched_orders=2.0,
                    ),
                    "dispatch_success_count": 2.0,
                    "dispatch_success_rate": 1.0,
                    "feasible_dispatch_ratio": 0.90,
                    "already_assigned_rate": 0.0,
                    "dispatch_already_assigned_count": 0.0,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(hold_secondary["current_dispatch_work_signal"], 0.0)
            self.assertEqual(no_work_dispatch["current_dispatch_work_signal"], 0.0)
            self.assertEqual(hold_secondary["premium_primary_underuse_penalty"], 0.0)
            self.assertEqual(no_work_dispatch["premium_primary_underuse_penalty"], 0.0)
            self.assertGreater(no_work_dispatch["no_current_or_unassigned_dispatch_penalty"], 0.0)
            self.assertEqual(useful_secondary["current_dispatch_work_signal"], 1.0)
            self.assertGreater(useful_secondary["premium_primary_underuse_penalty"], 0.0)
        finally:
            env.close()

    def test_fixed_medium_premium_pressure_can_overcome_primary_fleet_penalty_when_due_soon(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "premium_sla_ratio": 0.40,
                    "urgent_due_window_multiplier": 0.75,
                    "premium_lateness_penalty_multiplier": 1.5,
                }
            )
            _make_orders_urgent_and_aged(env)
            premium = env.simulation.snapshot()
            secondary = env._dqn_local_reward_components(
                premium,
                _discrete_info(dispatch="dispatch", mode="secondary_fleet", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            primary = env._dqn_local_reward_components(
                premium,
                _discrete_info(dispatch="dispatch", mode="primary_fleet", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(primary["premium_sla_pressure"], 0.0)
            self.assertGreater(primary["premium_primary_adaptation_credit"], 0.0)
            self.assertGreater(secondary["premium_primary_underuse_penalty"], 0.0)
            self.assertLess(primary["primary_fleet_cost_penalty"], primary["premium_primary_adaptation_credit"])
            self.assertGreater(primary["dqn_local"], secondary["dqn_local"])
        finally:
            env.close()

    def test_route_fleet_repair_preserves_dispatch_reward_protections(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            env.scenario_stress_state.update(
                {
                    "route_disruption_probability": 0.75,
                    "congestion_multiplier": 3.0,
                    "shortest_route_risk_multiplier": 2.5,
                    "premium_sla_ratio": 0.80,
                }
            )
            _make_orders_unassigned(env)
            for order in env.simulation.orders.values():
                for line in order.lines:
                    line.quantity_units = 220.0
            action = DiscreteLogisticsAction(
                dispatch=DispatchDecision.DISPATCH,
                route=RouteDecision.SHORTEST,
                mode=ModeDecision.PRIMARY_FLEET,
                reorder=ReorderDecision.NONE,
            )

            projection_info = {
                "action": action.as_dict(),
                **env._apply_discrete_action(action, macro_dispatch_release=1.0, macro_dispatch_budget=5),
            }
            components = env._dqn_local_reward_components(
                env.simulation.snapshot(),
                projection_info,
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(projection_info["dqn_dispatched_orders"], 0.0)
            self.assertEqual(projection_info["dispatch_success_count"], 0.0)
            self.assertGreater(projection_info["dispatch_route_failure"], 0.0)
            self.assertEqual(components["dispatch_progress_credit"], 0.0)
            self.assertEqual(components["dispatch_feasibility_credit"], 0.0)
            self.assertEqual(components["successful_dispatch_count_capped"], 0.0)
        finally:
            env.close()

    def test_dqn_emergency_reorder_is_less_bad_when_stockout_risk_is_severe(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_low_risk(env)
            low_risk = env.simulation.snapshot()
            low_risk_components = env._dqn_local_reward_components(
                low_risk,
                _discrete_info(reorder="emergency", override_units=10.0, override_useful_units=10.0, override_cost=30.0),
                delivered_delta=0,
                cost_delta=30.0,
                masked_order_ids=set(),
            )

            _make_inventory_high_risk(env)
            high_risk = env.simulation.snapshot()
            high_risk_components = env._dqn_local_reward_components(
                high_risk,
                _discrete_info(reorder="emergency", override_units=10.0, override_useful_units=10.0, override_cost=30.0),
                delivered_delta=0,
                cost_delta=30.0,
                masked_order_ids=set(),
            )

            self.assertLess(low_risk_components["emergency_gate"], 0.10)
            self.assertGreater(high_risk_components["emergency_gate"], 0.90)
            self.assertGreater(high_risk_components["dqn_local"], low_risk_components["dqn_local"])
        finally:
            env.close()

    def test_emergency_procurement_cost_scales_dqn_penalty(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_low_risk(env)
            low_risk = env.simulation.snapshot()
            cheap = env._dqn_local_reward_components(
                low_risk,
                _discrete_info(reorder="emergency", override_units=10.0, override_useful_units=10.0, override_cost=5.0),
                delivered_delta=0,
                cost_delta=5.0,
                masked_order_ids=set(),
            )
            expensive = env._dqn_local_reward_components(
                low_risk,
                _discrete_info(reorder="emergency", override_units=10.0, override_useful_units=10.0, override_cost=80.0),
                delivered_delta=0,
                cost_delta=80.0,
                masked_order_ids=set(),
            )

            self.assertGreater(expensive["emergency_procurement_penalty"], cheap["emergency_procurement_penalty"])
            self.assertLess(expensive["dqn_local"], cheap["dqn_local"])
        finally:
            env.close()

    def test_low_risk_emergency_procurement_is_net_negative_with_zero_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_low_risk(env)
            low_risk = env.simulation.snapshot()
            components = env._dqn_local_reward_components(
                low_risk,
                _discrete_info(reorder="emergency", override_units=10.0, override_useful_units=10.0, override_cost=5.0),
                delivered_delta=0,
                cost_delta=5.0,
                masked_order_ids=set(),
            )

            self.assertEqual(components["emergency_gate"], 0.0)
            self.assertEqual(components["emergency_procurement_justification_credit"], 0.0)
            self.assertGreater(components["emergency_procurement_penalty"], 0.0)
            self.assertLess(components["dqn_local"], 0.0)
        finally:
            env.close()

    def test_medium_risk_emergency_procurement_is_still_costly(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_medium_risk(env)
            medium_risk = env.simulation.snapshot()
            components = env._dqn_local_reward_components(
                medium_risk,
                _discrete_info(reorder="emergency", override_units=10.0, override_useful_units=10.0, override_cost=30.0),
                delivered_delta=0,
                cost_delta=30.0,
                masked_order_ids=set(),
            )

            self.assertGreater(components["emergency_gate"], 0.0)
            self.assertLess(components["emergency_gate"], 1.0)
            self.assertGreater(
                components["emergency_procurement_penalty"],
                components["emergency_procurement_justification_credit"],
            )
        finally:
            env.close()

    def test_high_risk_emergency_procurement_can_be_positive_only_with_useful_units(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_high_risk(env)
            high_risk = env.simulation.snapshot()
            useful = env._dqn_local_reward_components(
                high_risk,
                _discrete_info(reorder="emergency", override_units=10.0, override_useful_units=10.0, override_cost=5.0),
                delivered_delta=0,
                cost_delta=5.0,
                masked_order_ids=set(),
            )
            zero_useful = env._dqn_local_reward_components(
                high_risk,
                _discrete_info(reorder="emergency", override_units=10.0, override_useful_units=0.0, override_cost=5.0),
                delivered_delta=0,
                cost_delta=5.0,
                masked_order_ids=set(),
            )

            self.assertGreater(useful["emergency_gate"], 0.90)
            self.assertGreater(useful["emergency_procurement_justification_credit"], 0.0)
            self.assertEqual(zero_useful["emergency_procurement_justification_credit"], 0.0)
            self.assertGreater(useful["dqn_local"], zero_useful["dqn_local"])
        finally:
            env.close()

    def test_dqn_hold_penalty_is_zero_without_pre_crisis_pressure(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            snapshot = _calm_snapshot(env)
            components = env._dqn_local_reward_components(
                snapshot,
                _discrete_info(dispatch="hold", reorder="none"),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(components["pre_crisis_pressure"], 0.0)
            self.assertEqual(components["hold_gate"], 0.0)
            self.assertEqual(components["hold_penalty"], 0.0)
        finally:
            env.close()

    def test_dqn_hold_penalty_rises_when_pre_crisis_pressure_is_visible(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_urgent_and_aged(env)
            snapshot = env.simulation.snapshot()
            hold = env._dqn_local_reward_components(
                snapshot,
                _discrete_info(dispatch="hold", reorder="none"),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            dispatch = env._dqn_local_reward_components(
                snapshot,
                _discrete_info(dispatch="dispatch", reorder="none", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(hold["pre_crisis_pressure"], 0.25)
            self.assertGreater(hold["hold_gate"], 0.0)
            self.assertGreater(hold["hold_penalty"], 0.0)
            self.assertEqual(dispatch["hold_penalty"], 0.0)
        finally:
            env.close()

    def test_emergency_delay_abuse_penalty_reduces_late_heroic_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_inventory_high_risk(env)
            _make_orders_urgent_and_aged(env)
            high_risk = env.simulation.snapshot()

            env._previous_inaction_pressure = 0.0
            clean_emergency = env._dqn_local_reward_components(
                high_risk,
                _discrete_info(reorder="emergency", override_units=10.0, override_useful_units=10.0, override_cost=5.0),
                delivered_delta=0,
                cost_delta=5.0,
                masked_order_ids=set(),
            )

            env._previous_inaction_pressure = 1.0
            delayed_emergency = env._dqn_local_reward_components(
                high_risk,
                _discrete_info(reorder="emergency", override_units=10.0, override_useful_units=10.0, override_cost=5.0),
                delivered_delta=0,
                cost_delta=5.0,
                masked_order_ids=set(),
            )

            self.assertGreater(delayed_emergency["delay_abuse_penalty"], 0.0)
            self.assertLess(delayed_emergency["dqn_local"], clean_emergency["dqn_local"])
        finally:
            env.close()

    def test_dqn_dispatch_credit_requires_actual_dispatch(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            snapshot = env.simulation.snapshot()
            no_dispatch = env._dqn_local_reward_components(
                snapshot,
                _discrete_info(dispatch="dispatch", dispatched_orders=0.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            actual_dispatch = env._dqn_local_reward_components(
                snapshot,
                _discrete_info(dispatch="dispatch", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(no_dispatch["dispatch_feasibility_credit"], 0.0)
            self.assertGreater(actual_dispatch["dispatch_feasibility_credit"], 0.0)
            self.assertGreater(actual_dispatch["dqn_local"], no_dispatch["dqn_local"])
        finally:
            env.close()

    def test_dispatch_feasibility_telemetry_reports_no_vehicle_opportunity(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_unassigned(env)
            for vehicle in env.simulation.vehicles.values():
                vehicle.active = False

            result = env._apply_discrete_action(env.discrete_mapper.map(24))

            self.assertEqual(result["dqn_dispatched_orders"], 0.0)
            self.assertGreater(result["dispatchable_order_count"], 0.0)
            self.assertEqual(result["available_vehicle_count"], 0.0)
            self.assertEqual(result["feasible_dispatch_opportunity"], 0.0)
            self.assertEqual(result["feasible_dispatch_ratio"], 0.0)
            self.assertEqual(result["vehicle_availability_pressure"], 1.0)
            self.assertEqual(result["infeasible_dispatch_attempt"], 1.0)
            self.assertEqual(result["repeated_infeasible_dispatch_attempt_rate"], 1.0)
            self.assertEqual(result["dispatch_success_rate"], 0.0)
            self.assertGreater(result["no_vehicle_available_rate"], 0.0)
        finally:
            env.close()

    def test_dispatch_feasibility_telemetry_reports_feasible_opportunity(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_unassigned(env)
            _add_secondary_vehicles(env, count=5)

            result = env._apply_discrete_action(env.discrete_mapper.map(24))

            self.assertGreater(result["available_vehicle_count"], 0.0)
            self.assertGreater(result["dispatchable_order_count"], 0.0)
            self.assertEqual(result["feasible_dispatch_opportunity"], 1.0)
            self.assertGreater(result["feasible_dispatch_ratio"], 0.0)
            self.assertGreater(result["dispatch_success_count"], 0.0)
            self.assertGreater(result["dispatch_success_rate"], 0.0)
            self.assertEqual(result["infeasible_dispatch_attempt"], 0.0)
        finally:
            env.close()

    def test_passive_dispatch_feasibility_telemetry_does_not_change_dqn_reward(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_urgent_and_aged(env)
            snapshot = env.simulation.snapshot()
            base_info = _discrete_info(dispatch="dispatch", dispatched_orders=0.0)
            telemetry_info = {
                **base_info,
                "available_vehicle_count": 0.0,
                "dispatchable_order_count": 12.0,
                "feasible_dispatch_opportunity": 0.0,
                "feasible_dispatch_ratio": 0.0,
                "vehicle_availability_pressure": 1.0,
                "infeasible_dispatch_attempt": 0.0,
                "repeated_infeasible_dispatch_attempt_rate": 0.0,
                "dispatch_success_rate": 0.0,
                "no_vehicle_available_rate": 1.0,
                "already_assigned_rate": 0.0,
                "raw_macro_dispatch_budget": 5.0,
                "feasible_macro_dispatch_cap": 1.0,
                "macro_dispatch_budget": 1.0,
            }

            base = env._dqn_local_reward_components(
                snapshot,
                base_info,
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            with_telemetry = env._dqn_local_reward_components(
                snapshot,
                telemetry_info,
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(with_telemetry["dqn_local"], base["dqn_local"])
            self.assertEqual(with_telemetry["infeasible_dispatch_penalty"], 0.0)
            self.assertEqual(
                with_telemetry["dispatch_progress_credit"],
                base["dispatch_progress_credit"],
            )
            self.assertEqual(
                with_telemetry["unnecessary_dispatch_penalty"],
                base["unnecessary_dispatch_penalty"],
            )
        finally:
            env.close()

    def test_repeated_infeasible_dispatch_receives_small_penalty(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_urgent_and_aged(env)
            snapshot = env.simulation.snapshot()
            projection_info = {
                **_discrete_info(dispatch="dispatch", dispatched_orders=0.0),
                "infeasible_dispatch_attempt": 1.0,
                "repeated_infeasible_dispatch_attempt_rate": 1.0,
                "vehicle_availability_pressure": 1.0,
                "no_vehicle_available_rate": 1.0,
                "dispatch_success_rate": 0.0,
            }

            components = env._dqn_local_reward_components(
                snapshot,
                projection_info,
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(components["infeasible_dispatch_penalty"], 0.0)
            self.assertLessEqual(
                components["infeasible_dispatch_penalty"],
                env.config.infeasible_dispatch_penalty_weight,
            )
        finally:
            env.close()

    def test_one_off_infeasible_dispatch_is_not_over_punished(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
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
            first_attempt_info = env._apply_discrete_action(action)
            first_attempt_info["action"] = action.as_dict()
            one_off = env._dqn_local_reward_components(
                snapshot,
                first_attempt_info,
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            second_attempt_info = env._apply_discrete_action(action)
            second_attempt_info["action"] = action.as_dict()
            repeated = env._dqn_local_reward_components(
                snapshot,
                second_attempt_info,
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(one_off["infeasible_dispatch_penalty"], 0.0)
            self.assertGreater(repeated["infeasible_dispatch_penalty"], one_off["infeasible_dispatch_penalty"])
        finally:
            env.close()

    def test_valid_feasible_dispatch_has_progress_credit_and_no_infeasible_penalty(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_urgent_and_aged(env)
            snapshot = env.simulation.snapshot()
            components = env._dqn_local_reward_components(
                snapshot,
                {
                    **_discrete_info(dispatch="dispatch", dispatched_orders=2.0),
                    "macro_dispatch_budget": 2.0,
                    "infeasible_dispatch_attempt": 0.0,
                    "repeated_infeasible_dispatch_attempt_rate": 0.0,
                    "vehicle_availability_pressure": 0.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(components["dispatch_progress_credit"], 0.0)
            self.assertEqual(components["infeasible_dispatch_penalty"], 0.0)
        finally:
            env.close()

    def test_high_demand_projected_already_assigned_dispatch_is_penalized(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_urgent_and_aged(env)
            snapshot = env.simulation.snapshot()
            poor_dispatch = env._dqn_local_reward_components(
                snapshot,
                {
                    **_discrete_info(dispatch="dispatch", dispatched_orders=2.0),
                    "dispatch_success_count": 2.0,
                    "dispatch_success_rate": 0.95,
                    "feasible_dispatch_ratio": 0.05,
                    "already_assigned_rate": 0.80,
                    "dispatch_already_assigned_count": 8.0,
                    "projected": True,
                    "discrete_projected": True,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            useful_dispatch = env._dqn_local_reward_components(
                snapshot,
                {
                    **_discrete_info(dispatch="dispatch", dispatched_orders=2.0),
                    "dispatch_success_count": 2.0,
                    "dispatch_success_rate": 0.95,
                    "feasible_dispatch_ratio": 0.80,
                    "already_assigned_rate": 0.0,
                    "dispatch_already_assigned_count": 0.0,
                    "projected": False,
                    "discrete_projected": False,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=2,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(poor_dispatch["demand_dispatch_quality_penalty"], 0.0)
            self.assertLess(poor_dispatch["dispatch_progress_credit"], useful_dispatch["dispatch_progress_credit"])
            self.assertLess(poor_dispatch["dqn_local"], useful_dispatch["dqn_local"])
        finally:
            env.close()

    def test_high_dispatch_success_cannot_mask_service_lateness_collapse(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_urgent_and_aged(env)
            collapsed = env.simulation.snapshot()
            collapsed["service_level"] = 0.60
            components = env._dqn_local_reward_components(
                collapsed,
                {
                    **_discrete_info(dispatch="dispatch", dispatched_orders=2.0),
                    "dispatch_success_count": 2.0,
                    "dispatch_success_rate": 0.96,
                    "feasible_dispatch_ratio": 0.035,
                    "already_assigned_rate": 0.50,
                    "dispatch_already_assigned_count": 4.0,
                    "projected": True,
                    "discrete_projected": True,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(components["demand_dispatch_quality_penalty"], components["dispatch_progress_credit"])
            self.assertLessEqual(components["dqn_local"], 0.0)
        finally:
            env.close()

    def test_demand_collapse_gates_dispatch_feasibility_credit(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_urgent_and_aged(env)
            collapsed = env.simulation.snapshot()
            collapsed["service_level"] = 0.60
            poor_dispatch = env._dqn_local_reward_components(
                collapsed,
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", mode="secondary_fleet", dispatched_orders=2.0),
                    "dispatch_success_count": 2.0,
                    "dispatch_success_rate": 0.96,
                    "feasible_dispatch_ratio": 0.03,
                    "already_assigned_rate": 0.25,
                    "dispatch_already_assigned_count": 2.0,
                    "projected": False,
                    "discrete_projected": False,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            useful_dispatch = env._dqn_local_reward_components(
                collapsed,
                {
                    **_discrete_info(dispatch="dispatch", route="low_congestion", mode="secondary_fleet", dispatched_orders=2.0),
                    "dispatch_success_count": 2.0,
                    "dispatch_success_rate": 1.0,
                    "feasible_dispatch_ratio": 0.80,
                    "already_assigned_rate": 0.0,
                    "dispatch_already_assigned_count": 0.0,
                    "projected": False,
                    "discrete_projected": False,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=2,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(poor_dispatch["demand_service_collapse_penalty"], 0.0)
            self.assertLess(
                poor_dispatch["dispatch_feasibility_credit"],
                useful_dispatch["dispatch_feasibility_credit"],
            )
            self.assertLess(poor_dispatch["dqn_local"], -0.25)
            self.assertGreater(useful_dispatch["dqn_local"], 0.0)
        finally:
            env.close()

    def test_action_24_25_family_is_penalized_under_degraded_demand(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_urgent_and_aged(env)
            collapsed = env.simulation.snapshot()
            collapsed["service_level"] = 0.60
            action_24_like = env._dqn_local_reward_components(
                collapsed,
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", mode="secondary_fleet", reorder="none", dispatched_orders=2.0),
                    "dispatch_success_count": 2.0,
                    "dispatch_success_rate": 0.96,
                    "feasible_dispatch_ratio": 0.03,
                    "already_assigned_rate": 0.25,
                    "dispatch_already_assigned_count": 2.0,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            action_25_like = env._dqn_local_reward_components(
                collapsed,
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", mode="secondary_fleet", reorder="conservative", dispatched_orders=2.0),
                    "dispatch_success_count": 2.0,
                    "dispatch_success_rate": 0.96,
                    "feasible_dispatch_ratio": 0.03,
                    "already_assigned_rate": 0.25,
                    "dispatch_already_assigned_count": 2.0,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            resilient_family = env._dqn_local_reward_components(
                collapsed,
                {
                    **_discrete_info(dispatch="dispatch", route="low_congestion", mode="secondary_fleet", reorder="none", dispatched_orders=2.0),
                    "dispatch_success_count": 2.0,
                    "dispatch_success_rate": 0.96,
                    "feasible_dispatch_ratio": 0.03,
                    "already_assigned_rate": 0.25,
                    "dispatch_already_assigned_count": 2.0,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(action_24_like["demand_action_family_penalty"], 0.0)
            self.assertGreater(action_25_like["demand_action_family_penalty"], 0.0)
            self.assertEqual(resilient_family["demand_action_family_penalty"], 0.0)
            self.assertLess(action_24_like["dqn_local"], resilient_family["dqn_local"])
            self.assertLess(action_25_like["dqn_local"], resilient_family["dqn_local"])
        finally:
            env.close()

    def test_action_25_family_penalty_activates_with_already_assigned_even_before_service_collapse(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_urgent_and_aged(env)
            stressed = env.simulation.snapshot()
            stressed["service_level"] = 0.94
            action_25_like = env._dqn_local_reward_components(
                stressed,
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", mode="secondary_fleet", reorder="conservative", dispatched_orders=2.0),
                    "dispatch_success_count": 2.0,
                    "dispatch_success_rate": 1.0,
                    "feasible_dispatch_ratio": 0.15,
                    "already_assigned_rate": 0.55,
                    "dispatch_already_assigned_count": 5.0,
                    "dispatch_no_vehicle_available": 1.0,
                    "no_vehicle_available_rate": 0.25,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            useful_alternative = env._dqn_local_reward_components(
                stressed,
                {
                    **_discrete_info(dispatch="dispatch", route="low_congestion", mode="secondary_fleet", reorder="conservative", dispatched_orders=2.0),
                    "dispatch_success_count": 2.0,
                    "dispatch_success_rate": 1.0,
                    "feasible_dispatch_ratio": 0.80,
                    "already_assigned_rate": 0.0,
                    "dispatch_already_assigned_count": 0.0,
                    "dispatch_no_vehicle_available": 0.0,
                    "no_vehicle_available_rate": 0.0,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=1,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertGreater(action_25_like["no_vehicle_exposure"], 0.0)
            self.assertGreater(action_25_like["action_family_concentration_proxy"], 0.0)
            self.assertGreater(action_25_like["demand_action_family_penalty"], 0.0)
            self.assertLess(action_25_like["dqn_local"], useful_alternative["dqn_local"])
        finally:
            env.close()

    def test_useful_high_demand_dispatch_is_not_hit_by_collapse_penalty(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_urgent_and_aged(env)
            stressed = env.simulation.snapshot()
            stressed["service_level"] = 0.60
            components = env._dqn_local_reward_components(
                stressed,
                {
                    **_discrete_info(dispatch="dispatch", route="shortest", mode="secondary_fleet", dispatched_orders=2.0),
                    "dispatch_success_count": 2.0,
                    "dispatch_success_rate": 1.0,
                    "feasible_dispatch_ratio": 0.85,
                    "already_assigned_rate": 0.0,
                    "dispatch_already_assigned_count": 0.0,
                    "macro_dispatch_budget": 2.0,
                },
                delivered_delta=2,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(components["demand_service_collapse_penalty"], 0.0)
            self.assertEqual(components["demand_action_family_penalty"], 0.0)
            self.assertEqual(components["action_family_concentration_proxy"], 0.0)
            self.assertGreater(components["dispatch_feasibility_credit"], 0.0)
            self.assertGreater(components["dqn_local"], 0.0)
        finally:
            env.close()

    def test_normal_dispatch_is_not_penalized_by_demand_quality_repair(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            _make_orders_unassigned(env)
            snapshot = env.simulation.snapshot()
            components = env._dqn_local_reward_components(
                snapshot,
                {
                    **_discrete_info(dispatch="dispatch", dispatched_orders=1.0),
                    "dispatch_success_count": 1.0,
                    "dispatch_success_rate": 1.0,
                    "feasible_dispatch_ratio": 0.90,
                    "already_assigned_rate": 0.0,
                    "projected": False,
                    "macro_dispatch_budget": 1.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(components["demand_dispatch_quality_penalty"], 0.0)
            self.assertGreater(components["dispatch_progress_credit"], 0.0)
        finally:
            env.close()

    def test_hold_is_not_rewarded_or_infeasible_penalized_without_feasible_dispatch(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            snapshot = _calm_snapshot(env)
            components = env._dqn_local_reward_components(
                snapshot,
                {
                    **_discrete_info(dispatch="hold", dispatched_orders=0.0),
                    "feasible_dispatch_opportunity": 0.0,
                    "infeasible_dispatch_attempt": 0.0,
                    "repeated_infeasible_dispatch_attempt_rate": 1.0,
                },
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertEqual(components["dispatch_progress_credit"], 0.0)
            self.assertEqual(components["infeasible_dispatch_penalty"], 0.0)
            self.assertLessEqual(components["dqn_local"], 0.0)
        finally:
            env.close()

    def test_speed_is_not_justified_by_nonurgent_pending_orders_alone(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            snapshot = env.simulation.snapshot()
            now = float(snapshot["time"])
            for order in snapshot["orders"].values():
                if isinstance(order, dict):
                    order["status"] = "pending"
                    order["due_time"] = now + 86_400.0
            for row in snapshot["inventory"].values():
                if isinstance(row, dict):
                    row["backlog_units"] = 0.0
                    row["backlog_pressure"] = 0.0
            snapshot["disruption_score"] = 0.0

            components = env._ppo_local_reward_components(
                snapshot,
                _continuous_info(speed=1.35),
                cost_delta=0.0,
                speed_cost_delta=0.0,
            )

            self.assertEqual(components["speed_justification_credit"], 0.0)
            self.assertGreater(components["unjustified_speed_penalty"], 0.0)
        finally:
            env.close()

    def test_low_congestion_route_penalty_is_reduced_by_actual_congestion(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            calm = _calm_snapshot(env)
            no_congestion = env._dqn_local_reward_components(
                calm,
                _discrete_info(dispatch="dispatch", route="low_congestion", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )
            first_arc = next(iter(env.simulation.route_network.arcs))
            env.simulation.routing_physics.set_congestion(first_arc[0], first_arc[1], 0.90)
            congested = env._dqn_local_reward_components(
                calm,
                _discrete_info(dispatch="dispatch", route="low_congestion", dispatched_orders=1.0),
                delivered_delta=0,
                cost_delta=0.0,
                masked_order_ids=set(),
            )

            self.assertLess(congested["route_premium_penalty"], no_congestion["route_premium_penalty"])
            self.assertGreater(congested["dqn_local"], no_congestion["dqn_local"])
        finally:
            env.close()


def _continuous_info(
    *,
    reorder: float = 0.0,
    speed: float = 1.0,
    safety_stock: float = 1.0,
    capacity_buffer: float = 0.0,
    planned_units: float | None = None,
    useful_units: float | None = None,
    planned_cost: float | None = None,
) -> dict[str, Any]:
    actual_planned_units = 100.0 * reorder if planned_units is None else planned_units
    actual_useful_units = actual_planned_units if useful_units is None else useful_units
    actual_planned_cost = 5.0 * reorder if planned_cost is None else planned_cost
    return {
        "action": {
            "speed_multiplier": speed,
            "reorder_fraction": reorder,
            "dispatch_intensity": 0.0,
            "safety_stock_multiplier": safety_stock,
            "capacity_buffer_fraction": capacity_buffer,
        },
        "planned_replenishment_units": actual_planned_units,
        "planned_replenishment_received_units": 0.0,
        "planned_replenishment_useful_units": actual_useful_units,
        "planned_replenishment_cost": actual_planned_cost,
        "projected": False,
        "blocked": False,
    }


def _discrete_info(
    *,
    dispatch: str = "hold",
    route: str = "shortest",
    mode: str = "secondary_fleet",
    reorder: str = "none",
    override_units: float | None = None,
    override_useful_units: float | None = None,
    override_cost: float = 0.0,
    dispatched_orders: float = 0.0,
) -> dict[str, Any]:
    actual_override_units = 10.0 if override_units is None and override_cost > 0.0 else (override_units or 0.0)
    actual_useful_units = actual_override_units if override_useful_units is None else override_useful_units
    return {
        "action": {
            "dispatch": dispatch,
            "route": route,
            "mode": mode,
            "reorder": reorder,
        },
        "dqn_replenishment_override_cost": override_cost,
        "dqn_replenishment_override_units": actual_override_units,
        "dqn_replenishment_override_received_units": 0.0,
        "dqn_replenishment_override_useful_units": actual_useful_units,
        "planned_replenishment_units": 0.0,
        "dqn_dispatched_orders": dispatched_orders,
        "projected": False,
        "blocked": False,
        "discrete_blocked": False,
    }


def _joint_info(*, continuous: dict[str, float], discrete: dict[str, str]) -> dict[str, Any]:
    return {
        "action": {
            "continuous": continuous,
            "discrete": discrete,
        },
        "projected": False,
        "blocked": False,
        "planned_replenishment_units": 0.0,
        "planned_replenishment_received_units": 0.0,
        "planned_replenishment_useful_units": 0.0,
        "planned_replenishment_cost": 0.0,
        "dqn_replenishment_override_cost": 0.0,
        "dqn_replenishment_override_units": 0.0,
        "dqn_replenishment_override_received_units": 0.0,
        "dqn_replenishment_override_useful_units": 0.0,
        "dqn_dispatched_orders": 0.0,
        "discrete_blocked": False,
    }


def _calm_snapshot(env: FivePLDigitalTwinEnv) -> dict[str, Any]:
    _make_inventory_low_risk(env)
    snapshot = copy.deepcopy(env.simulation.snapshot())
    snapshot["disruption_score"] = 0.0
    snapshot["network_safety_potential"] = 0.05
    for order in snapshot["orders"].values():
        if isinstance(order, dict):
            order["status"] = "delivered"
    return snapshot


def _make_inventory_high_risk(env: FivePLDigitalTwinEnv) -> None:
    for position in env.simulation.inventory_network.positions.values():
        position.on_hand_units = 5.0
        position.reserved_units = 20.0
        position.in_transit_units = 0.0
        position.backlog_units = 150.0
        position.safety_stock_units = 1.0
        position.demand_mean = 120.0
        position.demand_variance = 80.0
        position.lead_time_mean = 3.0
        position.lead_time_variance = 2.0


def _make_inventory_medium_risk(env: FivePLDigitalTwinEnv) -> None:
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


def _make_inventory_low_risk(env: FivePLDigitalTwinEnv) -> None:
    for position in env.simulation.inventory_network.positions.values():
        position.on_hand_units = 5_000.0
        position.reserved_units = 0.0
        position.in_transit_units = 500.0
        position.backlog_units = 0.0
        position.safety_stock_units = 1_000.0
        position.demand_mean = 20.0
        position.demand_variance = 1.0
        position.lead_time_mean = 1.0
        position.lead_time_variance = 0.1


def _make_inventory_no_forecast_risk(env: FivePLDigitalTwinEnv) -> None:
    _make_inventory_low_risk(env)
    for position in env.simulation.inventory_network.positions.values():
        position.demand_variance = 0.0
        position.lead_time_variance = 0.0
        position.safety_stock_units = max(position.safety_stock_units, position.target_safety_stock())


def _make_capacity_pressure(env: FivePLDigitalTwinEnv) -> None:
    for state in env.simulation.capacity_states.values():
        state.reduce(0.75)


def _make_orders_urgent_and_aged(env: FivePLDigitalTwinEnv) -> None:
    now = env.simulation.env.now
    for order in env.simulation.orders.values():
        order.status = ShipmentStatus.CREATED
        order.release_time = now - 28_800.0
        order.due_time = now + 60.0
        order.delivery_time = None
        order.assigned_vehicle_id = None


def _make_orders_unassigned(env: FivePLDigitalTwinEnv) -> None:
    now = env.simulation.env.now
    for order in env.simulation.orders.values():
        order.status = ShipmentStatus.CREATED
        order.release_time = now
        order.due_time = now + 86_400.0
        order.pickup_time = None
        order.delivery_time = None
        order.assigned_vehicle_id = None


def _late_dispatch_pressure_snapshot(env: FivePLDigitalTwinEnv) -> dict[str, Any]:
    _make_orders_urgent_and_aged(env)
    snapshot = env.simulation.snapshot()
    now = float(snapshot["time"])
    snapshot["service_level"] = min(float(snapshot.get("service_level", 1.0)), 0.91)
    for order in snapshot["orders"].values():
        if isinstance(order, dict):
            order["status"] = ShipmentStatus.CREATED.value
            order["release_time"] = now - 28_800.0
            order["due_time"] = now - 3_600.0
            order["delivery_time"] = None
            order["assigned_vehicle_id"] = None
    return snapshot


def _apply_observed_fixed_medium_route_stress(env: FivePLDigitalTwinEnv) -> None:
    env.scenario_stress_state.update(
        {
            "route_disruption_probability": 0.25,
            "congestion_multiplier": 2.3125,
            "shortest_route_risk_multiplier": 1.5,
            "traversal_cost_multiplier": 1.3,
            "disruption_risk_multiplier": 1.5,
        }
    )


def _shortest_secondary_components(
    env: FivePLDigitalTwinEnv,
    *,
    reorder: str,
    projection_overrides: Mapping[str, Any] | None = None,
    dispatched_orders: float = 1.0,
    delivered_delta: int = 1,
) -> dict[str, float]:
    healthy = _calm_snapshot(env)
    projection_info = {
        **_discrete_info(
            dispatch="dispatch",
            route="shortest",
            mode="secondary_fleet",
            reorder=reorder,
            dispatched_orders=dispatched_orders,
        ),
        "dispatch_success_count": dispatched_orders,
        "dispatch_success_rate": 1.0 if dispatched_orders > 0.0 else 0.0,
        "feasible_dispatch_ratio": 0.80 if dispatched_orders > 0.0 else 0.0,
        "macro_dispatch_budget": 1.0,
    }
    if projection_overrides:
        projection_info.update(projection_overrides)
    return env._dqn_local_reward_components(
        healthy,
        projection_info,
        delivered_delta=delivered_delta,
        cost_delta=0.0,
        masked_order_ids=set(),
    )


def _assert_shortest_secondary_telemetry(
    testcase: unittest.TestCase,
    components: Mapping[str, float],
) -> None:
    for key in (
        "shortest_secondary_brittle_risk",
        "shortest_secondary_safe_useful_exception",
        "shortest_secondary_guard_active",
        "shortest_secondary_guard_relaxed_safe",
        "action24_25_candidate_credit_allowed",
        "action24_25_candidate_credit_blocked_reason",
    ):
        testcase.assertIn(key, components)


def _observation_features(env: FivePLDigitalTwinEnv) -> dict[str, float]:
    result = env.observation_builder.build(
        env.simulation,
        scenario_stress_state=env.scenario_stress_state,
    )
    required = {
        "shortest_candidate_score",
        "shortest_candidate_score_gap",
        "shortest_candidate_near_best",
        "shortest_secondary_safe_context",
        "shortest_secondary_brittle_risk",
    }
    missing = sorted(required.difference(OBSERVATION_FEATURE_INDEX))
    if missing:
        raise AssertionError(f"missing route candidate observation features: {missing}")
    return {
        name: float(result.vector[index])
        for name, index in OBSERVATION_FEATURE_INDEX.items()
    }


def _add_secondary_vehicles(env: FivePLDigitalTwinEnv, count: int) -> None:
    template = next(iter(env.simulation.vehicles.values()))
    for _ in range(count):
        env.simulation.add_vehicle(
            Vehicle(
                vehicle_id=uuid4(),
                asset_kind=AssetKind.VEHICLE,
                tier=VehicleTier.SECONDARY,
                capacity_units=template.capacity_units,
                capacity_weight_kg=template.capacity_weight_kg,
                capacity_volume_m3=template.capacity_volume_m3,
                nominal_speed_mps=template.nominal_speed_mps,
                fixed_usage_cost=template.fixed_usage_cost,
                transport_cost_per_meter=template.transport_cost_per_meter,
                current_node_id=template.current_node_id,
                current_location=template.current_location,
            )
        )


def _add_order_for_same_customer(env: FivePLDigitalTwinEnv, source_order: Order) -> Order:
    product_id = source_order.lines[0].product_id
    order = Order(
        order_id=uuid4(),
        customer_id=source_order.customer_id,
        origin_node_id=source_order.origin_node_id,
        destination_node_id=source_order.destination_node_id,
        lines=[OrderLine(product_id=product_id, quantity_units=10.0)],
        release_time=float(env.simulation.env.now),
        due_time=float(env.simulation.env.now + 86_400.0),
        metadata={"rolling_demand": True, "test_recurring_customer": True},
    )
    env.simulation.add_order(order)
    return order


def _add_backlog_orders(env: FivePLDigitalTwinEnv, *, count: int, quantity_units: float) -> None:
    source_order = next(iter(env.simulation.orders.values()))
    product_id = source_order.lines[0].product_id
    now = float(env.simulation.env.now)
    for _ in range(count):
        env.simulation.add_order(
            Order(
                order_id=uuid4(),
                customer_id=source_order.customer_id,
                origin_node_id=source_order.origin_node_id,
                destination_node_id=source_order.destination_node_id,
                lines=[OrderLine(product_id=product_id, quantity_units=quantity_units)],
                release_time=now,
                due_time=now + 3_600.0,
                metadata={"rolling_demand": True, "test_backlog": True},
            )
        )


if __name__ == "__main__":
    unittest.main()
