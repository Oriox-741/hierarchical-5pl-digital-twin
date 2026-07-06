from __future__ import annotations

import math
import unittest

import numpy as np

from src.act.entities import VehicleTier
from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv
from src.act.observation_builder import OBSERVATION_DIM, OBSERVATION_FEATURE_INDEX, OBSERVATION_FEATURES, ObservationBuilder
from src.learn.train_joint_torch import MDP_CONTRACT_VERSION


EXPECTED_V3_FEATURES = (
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
    "normalized_available_vehicle_count",
    "normalized_dispatchable_order_count",
    "feasible_dispatch_opportunity",
    "feasible_dispatch_ratio",
    "vehicle_availability_pressure",
)

EXPECTED_V4_SUFFIX = (
    "normalized_holding_cost_pressure",
    "normalized_stockout_penalty_pressure",
    "capacity_shock_pressure",
    "scenario_route_disruption_pressure",
    "route_cost_pressure",
    "premium_sla_pressure",
    "supplier_delay_pressure",
    "scenario_lead_time_volatility_pressure",
)

EXPECTED_V5_SUFFIX = (
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

EXPECTED_V4_FEATURES = EXPECTED_V3_FEATURES + EXPECTED_V4_SUFFIX
EXPECTED_V5_FEATURES = EXPECTED_V4_FEATURES + EXPECTED_V5_SUFFIX


class ObservationBuilderContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False, random_seed=7))
        self.builder = ObservationBuilder()

    def tearDown(self) -> None:
        self.env.close()

    def test_vector_length_and_normalized_finite_values(self) -> None:
        result = self.builder.build(self.env.simulation)

        self.assertEqual(result.vector.shape, (OBSERVATION_DIM,))
        self.assertTrue(np.isfinite(result.vector).all())
        self.assertTrue((result.vector >= 0.0).all())
        self.assertTrue((result.vector <= 1.0).all())

    def test_hot_path_context_preserves_observation_vector_values(self) -> None:
        snapshot = self.env.simulation.snapshot()

        baseline = self.builder.build(self.env.simulation, snapshot=snapshot)
        context = self.builder._build_hot_path_context(self.env.simulation, snapshot)
        optimized = self.builder.build(self.env.simulation, snapshot=snapshot, hot_path_context=context)

        self.assertEqual(optimized.vector.shape, (73,))
        np.testing.assert_allclose(optimized.vector, baseline.vector, rtol=0.0, atol=0.0)
        self.assertEqual(context.total_order_count, len(self.env.simulation.orders))
        self.assertEqual(context.total_vehicle_count, len(self.env.simulation.vehicles))

    def test_observation_contract_is_73_features_with_v5_contract(self) -> None:
        self.assertEqual(OBSERVATION_DIM, 73)
        self.assertEqual(len(OBSERVATION_FEATURE_INDEX), 73)
        self.assertEqual(self.env.observation_space.shape, (73,))
        self.assertEqual(MDP_CONTRACT_VERSION, "physical_reality_v5_route_candidate_visibility")

    def test_v4_feature_order_is_preserved_and_v5_features_are_appended(self) -> None:
        self.assertEqual(OBSERVATION_FEATURES[:49], EXPECTED_V3_FEATURES)
        self.assertEqual(OBSERVATION_FEATURES[49:57], EXPECTED_V4_SUFFIX)
        self.assertEqual(OBSERVATION_FEATURES[57:], EXPECTED_V5_SUFFIX)
        self.assertEqual(OBSERVATION_FEATURES, EXPECTED_V5_FEATURES)
        self.assertEqual(
            tuple(OBSERVATION_FEATURE_INDEX[name] for name in EXPECTED_V4_SUFFIX),
            tuple(range(49, 57)),
        )
        self.assertEqual(
            tuple(OBSERVATION_FEATURE_INDEX[name] for name in EXPECTED_V5_SUFFIX),
            tuple(range(57, 73)),
        )

    def test_stockout_risk_rises_when_backlog_rises(self) -> None:
        base = self._feature("stockout_risk")
        for position in self.env.simulation.inventory_network.positions.values():
            position.backlog_units += max(position.demand_mean, 1.0) * 5.0

        stressed = self._feature("stockout_risk")

        self.assertGreater(stressed, base)

    def test_speed_cost_exposure_rises_when_vehicle_speed_exceeds_baseline(self) -> None:
        base = self._feature("speed_cost_exposure")
        for vehicle in self.env.simulation.vehicles.values():
            baseline = vehicle.baseline_speed_mps or vehicle.nominal_speed_mps
            vehicle.nominal_speed_mps = baseline * 1.35

        stressed = self._feature("speed_cost_exposure")

        self.assertGreater(stressed, base)

    def test_safety_stock_target_gap_rises_when_safety_stock_is_below_target(self) -> None:
        for position in self.env.simulation.inventory_network.positions.values():
            target = position.target_safety_stock()
            position.safety_stock_units = max(position.safety_stock_units, target)
        sufficient = self._feature("safety_stock_target_gap")

        for position in self.env.simulation.inventory_network.positions.values():
            position.safety_stock_units = 0.0
        deficient = self._feature("safety_stock_target_gap")

        self.assertGreater(deficient, sufficient)

    def test_real_world_features_have_named_indices(self) -> None:
        required = {
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
        }

        self.assertTrue(required.issubset(OBSERVATION_FEATURE_INDEX))
        self.assertEqual(len(set(OBSERVATION_FEATURE_INDEX.values())), OBSERVATION_DIM)

    def test_dispatch_feasibility_features_are_appended_in_expected_order(self) -> None:
        expected_order = (
            "normalized_available_vehicle_count",
            "normalized_dispatchable_order_count",
            "feasible_dispatch_opportunity",
            "feasible_dispatch_ratio",
            "vehicle_availability_pressure",
        )

        self.assertEqual(
            tuple(OBSERVATION_FEATURE_INDEX[name] for name in expected_order),
            tuple(range(44, 49)),
        )

    def test_dispatch_feasibility_features_are_bounded(self) -> None:
        result = self.builder.build(self.env.simulation)

        for name in (
            "normalized_available_vehicle_count",
            "normalized_dispatchable_order_count",
            "feasible_dispatch_opportunity",
            "feasible_dispatch_ratio",
            "vehicle_availability_pressure",
        ):
            value = float(result.vector[OBSERVATION_FEATURE_INDEX[name]])
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertIn(self._feature("feasible_dispatch_opportunity"), {0.0, 1.0})

    def test_default_stress_features_are_baseline_normalized(self) -> None:
        self._assert_v5_features_present()
        result = self.builder.build(self.env.simulation, scenario_stress_state=self.env.scenario_stress_state)

        for name in EXPECTED_V4_SUFFIX:
            self.assertEqual(float(result.vector[OBSERVATION_FEATURE_INDEX[name]]), 0.0, name)

    def test_high_holding_cost_activates_holding_cost_pressure(self) -> None:
        self.env.scenario_stress_state["holding_cost_multiplier"] = 2.25
        self.env.scenario_stress_state["excess_inventory_penalty_multiplier"] = 2.0
        self.env.scenario_stress_state["planned_replenishment_cost_multiplier"] = 2.0

        self.assertGreater(self._feature("normalized_holding_cost_pressure"), 0.0)

    def test_stockout_penalty_multiplier_activates_stockout_penalty_pressure(self) -> None:
        self.env.scenario_stress_state["stockout_penalty_multiplier"] = 1.6
        self.env.scenario_stress_state["inventory_shortfall_penalty_multiplier"] = 1.5

        self.assertGreater(self._feature("normalized_stockout_penalty_pressure"), 0.0)

    def test_vehicle_scarcity_activates_capacity_shock_pressure(self) -> None:
        self.env.scenario_stress_state["vehicle_availability_multiplier"] = 0.5
        self.env.scenario_stress_state["fleet_capacity_multiplier"] = 0.55
        self.env.scenario_stress_state["capacity_shock_severity"] = 0.45

        self.assertGreater(self._feature("capacity_shock_pressure"), 0.0)

    def test_route_disruption_activates_route_stress_features(self) -> None:
        self.env.scenario_stress_state["route_disruption_probability"] = 0.6
        self.env.scenario_stress_state["congestion_multiplier"] = 3.0
        self.env.scenario_stress_state["traversal_cost_multiplier"] = 1.5

        self.assertGreater(self._feature("scenario_route_disruption_pressure"), 0.0)
        self.assertGreater(self._feature("route_cost_pressure"), 0.0)

    def test_premium_sla_activates_premium_sla_pressure(self) -> None:
        self.env.scenario_stress_state["premium_sla_ratio"] = 0.55
        self.env.scenario_stress_state["service_target"] = 0.98
        self.env.scenario_stress_state["urgent_due_window_multiplier"] = 0.65
        self.env.scenario_stress_state["premium_lateness_penalty_multiplier"] = 1.8

        self.assertGreater(self._feature("premium_sla_pressure"), 0.0)

    def test_supplier_delay_and_lead_time_activate_supplier_stress_features(self) -> None:
        self.env.scenario_stress_state["supplier_delay_probability"] = 0.45
        self.env.scenario_stress_state["lead_time_mean_multiplier"] = 1.8
        self.env.scenario_stress_state["lead_time_variance_multiplier"] = 3.0

        self.assertGreater(self._feature("supplier_delay_pressure"), 0.0)
        self.assertGreater(self._feature("scenario_lead_time_volatility_pressure"), 0.0)

    def test_no_vehicle_with_dispatchable_order_reports_low_feasibility_and_high_vehicle_pressure(self) -> None:
        for vehicle in self.env.simulation.vehicles.values():
            vehicle.active = False
        for order in self.env.simulation.orders.values():
            order.assigned_vehicle_id = None
            order.delivery_time = None

        self.assertGreater(self._feature("normalized_dispatchable_order_count"), 0.0)
        self.assertEqual(self._feature("normalized_available_vehicle_count"), 0.0)
        self.assertEqual(self._feature("feasible_dispatch_opportunity"), 0.0)
        self.assertEqual(self._feature("feasible_dispatch_ratio"), 0.0)
        self.assertEqual(self._feature("vehicle_availability_pressure"), 1.0)

    def test_available_vehicle_with_dispatchable_order_reports_feasible_opportunity(self) -> None:
        for vehicle in self.env.simulation.vehicles.values():
            vehicle.active = True
        for order in self.env.simulation.orders.values():
            order.assigned_vehicle_id = None
            order.delivery_time = None

        self.assertGreater(self._feature("normalized_available_vehicle_count"), 0.0)
        self.assertGreater(self._feature("normalized_dispatchable_order_count"), 0.0)
        self.assertEqual(self._feature("feasible_dispatch_opportunity"), 1.0)
        self.assertGreater(self._feature("feasible_dispatch_ratio"), 0.0)
        self.assertLess(self._feature("vehicle_availability_pressure"), 1.0)

    def test_route_candidate_visibility_features_are_bounded_and_pre_action_named(self) -> None:
        _make_orders_unassigned(self.env)
        _activate_all_vehicles(self.env)
        _apply_fixed_medium_route_stress(self.env)

        self._assert_v5_features_present()
        result = self.builder.build(self.env.simulation, scenario_stress_state=self.env.scenario_stress_state)

        for name in EXPECTED_V5_SUFFIX:
            value = float(result.vector[OBSERVATION_FEATURE_INDEX[name]])
            self.assertTrue(math.isfinite(value), name)
            self.assertGreaterEqual(value, 0.0, name)
            self.assertLessEqual(value, 1.0, name)
        forbidden_selected_action_telemetry = {
            "selected_route_candidate_action_gate",
            "route_candidate_alignment_credit",
            "selected_route_score_gap",
            "exact_no_vehicle_exposure",
            "exact_already_assigned_saturation",
            "action24_25_candidate_credit_blocked_reason",
        }
        self.assertTrue(forbidden_selected_action_telemetry.isdisjoint(OBSERVATION_FEATURE_INDEX))

    def test_candidate_score_gaps_follow_max_score(self) -> None:
        _make_orders_unassigned(self.env)
        _activate_all_vehicles(self.env)
        _apply_fixed_medium_route_stress(self.env)

        features = self._v5_feature_values()
        scores = {
            "shortest": features["shortest_candidate_score"],
            "low_congestion": features["low_congestion_candidate_score"],
            "high_resilience": features["high_resilience_candidate_score"],
        }
        best = max(scores.values())

        self.assertGreater(best, 0.0)
        for route, score in scores.items():
            self.assertAlmostEqual(
                features[f"{route}_candidate_score_gap"],
                best - score,
                places=6,
                msg=route,
            )

    def test_fixed_medium_safe_route_state_exposes_shortest_near_best_and_safe_context(self) -> None:
        _make_orders_unassigned(self.env)
        _activate_all_vehicles(self.env)
        _apply_fixed_medium_route_stress(self.env)

        features = self._v5_feature_values()

        self.assertGreater(features["shortest_candidate_score"], 0.0)
        self.assertLessEqual(features["shortest_candidate_score_gap"], 0.12)
        self.assertEqual(features["shortest_candidate_near_best"], 1.0)
        self.assertGreater(features["shortest_secondary_safe_context"], 0.0)
        self.assertLess(features["shortest_secondary_brittle_risk"], 0.5)

    def test_congestion_and_reliability_stress_shift_candidate_advantage(self) -> None:
        _make_orders_unassigned(self.env)
        _activate_all_vehicles(self.env)

        self.env.scenario_stress_state.update(
            {
                "route_disruption_probability": 0.05,
                "congestion_multiplier": 2.8,
                "shortest_route_risk_multiplier": 1.05,
                "traversal_cost_multiplier": 2.0,
                "disruption_risk_multiplier": 1.05,
            }
        )
        congestion_features = self._v5_feature_values()

        self.env.scenario_stress_state.clear()
        self.env.scenario_stress_state.update(
            {
                "route_disruption_probability": 0.50,
                "congestion_multiplier": 1.3,
                "shortest_route_risk_multiplier": 2.0,
                "traversal_cost_multiplier": 1.1,
                "disruption_risk_multiplier": 2.0,
            }
        )
        reliability_features = self._v5_feature_values()

        self.assertGreater(
            congestion_features["low_congestion_candidate_score"],
            congestion_features["high_resilience_candidate_score"],
        )
        self.assertGreater(
            reliability_features["high_resilience_candidate_score"],
            reliability_features["low_congestion_candidate_score"],
        )

    def test_secondary_fleet_scarcity_and_useful_dispatch_opportunity_are_visible(self) -> None:
        _make_orders_unassigned(self.env)
        _activate_all_vehicles(self.env)
        feasible = self._v5_feature_values()

        for vehicle in self.env.simulation.vehicles.values():
            if getattr(vehicle, "tier", None) == VehicleTier.SECONDARY:
                vehicle.active = False
        scarce = self._v5_feature_values()

        for order in self.env.simulation.orders.values():
            order.delivery_time = self.env.simulation.env.now
        no_work = self._v5_feature_values()

        self.assertGreater(feasible["secondary_fleet_feasible_dispatch_ratio"], 0.0)
        self.assertLess(
            feasible["secondary_fleet_vehicle_availability_pressure"],
            scarce["secondary_fleet_vehicle_availability_pressure"],
        )
        self.assertLess(
            scarce["secondary_fleet_feasible_dispatch_ratio"],
            feasible["secondary_fleet_feasible_dispatch_ratio"],
        )
        self.assertGreater(scarce["secondary_fleet_vehicle_availability_pressure"], feasible["secondary_fleet_vehicle_availability_pressure"])
        self.assertGreater(feasible["useful_dispatch_opportunity"], no_work["useful_dispatch_opportunity"])

    def _feature(self, name: str) -> float:
        result = self.builder.build(self.env.simulation, scenario_stress_state=self.env.scenario_stress_state)
        value = float(result.vector[OBSERVATION_FEATURE_INDEX[name]])
        self.assertTrue(math.isfinite(value))
        return value

    def _features(self) -> dict[str, float]:
        result = self.builder.build(self.env.simulation, scenario_stress_state=self.env.scenario_stress_state)
        return {
            name: float(result.vector[OBSERVATION_FEATURE_INDEX[name]])
            for name in OBSERVATION_FEATURES
        }

    def _v5_feature_values(self) -> dict[str, float]:
        self._assert_v5_features_present()
        return self._features()

    def _assert_v5_features_present(self) -> None:
        for name in EXPECTED_V5_SUFFIX:
            self.assertIn(name, OBSERVATION_FEATURE_INDEX)


def _make_orders_unassigned(env: FivePLDigitalTwinEnv) -> None:
    now = env.simulation.env.now
    for order in env.simulation.orders.values():
        order.assigned_vehicle_id = None
        order.delivery_time = None
        order.release_time = now
        order.due_time = now + 86_400.0


def _activate_all_vehicles(env: FivePLDigitalTwinEnv) -> None:
    for vehicle in env.simulation.vehicles.values():
        vehicle.active = True


def _apply_fixed_medium_route_stress(env: FivePLDigitalTwinEnv) -> None:
    env.scenario_stress_state.update(
        {
            "route_disruption_probability": 0.25,
            "congestion_multiplier": 2.3125,
            "shortest_route_risk_multiplier": 1.5,
            "traversal_cost_multiplier": 1.3,
            "disruption_risk_multiplier": 1.5,
        }
    )


if __name__ == "__main__":
    unittest.main()
