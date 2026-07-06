from __future__ import annotations

from pathlib import Path
import unittest

from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv
from src.eval.real_world_scenario_arena import ScenarioConfig, build_scenario_environment, load_scenario_configs
from src.eval.scenario_overrides import apply_config_overrides, apply_environment_overrides


BASE_CONFIG = {
    "mdp_contract_version": "physical_reality_v5_route_candidate_visibility",
    "shared_global_parameters": {
        "observation_dim": 73,
        "max_steps": 8,
        "decision_interval_seconds": 300,
        "max_extra_dispatch_budget": 4,
        "feasibility_gated_macro_budget": True,
    },
    "reward_physics": {
        "infeasible_dispatch_penalty_weight": 0.04,
        "flow_enablement_weight": 0.08,
        "dispatch_progress_weight": 0.05,
    },
}


def _scenario(overrides: dict[str, object]) -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="test",
        name="Test",
        description="test scenario",
        seeds=(1,),
        episode_count=1,
        environment_overrides=overrides,
    )


class ScenarioOverrideTests(unittest.TestCase):
    def test_no_override_preserves_environment_defaults(self) -> None:
        baseline = EnvironmentConfig()
        config = EnvironmentConfig()
        matrix = apply_config_overrides(config, {})

        self.assertEqual(matrix, {})
        self.assertEqual(config.rolling_demand_probability, baseline.rolling_demand_probability)
        self.assertEqual(config.rolling_demand_max_orders_per_step, baseline.rolling_demand_max_orders_per_step)
        self.assertEqual(config.urgent_order_probability, baseline.urgent_order_probability)
        self.assertEqual(config.stockout_penalty_multiplier, 1.0)
        self.assertEqual(config.inventory_shortfall_penalty_multiplier, 1.0)

    def test_demand_spike_overrides_change_demand_controls(self) -> None:
        config = EnvironmentConfig()
        apply_config_overrides(
            config,
            {
                "rolling_demand_probability_multiplier": 2.0,
                "rolling_demand_volume_multiplier": 2.0,
                "order_units_multiplier": 1.5,
                "urgent_order_probability_multiplier": 1.5,
            },
        )

        self.assertGreater(config.rolling_demand_probability, EnvironmentConfig().rolling_demand_probability)
        self.assertGreater(config.rolling_demand_max_orders_per_step, EnvironmentConfig().rolling_demand_max_orders_per_step)
        self.assertGreater(config.rolling_demand_min_units, EnvironmentConfig().rolling_demand_min_units)
        self.assertGreater(config.urgent_order_probability, EnvironmentConfig().urgent_order_probability)

    def test_holding_and_stockout_overrides_change_reward_economics(self) -> None:
        config = EnvironmentConfig()
        apply_config_overrides(
            config,
            {
                "holding_cost_multiplier": 2.0,
                "stockout_penalty_multiplier": 1.7,
                "inventory_shortfall_penalty_multiplier": 1.6,
                "emergency_procurement_cost_multiplier": 1.5,
            },
        )

        self.assertGreater(config.excess_inventory_penalty_weight, EnvironmentConfig().excess_inventory_penalty_weight)
        self.assertGreater(config.planned_replenishment_unit_cost, EnvironmentConfig().planned_replenishment_unit_cost)
        self.assertEqual(config.stockout_penalty_multiplier, 1.7)
        self.assertEqual(config.inventory_shortfall_penalty_multiplier, 1.6)
        self.assertGreater(config.emergency_replenishment_unit_cost, EnvironmentConfig().emergency_replenishment_unit_cost)

    def test_scenario_overrides_store_observation_stress_metadata(self) -> None:
        config = EnvironmentConfig()
        apply_config_overrides(
            config,
            {
                "holding_cost_multiplier": 2.25,
                "excess_inventory_penalty_multiplier": 2.0,
                "planned_replenishment_cost_multiplier": 2.0,
                "stockout_penalty_multiplier": 1.6,
                "inventory_shortfall_penalty_multiplier": 1.5,
                "premium_sla_ratio": 0.55,
                "urgent_due_window_multiplier": 0.65,
                "premium_lateness_penalty_multiplier": 1.8,
                "service_target": 0.98,
            },
        )
        env = FivePLDigitalTwinEnv(config=config)
        try:
            apply_environment_overrides(
                env,
                {
                    "holding_cost_multiplier": 2.25,
                    "excess_inventory_penalty_multiplier": 2.0,
                    "planned_replenishment_cost_multiplier": 2.0,
                    "stockout_penalty_multiplier": 1.6,
                    "inventory_shortfall_penalty_multiplier": 1.5,
                    "vehicle_availability_multiplier": 0.5,
                    "fleet_capacity_multiplier": 0.55,
                    "capacity_shock_severity": 0.45,
                    "route_disruption_probability": 0.60,
                    "congestion_multiplier": 3.0,
                    "shortest_route_risk_multiplier": 2.0,
                    "traversal_cost_multiplier": 1.5,
                    "disruption_risk_multiplier": 1.8,
                    "route_fleet_cost_multiplier": 1.4,
                    "premium_sla_ratio": 0.55,
                    "service_target": 0.98,
                    "urgent_due_window_multiplier": 0.65,
                    "premium_lateness_penalty_multiplier": 1.8,
                    "supplier_delay_probability": 0.45,
                    "lead_time_mean_multiplier": 1.8,
                    "lead_time_variance_multiplier": 3.0,
                },
                {},
            )

            stress = env.scenario_stress_state

            self.assertEqual(stress["holding_cost_multiplier"], 2.25)
            self.assertEqual(stress["stockout_penalty_multiplier"], 1.6)
            self.assertEqual(stress["capacity_shock_severity"], 0.45)
            self.assertEqual(stress["route_disruption_probability"], 0.60)
            self.assertEqual(stress["premium_sla_ratio"], 0.55)
            self.assertEqual(stress["supplier_delay_probability"], 0.45)
            self.assertEqual(stress["lead_time_mean_multiplier"], 1.8)
            self.assertEqual(stress["lead_time_variance_multiplier"], 3.0)
        finally:
            env.close()

    def test_lead_time_and_supplier_delay_overrides_change_inventory_state(self) -> None:
        env = FivePLDigitalTwinEnv(config=EnvironmentConfig(max_steps=4, random_seed=1))
        before = [
            (position.lead_time_mean, position.lead_time_variance)
            for position in env.simulation.inventory_network.positions.values()
        ]

        matrix = apply_environment_overrides(
            env,
            {
                "supplier_delay_probability": 0.5,
                "lead_time_mean_multiplier": 2.0,
                "lead_time_variance_multiplier": 3.0,
            },
        )
        after = [
            (position.lead_time_mean, position.lead_time_variance)
            for position in env.simulation.inventory_network.positions.values()
        ]

        self.assertEqual(matrix["supplier_delay_probability"].status, "APPLIED")
        self.assertTrue(all(new_mean > old_mean for (old_mean, _), (new_mean, _) in zip(before, after, strict=True)))
        self.assertTrue(all(new_var > old_var for (_, old_var), (_, new_var) in zip(before, after, strict=True)))

    def test_vehicle_scarcity_and_capacity_shock_change_supply(self) -> None:
        env = FivePLDigitalTwinEnv(config=EnvironmentConfig(max_steps=4, random_seed=1))
        before_active = sum(1 for vehicle in env.simulation.vehicles.values() if vehicle.active)
        before_capacity = sum(vehicle.capacity_units for vehicle in env.simulation.vehicles.values())

        matrix = apply_environment_overrides(
            env,
            {
                "vehicle_availability_multiplier": 0.5,
                "fleet_capacity_multiplier": 0.5,
                "capacity_shock_severity": 0.5,
            },
        )
        after_active = sum(1 for vehicle in env.simulation.vehicles.values() if vehicle.active)
        after_capacity = sum(vehicle.capacity_units for vehicle in env.simulation.vehicles.values())

        self.assertEqual(matrix["vehicle_availability_multiplier"].status, "APPLIED")
        self.assertLess(after_active, before_active)
        self.assertLess(after_capacity, before_capacity)

    def test_route_disruption_overrides_change_routing_physics(self) -> None:
        env = FivePLDigitalTwinEnv(config=EnvironmentConfig(max_steps=4, random_seed=1))
        before_cost_multiplier = env.simulation.routing_physics.traversal_cost_multiplier

        matrix = apply_environment_overrides(
            env,
            {
                "route_disruption_probability": 0.6,
                "congestion_multiplier": 3.0,
                "traversal_cost_multiplier": 1.5,
                "disruption_risk_multiplier": 2.0,
            },
        )

        self.assertEqual(matrix["route_disruption_probability"].status, "APPLIED")
        self.assertTrue(env.simulation.routing_physics.congestion_by_arc)
        self.assertTrue(env.simulation.routing_physics.arc_delay_multiplier)
        self.assertGreater(env.simulation.routing_physics.traversal_cost_multiplier, before_cost_multiplier)

    def test_premium_sla_overrides_change_urgency_controls(self) -> None:
        config = EnvironmentConfig()
        apply_config_overrides(
            config,
            {
                "premium_sla_ratio": 0.7,
                "urgent_due_window_multiplier": 0.5,
                "premium_lateness_penalty_multiplier": 1.5,
                "service_target": 0.98,
            },
        )

        self.assertGreaterEqual(config.urgent_order_probability, 0.7)
        self.assertEqual(config.urgent_due_window_multiplier, 0.5)
        self.assertGreater(config.delay_weight, EnvironmentConfig().delay_weight)
        self.assertEqual(config.service_level_target, 0.98)

    def test_invalid_override_is_not_silently_applied(self) -> None:
        config = EnvironmentConfig()
        matrix = apply_config_overrides(config, {"not_a_real_override": 1.0})

        self.assertEqual(matrix["not_a_real_override"].status, "UNSUPPORTED")

    def test_repository_scenarios_have_no_unsupported_core_knobs(self) -> None:
        scenarios = load_scenario_configs(Path("configs/eval_scenarios"))

        for scenario in scenarios:
            env, matrix = build_scenario_environment(BASE_CONFIG, scenario, seed=scenario.seeds[0])
            env.close()
            statuses = {item["key"]: item["status"] for item in matrix}
            unsupported = [key for key, status in statuses.items() if status == "UNSUPPORTED"]
            self.assertEqual(unsupported, [], f"{scenario.scenario_id} has unsupported overrides: {unsupported}")
            if scenario.scenario_id == "demand_spike_volatility":
                self.assertEqual(statuses["clustered_spike_windows"], "PARTIAL")


if __name__ == "__main__":
    unittest.main()
