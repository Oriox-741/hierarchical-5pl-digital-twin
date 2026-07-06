from __future__ import annotations

from pathlib import Path
import unittest

from src.eval.real_world_scenario_arena import load_scenario_configs


class RealWorldScenarioConfigTests(unittest.TestCase):
    def test_all_repository_scenarios_load_and_validate(self) -> None:
        scenarios = load_scenario_configs(Path("configs/eval_scenarios"))

        scenario_ids = {scenario.scenario_id for scenario in scenarios}
        self.assertEqual(
            scenario_ids,
            {
                "baseline_normal",
                "high_holding_cost",
                "demand_spike_volatility",
                "lead_time_volatility",
                "vehicle_scarcity_capacity_shock",
                "route_disruption_congestion",
                "premium_sla_pressure",
                "mixed_stress",
            },
        )
        for scenario in scenarios:
            self.assertGreater(scenario.episode_count, 0)
            self.assertGreater(len(scenario.seeds), 0)
            self.assertIsInstance(scenario.environment_overrides, dict)
            self.assertIsInstance(scenario.pass_fail_thresholds, dict)

    def test_stress_scenarios_have_operational_quality_thresholds(self) -> None:
        scenarios = {
            scenario.scenario_id: scenario
            for scenario in load_scenario_configs(Path("configs/eval_scenarios"))
        }

        for scenario_id in ("demand_spike_volatility", "mixed_stress"):
            thresholds = scenarios[scenario_id].pass_fail_thresholds
            self.assertIn("min_service_level", thresholds)
            self.assertIn("max_true_lateness_pressure", thresholds)
            self.assertIn("min_dispatch_success_per_attempt", thresholds)
            self.assertLess(thresholds["max_true_lateness_pressure"], 0.40)

    def test_action_concentration_thresholds_are_warning_first(self) -> None:
        scenarios = load_scenario_configs(Path("configs/eval_scenarios"))

        for scenario in scenarios:
            thresholds = scenario.pass_fail_thresholds
            if "max_action_24_25_concentration_warn" in thresholds:
                self.assertIn("max_action_24_25_concentration_fail", thresholds)
                self.assertTrue(thresholds.get("require_operational_degradation_for_action_concentration_fail"))


if __name__ == "__main__":
    unittest.main()
