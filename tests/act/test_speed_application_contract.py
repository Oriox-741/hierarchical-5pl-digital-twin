from __future__ import annotations

import unittest

from src.act.action_projector import PhysicalAction
from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv


class SpeedApplicationContractTest(unittest.TestCase):
    def test_speed_multiplier_is_relative_to_vehicle_baseline_not_previous_speed(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            vehicle = next(iter(env.simulation.vehicles.values()))
            vehicle.baseline_speed_mps = 20.0
            vehicle.nominal_speed_mps = 20.0
            vehicle.metadata["baseline_speed_mps"] = 20.0

            action = PhysicalAction(
                reorder_fraction=0.0,
                dispatch_intensity=0.0,
                speed_multiplier=1.35,
                safety_stock_multiplier=1.0,
                capacity_buffer_fraction=0.0,
            )

            for _ in range(3):
                env._apply_vehicle_continuous_action(action, allow_dispatch=False)

            self.assertAlmostEqual(vehicle.nominal_speed_mps, 27.0)
        finally:
            env.close()

    def test_nominal_speed_action_restores_baseline_after_high_speed_action(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            vehicle = next(iter(env.simulation.vehicles.values()))
            vehicle.baseline_speed_mps = 20.0
            vehicle.nominal_speed_mps = 20.0
            vehicle.metadata["baseline_speed_mps"] = 20.0

            high_speed = PhysicalAction(
                reorder_fraction=0.0,
                dispatch_intensity=0.0,
                speed_multiplier=1.35,
                safety_stock_multiplier=1.0,
                capacity_buffer_fraction=0.0,
            )
            nominal_speed = PhysicalAction(
                reorder_fraction=0.0,
                dispatch_intensity=0.0,
                speed_multiplier=1.0,
                safety_stock_multiplier=1.0,
                capacity_buffer_fraction=0.0,
            )

            env._apply_vehicle_continuous_action(high_speed, allow_dispatch=False)
            self.assertAlmostEqual(vehicle.nominal_speed_mps, 27.0)

            env._apply_vehicle_continuous_action(nominal_speed, allow_dispatch=False)
            self.assertAlmostEqual(vehicle.nominal_speed_mps, 20.0)
        finally:
            env.close()

    def test_multiple_vehicles_keep_distinct_baseline_relative_speeds(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            vehicles = list(env.simulation.vehicles.values())
            self.assertGreaterEqual(len(vehicles), 2)
            vehicles[0].baseline_speed_mps = 20.0
            vehicles[0].nominal_speed_mps = 20.0
            vehicles[1].baseline_speed_mps = 10.0
            vehicles[1].nominal_speed_mps = 10.0

            action = PhysicalAction(
                reorder_fraction=0.0,
                dispatch_intensity=0.0,
                speed_multiplier=1.25,
                safety_stock_multiplier=1.0,
                capacity_buffer_fraction=0.0,
            )
            env._apply_vehicle_continuous_action(action, allow_dispatch=False)

            self.assertAlmostEqual(vehicles[0].nominal_speed_mps, 25.0)
            self.assertAlmostEqual(vehicles[1].nominal_speed_mps, 12.5)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
