from __future__ import annotations

import unittest
from uuid import uuid4

from src.act.entities import RouteArc, Vehicle, VehicleTier
from src.act.routing_physics import RoutingPhysics
from src.shared.types import AssetKind


class RoutingPhysicsCostTest(unittest.TestCase):
    def test_speed_cost_is_nonlinear_and_monotonic_above_baseline(self) -> None:
        physics = RoutingPhysics(stochastic_noise_std=0.0)
        arc = RouteArc(
            origin_node_id=uuid4(),
            destination_node_id=uuid4(),
            distance_m=10_000.0,
            nominal_speed_mps=40.0,
            traversal_cost_per_unit=0.002,
        )
        vehicle = Vehicle(
            vehicle_id=uuid4(),
            asset_kind=AssetKind.VEHICLE,
            tier=VehicleTier.SECONDARY,
            capacity_units=100.0,
            capacity_weight_kg=1_000.0,
            capacity_volume_m3=10.0,
            nominal_speed_mps=20.0,
            baseline_speed_mps=20.0,
            transport_cost_per_meter=0.003,
        )

        vehicle.nominal_speed_mps = 20.0
        cost_100 = physics.traversal_cost_breakdown(arc, vehicle)
        vehicle.nominal_speed_mps = 24.0
        cost_120 = physics.traversal_cost_breakdown(arc, vehicle)
        vehicle.nominal_speed_mps = 27.0
        cost_135 = physics.traversal_cost_breakdown(arc, vehicle)

        self.assertAlmostEqual(cost_100.speed_ratio, 1.0)
        self.assertAlmostEqual(cost_100.total_cost, cost_100.distance_cost)
        self.assertGreater(cost_120.total_cost, cost_100.total_cost)
        self.assertGreater(cost_135.total_cost, cost_120.total_cost)
        self.assertGreater(
            cost_135.total_cost - cost_120.total_cost,
            cost_120.total_cost - cost_100.total_cost,
        )
        self.assertGreater(cost_135.energy_multiplier, cost_120.energy_multiplier)
        self.assertGreater(cost_135.risk_multiplier, cost_120.risk_multiplier)

    def test_arc_speed_cap_prevents_false_speed_cost_when_effective_speed_does_not_exceed_baseline(self) -> None:
        physics = RoutingPhysics(stochastic_noise_std=0.0)
        arc = RouteArc(
            origin_node_id=uuid4(),
            destination_node_id=uuid4(),
            distance_m=10_000.0,
            nominal_speed_mps=18.0,
            traversal_cost_per_unit=0.002,
        )
        vehicle = Vehicle(
            vehicle_id=uuid4(),
            asset_kind=AssetKind.VEHICLE,
            tier=VehicleTier.SECONDARY,
            capacity_units=100.0,
            capacity_weight_kg=1_000.0,
            capacity_volume_m3=10.0,
            nominal_speed_mps=27.0,
            baseline_speed_mps=20.0,
            transport_cost_per_meter=0.003,
        )

        cost = physics.traversal_cost_breakdown(arc, vehicle)

        self.assertLess(cost.speed_ratio, 1.0)
        self.assertEqual(cost.speed_cost, 0.0)
        self.assertAlmostEqual(cost.total_cost, cost.distance_cost)


if __name__ == "__main__":
    unittest.main()
