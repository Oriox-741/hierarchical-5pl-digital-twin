from __future__ import annotations

import unittest

from src.act.action_projector import PhysicalAction
from src.act.discrete_action_mapper import DispatchDecision, DiscreteLogisticsAction, ModeDecision, ReorderDecision, RouteDecision
from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv


class ReplenishmentOwnershipTest(unittest.TestCase):
    def test_ppo_planned_reorder_is_separate_from_dqn_none(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            before = _total_on_hand(env)

            planned = env._apply_physical_action(
                PhysicalAction(
                    reorder_fraction=1.0,
                    dispatch_intensity=0.0,
                    speed_multiplier=1.0,
                    safety_stock_multiplier=1.0,
                    capacity_buffer_fraction=0.0,
                ),
                allow_dispatch=False,
            )
            after_planned = _total_on_hand(env)
            after_planned_in_transit = _total_in_transit(env)

            none_override = env._apply_discrete_action(
                DiscreteLogisticsAction(
                    dispatch=DispatchDecision.HOLD,
                    route=RouteDecision.SHORTEST,
                    mode=ModeDecision.SECONDARY_FLEET,
                    reorder=ReorderDecision.NONE,
                )
            )

            self.assertEqual(after_planned, before)
            self.assertGreater(after_planned_in_transit, 0.0)
            self.assertGreater(planned["planned_replenishment_units"], 0.0)
            self.assertEqual(planned["planned_replenishment_received_units"], 0.0)
            self.assertGreater(planned["planned_replenishment_cost"], 0.0)
            self.assertEqual(none_override["dqn_replenishment_override_units"], 0.0)
            self.assertEqual(none_override["dqn_replenishment_override_received_units"], 0.0)
            self.assertEqual(none_override["dqn_replenishment_override_cost"], 0.0)
            self.assertEqual(_total_on_hand(env), after_planned)
        finally:
            env.close()

    def test_dqn_conservative_does_not_create_extra_stock(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            before = _total_on_hand(env)
            result = env._apply_discrete_action(
                DiscreteLogisticsAction(
                    dispatch=DispatchDecision.HOLD,
                    route=RouteDecision.SHORTEST,
                    mode=ModeDecision.SECONDARY_FLEET,
                    reorder=ReorderDecision.CONSERVATIVE,
                )
            )

            self.assertEqual(result["dqn_replenishment_override_units"], 0.0)
            self.assertEqual(result["dqn_replenishment_override_received_units"], 0.0)
            self.assertEqual(result["dqn_replenishment_override_cost"], 0.0)
            self.assertEqual(_total_on_hand(env), before)
        finally:
            env.close()

    def test_dqn_emergency_override_is_recorded_as_costly_procurement(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            before = _total_on_hand(env)
            before_safety_stock = _total_safety_stock(env)
            previous_cost = env.simulation.state.total_transport_cost

            result = env._apply_discrete_action(
                DiscreteLogisticsAction(
                    dispatch=DispatchDecision.HOLD,
                    route=RouteDecision.SHORTEST,
                    mode=ModeDecision.SECONDARY_FLEET,
                    reorder=ReorderDecision.EMERGENCY,
                )
            )

            self.assertEqual(_total_on_hand(env), before)
            self.assertGreater(_total_in_transit(env), 0.0)
            self.assertGreater(result["dqn_replenishment_override_units"], 0.0)
            self.assertEqual(result["dqn_replenishment_override_received_units"], 0.0)
            self.assertGreater(result["dqn_replenishment_override_cost"], 0.0)
            self.assertEqual(_total_safety_stock(env), before_safety_stock)
            self.assertGreater(env.simulation.state.total_transport_cost, previous_cost)
            self.assertTrue(
                any(
                    event.get("event") == "inventory_replenishment"
                    and event.get("source") == "dqn_emergency_override"
                    and float(event.get("ordered_units", 0.0)) > 0.0
                    and float(event.get("received_units", -1.0)) == 0.0
                    and float(event.get("cost", 0.0)) > 0.0
                    for event in env.simulation.state.event_log
                )
            )
        finally:
            env.close()

    def test_reward_components_expose_planned_replenishment_cost(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            previous = env.simulation.snapshot()
            projection_info = env._apply_action(
                {
                    "continuous": [1.0, -1.0, 0.0, 0.0, -1.0],
                    "discrete": 0,
                }
            )
            current = env.simulation.snapshot()

            _, components = env._reward(previous, current, projection_info)

            self.assertGreater(components["planned_replenishment_cost"], 0.0)
            self.assertEqual(components["dqn_replenishment_override_cost"], 0.0)
            self.assertGreater(components["ppo_cost_delta"], 0.0)
        finally:
            env.close()

    def test_reward_components_expose_dqn_override_cost(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            previous = env.simulation.snapshot()
            projection_info = env._apply_action(
                {
                    "continuous": [-1.0, -1.0, 0.0, 0.0, -1.0],
                    "discrete": 3,
                }
            )
            current = env.simulation.snapshot()

            _, components = env._reward(previous, current, projection_info)

            self.assertEqual(components["planned_replenishment_cost"], 0.0)
            self.assertGreater(components["dqn_replenishment_override_cost"], 0.0)
            self.assertGreater(components["dqn_cost_delta"], 0.0)
        finally:
            env.close()

    def test_ppo_safety_stock_multiplier_updates_targets_without_dqn_override(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False))
        try:
            before = _total_safety_stock(env)
            planned = env._apply_physical_action(
                PhysicalAction(
                    reorder_fraction=0.0,
                    dispatch_intensity=0.0,
                    speed_multiplier=1.0,
                    safety_stock_multiplier=2.0,
                    capacity_buffer_fraction=0.0,
                ),
                allow_dispatch=False,
            )
            after_planned = _total_safety_stock(env)
            none_override = env._apply_discrete_action(
                DiscreteLogisticsAction(
                    dispatch=DispatchDecision.HOLD,
                    route=RouteDecision.SHORTEST,
                    mode=ModeDecision.SECONDARY_FLEET,
                    reorder=ReorderDecision.NONE,
                )
            )

            self.assertGreater(after_planned, before)
            self.assertEqual(planned["planned_replenishment_units"], 0.0)
            self.assertEqual(none_override["dqn_replenishment_override_units"], 0.0)
            self.assertEqual(none_override["dqn_replenishment_override_received_units"], 0.0)
            self.assertEqual(_total_safety_stock(env), after_planned)
        finally:
            env.close()


def _total_on_hand(env: FivePLDigitalTwinEnv) -> float:
    return sum(position.on_hand_units for position in env.simulation.inventory_network.positions.values())


def _total_in_transit(env: FivePLDigitalTwinEnv) -> float:
    return sum(position.in_transit_units for position in env.simulation.inventory_network.positions.values())


def _total_safety_stock(env: FivePLDigitalTwinEnv) -> float:
    return sum(position.safety_stock_units for position in env.simulation.inventory_network.positions.values())


if __name__ == "__main__":
    unittest.main()
