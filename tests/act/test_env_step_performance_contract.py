from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import numpy as np

import src.act.env_5pl as env_5pl_module
from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv
from src.act.observation_builder import OBSERVATION_DIM
from src.learn.train_joint_torch import MDP_CONTRACT_VERSION


class EnvStepPerformanceContractTest(unittest.TestCase):
    def test_compact_info_mode_omits_snapshot_and_avoids_full_snapshot_deepcopy(self) -> None:
        env = FivePLDigitalTwinEnv(
            EnvironmentConfig(info_mode="compact", rolling_demand_enabled=False, random_seed=31)
        )
        try:
            env.reset(seed=31)

            with patch.object(env_5pl_module.copy, "deepcopy", side_effect=AssertionError("deepcopy should not run")):
                observation, reward, terminated, truncated, info = env.step(_neutral_joint_action())

            self.assertEqual(observation.shape, (OBSERVATION_DIM,))
            self.assertTrue(np.isfinite(reward))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertNotIn("snapshot", info)
            self.assertIn("projected_action", info)
            self.assertIn("reward_components", info)

            components = info["reward_components"]
            for field in _COMPACT_REWARD_COMPONENT_FIELDS:
                self.assertIn(field, components)
        finally:
            env.close()

    def test_full_info_mode_keeps_snapshot_and_protects_internal_previous_snapshot(self) -> None:
        env = FivePLDigitalTwinEnv(
            EnvironmentConfig(info_mode="full", rolling_demand_enabled=False, random_seed=32)
        )
        try:
            env.reset(seed=32)
            _observation, _reward, _terminated, _truncated, info = env.step(_neutral_joint_action())

            self.assertIn("snapshot", info)
            self.assertIsNot(info["snapshot"], env._previous_snapshot)

            original_time = env._previous_snapshot["time"]
            info["snapshot"]["time"] = -1.0
            self.assertEqual(env._previous_snapshot["time"], original_time)
        finally:
            env.close()

    def test_step_reuses_same_state_snapshots(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False, random_seed=11))
        try:
            env.reset(seed=11)
            original_snapshot: Callable[[], dict[str, Any]] = env.simulation.snapshot
            snapshot_calls = 0

            def counted_snapshot() -> dict[str, Any]:
                nonlocal snapshot_calls
                snapshot_calls += 1
                return original_snapshot()

            env.simulation.snapshot = counted_snapshot  # type: ignore[method-assign]

            observation, reward, terminated, truncated, info = env.step(_neutral_joint_action())

            self.assertEqual(observation.shape, (OBSERVATION_DIM,))
            self.assertTrue(np.isfinite(observation).all())
            self.assertTrue(np.isfinite(reward))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertIn("reward_components", info)
            self.assertIn("snapshot", info)
            self.assertEqual(snapshot_calls, 1)
        finally:
            env.close()

    def test_same_seed_action_sequence_is_deterministic_and_info_snapshot_is_independent(self) -> None:
        left = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False, random_seed=21))
        right = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False, random_seed=21))
        try:
            left_observation, _ = left.reset(seed=21)
            right_observation, _ = right.reset(seed=21)
            np.testing.assert_allclose(left_observation, right_observation, rtol=0.0, atol=0.0)

            actions = [_neutral_joint_action(), _neutral_joint_action(), _neutral_joint_action()]
            for action in actions:
                left_step = left.step(action)
                right_step = right.step(action)

                np.testing.assert_allclose(left_step[0], right_step[0], rtol=0.0, atol=0.0)
                self.assertEqual(left_step[1], right_step[1])
                self.assertEqual(left_step[2], right_step[2])
                self.assertEqual(left_step[3], right_step[3])

                left_info = left_step[4]
                right_info = right_step[4]
                self.assertEqual(left_info["reward_components"], right_info["reward_components"])
                self.assertEqual(left_info["projected"], right_info["projected"])
                self.assertEqual(left_info["blocked"], right_info["blocked"])
                self.assertEqual(left_info["projected_action"], right_info["projected_action"])
                self.assertEqual(
                    _snapshot_summary(left_info["snapshot"]),
                    _snapshot_summary(right_info["snapshot"]),
                )
                self.assertIsNot(left_info["snapshot"], left._previous_snapshot)

                original_time = left._previous_snapshot["time"]
                left_info["snapshot"]["time"] = -1.0
                self.assertEqual(left._previous_snapshot["time"], original_time)
        finally:
            left.close()
            right.close()

    def test_step_local_order_flow_cache_does_not_leak_between_steps_or_reset(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False, random_seed=12))
        try:
            env.reset(seed=12)
            env.step(_neutral_joint_action())
            self.assertIsNone(env._step_order_flow_metrics_cache)

            env.step(_neutral_joint_action())
            self.assertIsNone(env._step_order_flow_metrics_cache)

            env.reset(seed=13)
            self.assertIsNone(env._step_order_flow_metrics_cache)
        finally:
            env.close()

    def test_event_float_sum_cache_reuses_same_step_event_scan_and_tracks_keys(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False, random_seed=14))
        snapshot = {
            "event_log": [
                {"event": "route_arc_traversed", "time": 5.0, "speed_cost": 1.25, "distance_cost": 10.0},
                {"event": "route_arc_traversed", "time": 7.0, "speed_cost": 2.75, "distance_cost": 15.0},
                {"event": "other", "time": 9.0, "speed_cost": 99.0, "distance_cost": 99.0},
                {"event": "route_arc_traversed", "time": 1.0, "speed_cost": 4.0, "distance_cost": 20.0},
            ],
        }
        try:
            env._step_event_float_sum_cache = {}

            speed_cost = env._event_float_sum_since(
                snapshot,
                after_time=3.0,
                event_name="route_arc_traversed",
                key="speed_cost",
            )
            distance_cost = env._event_float_sum_since(
                snapshot,
                after_time=3.0,
                event_name="route_arc_traversed",
                key="distance_cost",
            )

            self.assertAlmostEqual(speed_cost, 4.0)
            self.assertAlmostEqual(distance_cost, 25.0)
            self.assertEqual(len(env._step_event_float_sum_cache), 1)
        finally:
            env._step_event_float_sum_cache = None
            env.close()

    def test_event_float_sum_cache_key_changes_when_snapshot_event_log_grows(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False, random_seed=15))
        event_log = [
            {"event": "route_arc_traversed", "time": 5.0, "speed_cost": 1.0},
        ]
        snapshot = {"event_log": event_log}
        try:
            env._step_event_float_sum_cache = {}

            first = env._event_float_sum_since(
                snapshot,
                after_time=0.0,
                event_name="route_arc_traversed",
                key="speed_cost",
            )
            event_log.append({"event": "route_arc_traversed", "time": 6.0, "speed_cost": 2.0})
            second = env._event_float_sum_since(
                snapshot,
                after_time=0.0,
                event_name="route_arc_traversed",
                key="speed_cost",
            )

            self.assertAlmostEqual(first, 1.0)
            self.assertAlmostEqual(second, 3.0)
            self.assertEqual(len(env._step_event_float_sum_cache), 2)
        finally:
            env._step_event_float_sum_cache = None
            env.close()

    def test_contract_dimensions_remain_unchanged(self) -> None:
        env = FivePLDigitalTwinEnv(EnvironmentConfig(rolling_demand_enabled=False, random_seed=14))
        try:
            observation, _ = env.reset(seed=14)

            self.assertEqual(OBSERVATION_DIM, 73)
            self.assertEqual(observation.shape, (73,))
            self.assertEqual(DISCRETE_ACTION_COUNT, 48)
            self.assertEqual(env.action_space["discrete"].n, 48)
            self.assertEqual(MDP_CONTRACT_VERSION, "physical_reality_v5_route_candidate_visibility")
        finally:
            env.close()


def _neutral_joint_action() -> dict[str, Any]:
    return {
        "continuous": [-1.0, -1.0, 0.0, -1.0, -1.0][:CONTINUOUS_ACTION_DIM],
        "discrete": 0,
    }


_COMPACT_REWARD_COMPONENT_FIELDS = (
    "global",
    "ppo_local",
    "dqn_local",
    "dqn_dispatched_orders",
    "dispatch_success_count",
    "dispatch_no_unassigned_orders",
    "dispatch_no_vehicle_available",
    "dispatch_already_assigned_count",
    "current_dispatch_work_count",
    "current_dispatch_work_signal",
    "dqn_delivery_credit",
    "dqn_delivery_credit_blocked_no_current_dispatch_work",
    "hold_inflight_completion_credit_blocked_for_route",
    "action24_25_candidate_credit_allowed",
    "action24_25_candidate_credit_blocked_reason",
    "premium_sla_fleet_credit_blocked_no_current_work",
    "premium_primary_adaptation_credit_blocked_no_current_work",
    "premium_fleet_credit_allowed",
)


def _snapshot_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    orders = snapshot.get("orders", {})
    vehicles = snapshot.get("vehicles", {})
    return {
        "time": snapshot.get("time"),
        "order_statuses": sorted(order.get("status") for order in orders.values()),
        "vehicle_tiers": sorted(vehicle.get("tier") for vehicle in vehicles.values()),
        "service_level": snapshot.get("service_level"),
        "total_transport_cost": snapshot.get("total_transport_cost"),
        "event_count": len(snapshot.get("event_log", ())),
    }
