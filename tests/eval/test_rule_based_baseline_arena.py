from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.act.action_projector import ActionProjector
from src.eval.rule_based_baselines import ACTION_DISPATCH_SHORTEST_PRIMARY_NONE, RuleBasedDecision


class RuleBasedBaselineArenaTests(unittest.TestCase):
    def test_neutral_continuous_action_is_valid_joint_control(self) -> None:
        from src.eval.rule_based_baseline_arena import neutral_continuous_action

        action = neutral_continuous_action()

        self.assertEqual(action.shape, (5,))
        self.assertTrue(all(math.isfinite(float(value)) for value in action))
        self.assertTrue(all(-1.0 <= float(value) <= 1.0 for value in action))
        self.assertTrue(np.array_equal(action, np.zeros((5,), dtype=np.float32)))

    def test_neutral_continuous_action_projects_to_midpoint_physical_controls(self) -> None:
        from src.eval.rule_based_baseline_arena import neutral_continuous_action

        physical = ActionProjector().project(neutral_continuous_action())

        self.assertAlmostEqual(physical.reorder_fraction, 0.5)
        self.assertAlmostEqual(physical.dispatch_intensity, 0.5)
        self.assertAlmostEqual(physical.speed_multiplier, 1.0)
        self.assertAlmostEqual(physical.safety_stock_multiplier, 1.5)
        self.assertAlmostEqual(physical.capacity_buffer_fraction, 0.25)

    def test_continuous_baseline_modes_are_declared(self) -> None:
        from src.eval.rule_based_baseline_arena import ContinuousBaselineMode

        self.assertEqual(ContinuousBaselineMode.NEUTRAL.value, "neutral_continuous")
        self.assertEqual(ContinuousBaselineMode.HEURISTIC.value, "heuristic_continuous")
        self.assertEqual(ContinuousBaselineMode.PPO_ASSISTED.value, "ppo_assisted_continuous")
        self.assertEqual(ContinuousBaselineMode.ORACLE_DIAGNOSTIC.value, "oracle_diagnostic_continuous")

    def test_heuristic_continuous_action_is_valid_bounded_vector(self) -> None:
        from src.eval.rule_based_baseline_arena import ContinuousBaselineMode, build_continuous_action

        action = build_continuous_action(
            mode=ContinuousBaselineMode.HEURISTIC,
            context=_continuous_context(stockout_risk=0.4, safety_stock_gap=0.3, inventory_coverage=0.6),
        )

        self.assertEqual(action.shape, (5,))
        self.assertTrue(all(math.isfinite(float(value)) for value in action))
        self.assertTrue(all(-1.0 <= float(value) <= 1.0 for value in action))

    def test_heuristic_continuous_increases_inventory_posture_under_stock_pressure(self) -> None:
        from src.eval.rule_based_baseline_arena import ContinuousBaselineMode, build_continuous_action

        low = ActionProjector().project(
            build_continuous_action(
                mode=ContinuousBaselineMode.HEURISTIC,
                context=_continuous_context(stockout_risk=0.0, safety_stock_gap=0.0, inventory_coverage=1.0),
            )
        )
        high = ActionProjector().project(
            build_continuous_action(
                mode=ContinuousBaselineMode.HEURISTIC,
                context=_continuous_context(stockout_risk=0.95, safety_stock_gap=0.9, inventory_coverage=0.05),
            )
        )

        self.assertGreater(high.reorder_fraction, low.reorder_fraction)
        self.assertGreater(high.safety_stock_multiplier, low.safety_stock_multiplier)

    def test_heuristic_continuous_suppresses_reorder_when_holding_high_and_inventory_safe(self) -> None:
        from src.eval.rule_based_baseline_arena import ContinuousBaselineMode, build_continuous_action

        high_holding = ActionProjector().project(
            build_continuous_action(
                mode=ContinuousBaselineMode.HEURISTIC,
                context=_continuous_context(
                    stockout_risk=0.0,
                    safety_stock_gap=0.0,
                    inventory_coverage=1.0,
                    holding_cost_pressure=1.0,
                ),
            )
        )
        high_stock_pressure = ActionProjector().project(
            build_continuous_action(
                mode=ContinuousBaselineMode.HEURISTIC,
                context=_continuous_context(
                    stockout_risk=0.9,
                    safety_stock_gap=0.8,
                    inventory_coverage=0.1,
                    holding_cost_pressure=0.0,
                ),
            )
        )

        self.assertLess(high_holding.reorder_fraction, high_stock_pressure.reorder_fraction)
        self.assertLessEqual(high_holding.reorder_fraction, 0.05)

    def test_heuristic_dispatch_intensity_responds_to_useful_work_and_feasibility(self) -> None:
        from src.eval.rule_based_baseline_arena import ContinuousBaselineMode, build_continuous_action

        feasible = ActionProjector().project(
            build_continuous_action(
                mode=ContinuousBaselineMode.HEURISTIC,
                context=_continuous_context(primary_vehicle_available=True, secondary_vehicle_available=True),
            )
        )
        infeasible = ActionProjector().project(
            build_continuous_action(
                mode=ContinuousBaselineMode.HEURISTIC,
                context=_continuous_context(primary_vehicle_available=False, secondary_vehicle_available=False),
            )
        )

        self.assertGreater(feasible.dispatch_intensity, infeasible.dispatch_intensity)
        self.assertLessEqual(infeasible.dispatch_intensity, 0.05)

    def test_heuristic_capacity_buffer_increases_under_vehicle_scarcity(self) -> None:
        from src.eval.rule_based_baseline_arena import ContinuousBaselineMode, build_continuous_action

        normal = ActionProjector().project(
            build_continuous_action(
                mode=ContinuousBaselineMode.HEURISTIC,
                context=_continuous_context(primary_vehicle_available=True, secondary_vehicle_available=True),
            )
        )
        scarce = ActionProjector().project(
            build_continuous_action(
                mode=ContinuousBaselineMode.HEURISTIC,
                context=_continuous_context(primary_vehicle_available=False, secondary_vehicle_available=True),
            )
        )

        self.assertGreater(scarce.capacity_buffer_fraction, normal.capacity_buffer_fraction)

    def test_heuristic_capacity_buffer_increases_under_optional_capacity_shock_signal(self) -> None:
        from src.eval.rule_based_baseline_arena import ContinuousBaselineMode, build_continuous_action

        normal = ActionProjector().project(
            build_continuous_action(
                mode=ContinuousBaselineMode.HEURISTIC,
                context=_continuous_context_with_extra(capacity_shock_pressure=0.0),
            )
        )
        shock = ActionProjector().project(
            build_continuous_action(
                mode=ContinuousBaselineMode.HEURISTIC,
                context=_continuous_context_with_extra(capacity_shock_pressure=1.0),
            )
        )

        self.assertGreater(shock.capacity_buffer_fraction, normal.capacity_buffer_fraction)

    def test_heuristic_inventory_posture_responds_to_optional_lead_time_pressure(self) -> None:
        from src.eval.rule_based_baseline_arena import ContinuousBaselineMode, build_continuous_action

        low = ActionProjector().project(
            build_continuous_action(
                mode=ContinuousBaselineMode.HEURISTIC,
                context=_continuous_context_with_extra(lead_time_volatility_pressure=0.0),
            )
        )
        high = ActionProjector().project(
            build_continuous_action(
                mode=ContinuousBaselineMode.HEURISTIC,
                context=_continuous_context_with_extra(lead_time_volatility_pressure=0.95),
            )
        )

        self.assertGreater(high.reorder_fraction, low.reorder_fraction)
        self.assertGreater(high.safety_stock_multiplier, low.safety_stock_multiplier)

    def test_heuristic_route_pressure_changes_continuous_posture_not_discrete_route_id(self) -> None:
        from src.eval.rule_based_baseline_arena import ContinuousBaselineMode, build_joint_action_from_rule_decision

        low = build_joint_action_from_rule_decision(
            RuleBasedDecision(action_id=24),
            continuous_mode=ContinuousBaselineMode.HEURISTIC,
            continuous_context=_continuous_context(route_disruption_pressure=0.0, congestion_pressure=0.0),
        )
        high = build_joint_action_from_rule_decision(
            RuleBasedDecision(action_id=24),
            continuous_mode=ContinuousBaselineMode.HEURISTIC,
            continuous_context=_continuous_context(route_disruption_pressure=0.9, congestion_pressure=0.8),
        )

        low_physical = ActionProjector().project(low["continuous"])
        high_physical = ActionProjector().project(high["continuous"])
        self.assertEqual(low["discrete"], 24)
        self.assertEqual(high["discrete"], 24)
        self.assertGreater(high_physical.speed_multiplier, low_physical.speed_multiplier)
        self.assertGreater(high_physical.capacity_buffer_fraction, low_physical.capacity_buffer_fraction)

    def test_ppo_assisted_continuous_uses_injected_provider_and_validates_output(self) -> None:
        from src.eval.rule_based_baseline_arena import ContinuousBaselineMode, build_continuous_action

        context = _continuous_context()
        action = build_continuous_action(
            mode=ContinuousBaselineMode.PPO_ASSISTED,
            context=context,
            continuous_provider=lambda provided_context: np.array([0.1, 0.2, 0.0, -0.2, -0.1], dtype=np.float32),
        )

        self.assertTrue(np.allclose(action, np.array([0.1, 0.2, 0.0, -0.2, -0.1], dtype=np.float32)))
        with self.assertRaises(ValueError):
            build_continuous_action(
                mode=ContinuousBaselineMode.PPO_ASSISTED,
                context=context,
                continuous_provider=lambda _context: np.array([0.0, 0.0], dtype=np.float32),
            )
        with self.assertRaises(ValueError):
            build_continuous_action(
                mode=ContinuousBaselineMode.PPO_ASSISTED,
                context=context,
                continuous_provider=lambda _context: np.array([2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
        with self.assertRaises(ValueError):
            build_continuous_action(
                mode=ContinuousBaselineMode.PPO_ASSISTED,
                context=context,
                continuous_provider=lambda _context: np.array([math.nan, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )

    def test_oracle_diagnostic_mode_is_non_deployable_and_requires_explicit_allow(self) -> None:
        from src.eval.rule_based_baseline_arena import (
            ContinuousBaselineMode,
            build_continuous_action,
            continuous_mode_metadata,
        )

        metadata = continuous_mode_metadata(ContinuousBaselineMode.ORACLE_DIAGNOSTIC)

        self.assertFalse(metadata["deployable"])
        with self.assertRaises(ValueError):
            build_continuous_action(
                mode=ContinuousBaselineMode.ORACLE_DIAGNOSTIC,
                context=_continuous_context(),
            )
        allowed = build_continuous_action(
            mode=ContinuousBaselineMode.ORACLE_DIAGNOSTIC,
            context=_continuous_context(),
            allow_oracle_diagnostic=True,
        )
        self.assertEqual(allowed.shape, (5,))

    def test_build_joint_action_validates_discrete_action_bounds(self) -> None:
        from src.eval.rule_based_baseline_arena import build_joint_action_from_rule_decision

        joint = build_joint_action_from_rule_decision(
            RuleBasedDecision(
                action_id=ACTION_DISPATCH_SHORTEST_PRIMARY_NONE,
                selected_order_id="order-1",
                baseline_name="test",
                reason="unit",
            )
        )

        self.assertEqual(joint["discrete"], ACTION_DISPATCH_SHORTEST_PRIMARY_NONE)
        self.assertEqual(joint["continuous"].shape, (5,))

        with self.assertRaises(ValueError):
            build_joint_action_from_rule_decision(RuleBasedDecision(action_id=-1))
        with self.assertRaises(ValueError):
            build_joint_action_from_rule_decision(RuleBasedDecision(action_id=48))

    def test_context_adapter_extracts_dispatchable_orders_and_pressures(self) -> None:
        from src.eval.rule_based_baseline_arena import build_rule_context_from_environment

        env = _fake_env()

        context = build_rule_context_from_environment(env)

        self.assertEqual([order.order_id for order in context.dispatchable_orders], ["o1"])
        self.assertTrue(context.primary_vehicle_available)
        self.assertTrue(context.secondary_vehicle_available)
        self.assertGreaterEqual(context.route_disruption_pressure, 0.7)
        self.assertGreaterEqual(context.congestion_pressure, 0.6)
        self.assertAlmostEqual(context.stockout_risk, 0.4)
        self.assertAlmostEqual(context.holding_cost_pressure, 0.3)
        self.assertAlmostEqual(context.safety_stock_gap, 0.2)
        self.assertAlmostEqual(context.inventory_coverage, 0.8)

    def test_rule_episode_rollout_accepts_injected_fake_environment(self) -> None:
        from src.eval.rule_based_baseline_arena import run_rule_baseline_episode
        from src.eval.rule_based_baselines import build_rule_based_baselines
        from src.eval.real_world_scenario_arena import ScenarioConfig

        baseline = build_rule_based_baselines()["fifo_shortest_primary_none"]
        scenario = ScenarioConfig(
            scenario_id="unit_scenario",
            name="Unit",
            description="unit",
            seeds=(123,),
            episode_count=1,
        )

        metric = run_rule_baseline_episode(
            scenario,
            baseline,
            seed=123,
            episode_index=0,
            env_factory=lambda **_: (_FakeRolloutEnv(), {"synthetic": True}),
        )

        self.assertEqual(metric.scenario_id, "unit_scenario")
        self.assertEqual(metric.episode_index, 0)
        self.assertEqual(metric.steps, 1)
        self.assertEqual(metric.action_distribution[str(ACTION_DISPATCH_SHORTEST_PRIMARY_NONE)], 1)

    def test_rule_episode_rollout_accepts_heuristic_continuous_mode_without_writing_outputs(self) -> None:
        from src.eval.rule_based_baseline_arena import ContinuousBaselineMode, run_rule_baseline_episode
        from src.eval.rule_based_baselines import build_rule_based_baselines
        from src.eval.real_world_scenario_arena import ScenarioConfig

        env = _FakeRolloutEnv()
        baseline = build_rule_based_baselines()["fifo_shortest_primary_none"]
        scenario = ScenarioConfig(
            scenario_id="unit_scenario",
            name="Unit",
            description="unit",
            seeds=(123,),
            episode_count=1,
        )

        metric = run_rule_baseline_episode(
            scenario,
            baseline,
            seed=123,
            episode_index=0,
            env_factory=lambda **_: (env, {"synthetic": True}),
            continuous_mode=ContinuousBaselineMode.HEURISTIC,
        )

        self.assertEqual(metric.steps, 1)
        self.assertIsNotNone(env.last_action)
        self.assertEqual(env.last_action["continuous"].shape, (5,))

    def test_rule_episode_rollout_passes_current_observation_to_ppo_provider(self) -> None:
        from src.eval.rule_based_baseline_arena import ContinuousBaselineMode, run_rule_baseline_episode
        from src.eval.rule_based_baselines import build_rule_based_baselines
        from src.eval.real_world_scenario_arena import ScenarioConfig

        received_shapes = []

        def provider(observation):
            received_shapes.append(np.asarray(observation).shape)
            return np.array([0.2, 0.1, 0.0, -0.1, -0.2], dtype=np.float32)

        env = _FakeRolloutEnv()
        baseline = build_rule_based_baselines()["fifo_shortest_primary_none"]
        scenario = ScenarioConfig(
            scenario_id="unit_scenario",
            name="Unit",
            description="unit",
            seeds=(123,),
            episode_count=1,
        )

        run_rule_baseline_episode(
            scenario,
            baseline,
            seed=123,
            episode_index=0,
            env_factory=lambda **_: (env, {"synthetic": True}),
            continuous_mode=ContinuousBaselineMode.PPO_ASSISTED,
            continuous_provider=provider,
        )

        self.assertEqual(received_shapes, [(73,)])
        self.assertEqual(env.last_action["continuous"][0], np.float32(0.2))

    def test_write_outputs_refuses_existing_or_protected_output_dir(self) -> None:
        from src.eval.rule_based_baseline_arena import ensure_fresh_benchmark_output_dir

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "fresh"
            ensure_fresh_benchmark_output_dir(output_dir)
            self.assertTrue(output_dir.exists())

            with self.assertRaises(FileExistsError):
                ensure_fresh_benchmark_output_dir(output_dir)

        with self.assertRaises(ValueError):
            ensure_fresh_benchmark_output_dir(Path("models/eval/not_allowed_for_rule_baseline"))
        with self.assertRaises(ValueError):
            ensure_fresh_benchmark_output_dir(
                Path.cwd() / "models" / "production" / "not_allowed_for_rule_baseline"
            )

    def test_write_benchmark_outputs_creates_json_csv_and_markdown(self) -> None:
        from src.eval.rule_based_baseline_arena import write_rule_benchmark_outputs
        from src.eval.scenario_metrics import EpisodeMetrics

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "rule_report"
            metric = EpisodeMetrics(scenario_id="s1", episode_index=0, seed=7, steps=1)

            report = write_rule_benchmark_outputs(
                output_dir=output_dir,
                benchmark_metadata={"episodes": 1, "seed": 7},
                baseline_summaries=[
                    {
                        "baseline_name": "unit",
                        "scenario_id": "s1",
                        "episodes": 1,
                        "steps": 1,
                        "service_level": 1.0,
                        "mean_lateness": 0.0,
                        "dispatch_success_rate": 1.0,
                        "hard_blockers": {},
                    }
                ],
                episode_metrics=[("unit", metric)],
                comparison_to_hierarchical=[],
            )

            self.assertEqual(report["decision"], "RULE_BASED_BASELINE_BENCHMARK_READY")
            self.assertTrue((output_dir / "rule_based_baseline_report.json").exists())
            self.assertTrue((output_dir / "rule_based_baseline_summary.csv").exists())
            self.assertTrue((output_dir / "rule_based_baseline_episode_metrics.jsonl").exists())
            self.assertTrue((output_dir / "rule_based_baseline_report.md").exists())
            payload = json.loads((output_dir / "rule_based_baseline_report.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["decision"], "RULE_BASED_BASELINE_BENCHMARK_READY")

    def test_write_benchmark_outputs_refuses_existing_output_dir(self) -> None:
        from src.eval.rule_based_baseline_arena import write_rule_benchmark_outputs

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "existing"
            output_dir.mkdir()

            with self.assertRaises(FileExistsError):
                write_rule_benchmark_outputs(
                    output_dir=output_dir,
                    benchmark_metadata={"episodes": 1, "seed": 7},
                    baseline_summaries=[],
                    episode_metrics=[],
                    comparison_to_hierarchical=[],
                )


def _fake_env() -> SimpleNamespace:
    return SimpleNamespace(
        simulation=SimpleNamespace(
            env=SimpleNamespace(now=10.0),
            orders={
                "o1": SimpleNamespace(
                    order_id="o1",
                    created_at=1.0,
                    due_time=30.0,
                    total_units=4.0,
                    assigned_vehicle_id=None,
                    delivery_time=None,
                    status="created",
                    metadata={"premium": True},
                ),
                "o2": SimpleNamespace(
                    order_id="o2",
                    created_at=2.0,
                    due_time=20.0,
                    total_units=3.0,
                    assigned_vehicle_id="v1",
                    delivery_time=None,
                    status="in_transit",
                    metadata={},
                ),
                "o3": SimpleNamespace(
                    order_id="o3",
                    created_at=3.0,
                    due_time=25.0,
                    total_units=5.0,
                    assigned_vehicle_id=None,
                    delivery_time=None,
                    status="delayed",
                    metadata={},
                ),
            },
            vehicles={
                "v1": SimpleNamespace(
                    vehicle_id="v1",
                    tier="primary",
                    capacity_units=10.0,
                    current_load_units=0.0,
                    active=True,
                ),
                "v2": SimpleNamespace(
                    vehicle_id="v2",
                    tier="secondary",
                    capacity_units=10.0,
                    current_load_units=0.0,
                    active=True,
                ),
            },
        ),
        scenario_stress_state={
            "route_disruption_probability": 0.7,
            "congestion_multiplier": 2.8,
        },
        _stockout_risk=lambda _snapshot: 0.4,
        _holding_cost_pressure=lambda _snapshot: 0.3,
        _safety_stock_target_gap=lambda _snapshot: 0.2,
        _inventory_coverage=lambda _snapshot: 0.8,
    )


def _continuous_context(**overrides):
    from src.eval.rule_based_baselines import OrderView, RuleBasedDecisionContext

    values = {
        "dispatchable_orders": (
            OrderView(order_id="urgent", created_at=1.0, due_time=5.0, total_units=3.0, urgent=True),
            OrderView(order_id="normal", created_at=2.0, due_time=20.0, total_units=2.0),
        ),
        "primary_vehicle_available": True,
        "secondary_vehicle_available": True,
        "route_disruption_pressure": 0.0,
        "congestion_pressure": 0.0,
        "stockout_risk": 0.0,
        "holding_cost_pressure": 0.0,
        "safety_stock_gap": 0.0,
        "inventory_coverage": 1.0,
    }
    values.update(overrides)
    return RuleBasedDecisionContext(**values)


def _continuous_context_with_extra(**overrides):
    base = _continuous_context()
    values = {
        "dispatchable_orders": base.dispatchable_orders,
        "primary_vehicle_available": base.primary_vehicle_available,
        "secondary_vehicle_available": base.secondary_vehicle_available,
        "route_disruption_pressure": base.route_disruption_pressure,
        "congestion_pressure": base.congestion_pressure,
        "stockout_risk": base.stockout_risk,
        "holding_cost_pressure": base.holding_cost_pressure,
        "safety_stock_gap": base.safety_stock_gap,
        "inventory_coverage": base.inventory_coverage,
        "capacity_shock_pressure": 0.0,
        "lead_time_volatility_pressure": 0.0,
        "supplier_delay_pressure": 0.0,
        "scenario_lead_time_volatility_pressure": 0.0,
        "vehicle_availability_pressure": 0.0,
        "feasible_dispatch_ratio": None,
        "pending_work_pressure": None,
        "lateness_risk": 0.0,
        "premium_sla_pressure": 0.0,
        "useful_dispatch_opportunity": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeRolloutEnv:
    def __init__(self) -> None:
        self.simulation = _fake_env().simulation
        self.scenario_stress_state = _fake_env().scenario_stress_state
        self._closed = False
        self.last_action = None

    def reset(self, *, seed: int | None = None):
        return [0.0] * 73, {}

    def step(self, action):
        self.last_action = action
        info = {
            "projected_action": {
                "dispatch": "dispatch",
                "route": "shortest",
                "mode": "primary_fleet",
                "reorder": "none",
                "selected_order_id": "o1",
                "dispatch_success": True,
                "dispatch_failure_reason": None,
            },
            "raw_action": {"discrete": int(action["discrete"])},
            "service_level": 1.0,
            "lateness": 0.0,
        }
        return [0.0] * 73, 1.0, True, False, info

    def close(self) -> None:
        self._closed = True


if __name__ == "__main__":
    unittest.main()
