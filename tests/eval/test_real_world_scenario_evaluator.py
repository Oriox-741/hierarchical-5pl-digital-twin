from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from src.eval.evaluate_real_world_scenarios import main
from src.eval.real_world_scenario_arena import (
    EXPECTED_MDP_CONTRACT,
    EXPECTED_OBSERVATION_DIM,
    ScenarioConfig,
    load_policy_checkpoint,
    validate_checkpoint_metadata,
    write_evaluation_outputs,
)
from src.eval.scenario_metrics import (
    EpisodeMetricAccumulator,
    evaluate_thresholds,
    evaluate_warnings,
    summarize_episodes,
)
from src.think.joint_policies import HIERARCHICAL_DQN_ARCHITECTURE, JointPolicyBundle


def _write_checkpoint(
    path: Path,
    *,
    contract: str = EXPECTED_MDP_CONTRACT,
    observation_dim: int = EXPECTED_OBSERVATION_DIM,
    dqn_architecture: str | None = None,
) -> None:
    policy = JointPolicyBundle(
        observation_dim=observation_dim,
        **({"dqn_architecture": dqn_architecture} if dqn_architecture is not None else {}),
    )
    payload = policy.build_checkpoint(
        global_step=3_000_000,
        config={
            "mdp_contract_version": contract,
            "shared_global_parameters": {
                "observation_dim": observation_dim,
                "max_steps": 2,
                "decision_interval_seconds": 300,
                "max_extra_dispatch_budget": 4,
                "feasibility_gated_macro_budget": True,
            },
            "reward_physics": {
                "infeasible_dispatch_penalty_weight": 0.04,
            },
        },
    )
    torch.save(payload, path)


def _write_scenario(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "scenario_id": "tiny_baseline",
                "name": "Tiny Baseline",
                "description": "Small dry-run scenario.",
                "seeds": [7],
                "episode_count": 1,
                "environment_overrides": {"service_level_target": 0.95},
                "unsupported_environment_overrides": {
                    "route_disruption_probability": "unsupported test knob"
                },
                "expected_behavior": "dry-run only",
                "pass_fail_thresholds": {
                    "fail_on_fake_dispatch_credit": True,
                    "fail_on_nan_inf": True
                },
            }
        ),
        encoding="utf-8",
    )


class RealWorldScenarioEvaluatorTests(unittest.TestCase):
    def test_checkpoint_metadata_validation_rejects_wrong_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            validate_checkpoint_metadata(
                {
                    "config": {
                        "mdp_contract_version": "old_contract",
                        "shared_global_parameters": {"observation_dim": EXPECTED_OBSERVATION_DIM},
                    },
                    "observation_dim": EXPECTED_OBSERVATION_DIM,
                }
            )

    def test_checkpoint_metadata_validation_rejects_wrong_observation_dim(self) -> None:
        with self.assertRaisesRegex(ValueError, "observation_dim mismatch"):
            validate_checkpoint_metadata(
                {
                    "config": {
                        "mdp_contract_version": EXPECTED_MDP_CONTRACT,
                        "shared_global_parameters": {"observation_dim": 44},
                    },
                    "observation_dim": 44,
                }
            )

    def test_load_policy_checkpoint_sets_modules_to_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "candidate.pt"
            _write_checkpoint(checkpoint)

            loaded = load_policy_checkpoint(checkpoint, device=torch.device("cpu"))

            self.assertFalse(loaded.policies.training)
            self.assertEqual(loaded.checkpoint["global_step"], 3_000_000)

    def test_load_policy_checkpoint_preserves_hierarchical_dqn_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "candidate.pt"
            _write_checkpoint(checkpoint, dqn_architecture=HIERARCHICAL_DQN_ARCHITECTURE)

            loaded = load_policy_checkpoint(checkpoint, device=torch.device("cpu"))

            self.assertFalse(loaded.policies.training)
            self.assertEqual(HIERARCHICAL_DQN_ARCHITECTURE, loaded.policies.dqn_architecture)

    def test_dry_run_writes_only_requested_outputs_and_does_not_mutate_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "candidate.pt"
            scenario_dir = root / "scenarios"
            output_dir = root / "reports"
            scenario_dir.mkdir()
            _write_checkpoint(checkpoint)
            _write_scenario(scenario_dir / "tiny.json")
            before_stat = checkpoint.stat()
            time.sleep(0.01)

            result = main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--scenario-dir",
                    str(scenario_dir),
                    "--output-dir",
                    str(output_dir),
                    "--episodes",
                    "1",
                    "--seed",
                    "7",
                    "--device",
                    "cpu",
                    "--deterministic",
                    "--dry-run",
                ]
            )

            self.assertEqual(result, 0)
            self.assertEqual(checkpoint.stat().st_size, before_stat.st_size)
            self.assertEqual(checkpoint.stat().st_mtime_ns, before_stat.st_mtime_ns)
            self.assertTrue((output_dir / "scenario_summary.csv").exists())
            self.assertTrue((output_dir / "scenario_summary.json").exists())
            self.assertTrue((output_dir / "episode_metrics.jsonl").exists())
            self.assertTrue((output_dir / "real_world_evaluation_report.md").exists())
            self.assertFalse((root / "models" / "registry").exists())
            unexpected_pt_files = [path for path in root.rglob("*.pt") if path != checkpoint]
            self.assertEqual(unexpected_pt_files, [])

    def test_dry_run_does_not_call_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "candidate.pt"
            scenario_dir = root / "scenarios"
            output_dir = root / "reports"
            scenario_dir.mkdir()
            _write_checkpoint(checkpoint)
            _write_scenario(scenario_dir / "tiny.json")

            def fail_step(*_args: object, **_kwargs: object) -> None:
                raise AssertionError("optimizer.step must not be called by evaluation")

            with patch("torch.optim.Optimizer.step", side_effect=fail_step):
                main(
                    [
                        "--checkpoint",
                        str(checkpoint),
                        "--scenario-dir",
                        str(scenario_dir),
                        "--output-dir",
                        str(output_dir),
                        "--episodes",
                        "1",
                        "--seed",
                        "7",
                        "--device",
                        "cpu",
                        "--dry-run",
                    ]
                )


class ScenarioMetricAlignmentTests(unittest.TestCase):
    def test_delivered_delta_aggregation_reports_final_sum_and_step_mean(self) -> None:
        accumulator = EpisodeMetricAccumulator(scenario_id="baseline", seed=1, episode_index=0)

        for delivered_delta in (1.0, 0.0, 2.0, 0.0):
            accumulator.observe(
                reward=0.0,
                info={
                    "raw_action": {"discrete": 0},
                    "projected_action": {"discrete": {"dispatch": "hold"}},
                    "reward_components": {"delivered_delta": delivered_delta},
                },
            )

        episode = accumulator.finish()
        summary = summarize_episodes("baseline", [episode])

        self.assertEqual(episode.final_step_delivered_delta, 0.0)
        self.assertEqual(episode.episode_delivered_delta_sum, 3.0)
        self.assertEqual(episode.mean_step_delivered_delta, 0.75)
        self.assertEqual(episode.delivered_delta, 0.75)
        self.assertEqual(summary["final_step_delivered_delta"], 0.0)
        self.assertEqual(summary["episode_delivered_delta_sum"], 3.0)
        self.assertEqual(summary["mean_step_delivered_delta"], 0.75)
        self.assertEqual(summary["delivered_delta"], 0.75)

    def test_legacy_min_delivered_delta_threshold_uses_mean_step_delivery(self) -> None:
        healthy = {
            "service_level": 0.96,
            "true_lateness_pressure": 0.0,
            "dispatch_success_per_attempt": 0.9,
            "final_step_delivered_delta": 0.0,
            "episode_delivered_delta_sum": 3.0,
            "mean_step_delivered_delta": 0.75,
            "delivered_delta": 0.75,
        }
        collapsed = {**healthy, "mean_step_delivered_delta": 0.20, "delivered_delta": 0.20}

        self.assertEqual(
            evaluate_thresholds(healthy, {"min_delivered_delta": 0.45}),
            [],
        )
        self.assertIn(
            "mean_step_delivered_delta 0.200 < legacy min_delivered_delta 0.450",
            evaluate_thresholds(collapsed, {"min_delivered_delta": 0.45}),
        )

    def test_hard_blocker_no_current_work_delivery_credit_counter_increments(self) -> None:
        accumulator = EpisodeMetricAccumulator(scenario_id="baseline", seed=1, episode_index=0)

        accumulator.observe(
            reward=0.0,
            info={
                "raw_action": {"discrete": 1},
                "projected_action": {"discrete": {"dispatch": "dispatch"}},
                "reward_components": {
                    "current_dispatch_work_signal": 0.0,
                    "dqn_delivery_credit": 0.1,
                },
            },
        )

        self.assertEqual(accumulator.finish().no_current_work_dqn_delivery_credit, 1)

    def test_hard_blocker_hold_delivery_credit_leak_counter_increments(self) -> None:
        accumulator = EpisodeMetricAccumulator(scenario_id="baseline", seed=1, episode_index=0)

        accumulator.observe(
            reward=0.0,
            info={
                "raw_action": {"discrete": 0},
                "projected_action": {"discrete": {"dispatch": "hold"}},
                "reward_components": {"dqn_delivery_credit": 0.1},
            },
        )

        self.assertEqual(accumulator.finish().hold_delivery_credit_leak, 1)

    def test_hard_blocker_action8_route_credit_leak_counter_increments(self) -> None:
        accumulator = EpisodeMetricAccumulator(scenario_id="baseline", seed=1, episode_index=0)

        accumulator.observe(
            reward=0.0,
            info={
                "raw_action": {"discrete": 8},
                "projected_action": {
                    "discrete": {
                        "dispatch": "hold",
                        "route": "low_congestion",
                        "mode": "secondary_fleet",
                        "reorder": "none",
                    }
                },
                "reward_components": {"route_candidate_alignment_credit": 0.1},
            },
        )

        self.assertEqual(accumulator.finish().action8_route_or_delivery_credit_leak, 1)

    def test_hard_blocker_unsafe_24_25_candidate_credit_counter_increments(self) -> None:
        accumulator = EpisodeMetricAccumulator(scenario_id="baseline", seed=1, episode_index=0)

        accumulator.observe(
            reward=0.0,
            info={
                "raw_action": {"discrete": 24},
                "projected_action": {
                    "discrete": {
                        "dispatch": "dispatch",
                        "route": "shortest",
                        "mode": "secondary_fleet",
                        "reorder": "none",
                    }
                },
                "reward_components": {
                    "action24_25_candidate_credit_allowed": 0.0,
                    "route_candidate_alignment_credit": 0.1,
                },
            },
        )

        self.assertEqual(accumulator.finish().unsafe_24_25_candidate_credit, 1)

    def test_hard_blocker_no_work_positive_dqn_local_counter_increments(self) -> None:
        accumulator = EpisodeMetricAccumulator(scenario_id="baseline", seed=1, episode_index=0)

        accumulator.observe(
            reward=0.0,
            info={
                "raw_action": {"discrete": 0},
                "projected_action": {"discrete": {"dispatch": "hold"}},
                "reward_components": {
                    "current_dispatch_work_count": 0.0,
                    "current_dispatch_work_signal": 0.0,
                    "dqn_local": 0.1,
                },
            },
        )

        self.assertEqual(accumulator.finish().no_work_positive_dqn_local, 1)

    def test_hard_blocker_emergency_zero_useful_positive_credit_counter_increments(self) -> None:
        accumulator = EpisodeMetricAccumulator(scenario_id="baseline", seed=1, episode_index=0)

        accumulator.observe(
            reward=0.0,
            info={
                "raw_action": {"discrete": 47},
                "projected_action": {"discrete": {"dispatch": "dispatch", "reorder": "emergency"}},
                "reward_components": {
                    "emergency_useful_factor": 0.0,
                    "emergency_procurement_justification_credit": 0.1,
                },
            },
        )

        self.assertEqual(accumulator.finish().emergency_zero_useful_positive_credit, 1)

    def test_premium_guard_telemetry_counters_increment(self) -> None:
        accumulator = EpisodeMetricAccumulator(
            scenario_id="premium_sla_pressure",
            seed=102,
            episode_index=0,
        )

        accumulator.observe(
            reward=0.0,
            info={
                "projected_action": {
                    "discrete": {
                        "dispatch": "hold",
                        "route": "high_resilience",
                        "mode": "primary_fleet",
                        "reorder": "conservative",
                    }
                },
                "reward_components": {
                    "premium_sla_fleet_credit_blocked_no_current_work": 1.0,
                    "premium_primary_adaptation_credit_blocked_no_current_work": 1.0,
                    "premium_fleet_credit_allowed": 0.0,
                },
            },
        )
        accumulator.observe(
            reward=0.1,
            info={
                "projected_action": {
                    "discrete": {
                        "dispatch": "dispatch",
                        "route": "shortest",
                        "mode": "primary_fleet",
                        "reorder": "conservative",
                    }
                },
                "reward_components": {
                    "premium_sla_fleet_credit_blocked_no_current_work": 0.0,
                    "premium_primary_adaptation_credit_blocked_no_current_work": 0.0,
                    "premium_fleet_credit_allowed": 1.0,
                },
            },
        )

        episode = accumulator.finish()

        self.assertEqual(episode.premium_sla_fleet_credit_blocked_no_current_work, 1)
        self.assertEqual(episode.premium_primary_adaptation_credit_blocked_no_current_work, 1)
        self.assertEqual(episode.premium_fleet_credit_allowed, 1)
        self.assertEqual(
            episode.to_dict()["premium_sla_fleet_credit_blocked_no_current_work"],
            1,
        )
        self.assertEqual(
            episode.to_dict()["premium_primary_adaptation_credit_blocked_no_current_work"],
            1,
        )
        self.assertEqual(episode.to_dict()["premium_fleet_credit_allowed"], 1)

    def test_hard_blocker_counters_summarize_and_fail_thresholds(self) -> None:
        episodes = [
            EpisodeMetricAccumulator(scenario_id="baseline", seed=1, episode_index=0).finish(),
            EpisodeMetricAccumulator(scenario_id="baseline", seed=2, episode_index=1).finish(),
        ]
        episodes[0].no_current_work_dqn_delivery_credit = 1
        episodes[0].hold_delivery_credit_leak = 2
        episodes[1].action8_route_or_delivery_credit_leak = 3
        episodes[1].unsafe_24_25_candidate_credit = 4
        episodes[1].no_work_positive_dqn_local = 5
        episodes[1].emergency_zero_useful_positive_credit = 6

        summary = summarize_episodes("baseline", episodes)

        self.assertEqual(summary["no_current_work_dqn_delivery_credit"], 1)
        self.assertEqual(summary["hold_delivery_credit_leak"], 2)
        self.assertEqual(summary["action8_route_or_delivery_credit_leak"], 3)
        self.assertEqual(summary["unsafe_24_25_candidate_credit"], 4)
        self.assertEqual(summary["no_work_positive_dqn_local"], 5)
        self.assertEqual(summary["emergency_zero_useful_positive_credit"], 6)
        failures = evaluate_thresholds(
            summary,
            {
                "fail_on_no_current_work_dqn_delivery_credit": True,
                "fail_on_hold_delivery_credit_leak": True,
                "fail_on_action8_route_or_delivery_credit_leak": True,
                "fail_on_unsafe_24_25_candidate_credit": True,
                "fail_on_no_work_positive_dqn_local": True,
                "fail_on_emergency_zero_useful_positive_credit": True,
            },
        )
        self.assertIn("no-current-work dqn_delivery_credit present: 1", failures)
        self.assertIn("hold route/delivery credit leak present: 2", failures)
        self.assertIn("action8 route/delivery credit leak present: 3", failures)
        self.assertIn("unsafe 24/25 candidate credit present: 4", failures)
        self.assertIn("no-work positive dqn_local present: 5", failures)
        self.assertIn("emergency zero-useful positive credit present: 6", failures)

    def test_global_hard_blocker_fails_without_scenario_flag(self) -> None:
        summary = {
            "service_level": 0.96,
            "true_lateness_pressure": 0.0,
            "dispatch_success_per_attempt": 0.95,
            "mean_step_delivered_delta": 0.75,
            "delivered_delta": 0.75,
            "no_work_positive_dqn_local": 1,
        }

        failures = evaluate_thresholds(summary, {})

        self.assertIn("no-work positive dqn_local present: 1", failures)

    def test_zero_global_hard_blockers_can_pass_normal_thresholds(self) -> None:
        summary = {
            "service_level": 0.96,
            "true_lateness_pressure": 0.0,
            "dispatch_success_per_attempt": 0.95,
            "mean_step_delivered_delta": 0.75,
            "delivered_delta": 0.75,
            "fake_dispatch_credit": 0,
            "customer_revisited": 0,
            "route_failure_positive_dispatch_credit": 0,
            "nan_inf_detected": False,
            "dqn_local_negative_positive_train_rows": 0,
            "no_current_work_dqn_delivery_credit": 0,
            "hold_delivery_credit_leak": 0,
            "action8_route_or_delivery_credit_leak": 0,
            "unsafe_24_25_candidate_credit": 0,
            "no_work_positive_dqn_local": 0,
            "emergency_zero_useful_positive_credit": 0,
        }

        self.assertEqual(evaluate_thresholds(summary, {"min_service_level": 0.93}), [])

    def test_evaluation_outputs_include_delivered_and_hard_blocker_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "reports"
            scenario = ScenarioConfig(
                scenario_id="baseline",
                name="Baseline",
                description="Synthetic report fixture.",
                seeds=[1],
                episode_count=1,
                environment_overrides={},
                expected_behavior="fixture",
                pass_fail_thresholds={
                    "fail_on_no_current_work_dqn_delivery_credit": True,
                },
            )
            episode = EpisodeMetricAccumulator(scenario_id="baseline", seed=1, episode_index=0).finish()
            episode.final_step_delivered_delta = 0.0
            episode.episode_delivered_delta_sum = 3.0
            episode.mean_step_delivered_delta = 0.75
            episode.delivered_delta = 0.75
            episode.no_current_work_dqn_delivery_credit = 1

            write_evaluation_outputs(
                output_dir=output_dir,
                checkpoint_path=Path("candidate.pt"),
                scenarios=[scenario],
                episode_rows=[episode],
                override_capabilities={},
            )

            episode_json = json.loads((output_dir / "episode_metrics.jsonl").read_text(encoding="utf-8"))
            summary_json = json.loads((output_dir / "scenario_summary.json").read_text(encoding="utf-8"))
            csv_text = (output_dir / "scenario_summary.csv").read_text(encoding="utf-8")
            report_text = (output_dir / "real_world_evaluation_report.md").read_text(encoding="utf-8")
            scenario_summary = summary_json["scenarios"][0]

            self.assertEqual(episode_json["final_step_delivered_delta"], 0.0)
            self.assertEqual(episode_json["episode_delivered_delta_sum"], 3.0)
            self.assertEqual(episode_json["mean_step_delivered_delta"], 0.75)
            self.assertEqual(scenario_summary["mean_step_delivered_delta"], 0.75)
            self.assertIn("mean_step_delivered_delta", csv_text)
            self.assertIn("no_current_work_dqn_delivery_credit", csv_text)
            self.assertIn("Hard-Blocker Telemetry", report_text)
            self.assertIn("no_current_work_dqn_delivery_credit", report_text)

    def test_evaluation_outputs_include_premium_guard_telemetry_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "reports"
            scenario = ScenarioConfig(
                scenario_id="premium_sla_pressure",
                name="Premium SLA",
                description="Synthetic premium guard telemetry fixture.",
                seeds=[102],
                episode_count=1,
                environment_overrides={},
                expected_behavior="fixture",
                pass_fail_thresholds={},
            )
            episode = EpisodeMetricAccumulator(
                scenario_id="premium_sla_pressure",
                seed=102,
                episode_index=0,
            ).finish()
            episode.premium_sla_fleet_credit_blocked_no_current_work = 2
            episode.premium_primary_adaptation_credit_blocked_no_current_work = 3
            episode.premium_fleet_credit_allowed = 5

            write_evaluation_outputs(
                output_dir=output_dir,
                checkpoint_path=Path("candidate.pt"),
                scenarios=[scenario],
                episode_rows=[episode],
                override_capabilities={},
            )

            episode_json = json.loads((output_dir / "episode_metrics.jsonl").read_text(encoding="utf-8"))
            summary_json = json.loads((output_dir / "scenario_summary.json").read_text(encoding="utf-8"))
            csv_text = (output_dir / "scenario_summary.csv").read_text(encoding="utf-8")
            report_text = (output_dir / "real_world_evaluation_report.md").read_text(encoding="utf-8")
            scenario_summary = summary_json["scenarios"][0]

            self.assertEqual(episode_json["premium_sla_fleet_credit_blocked_no_current_work"], 2)
            self.assertEqual(episode_json["premium_primary_adaptation_credit_blocked_no_current_work"], 3)
            self.assertEqual(episode_json["premium_fleet_credit_allowed"], 5)
            self.assertEqual(scenario_summary["premium_sla_fleet_credit_blocked_no_current_work"], 2)
            self.assertEqual(scenario_summary["premium_primary_adaptation_credit_blocked_no_current_work"], 3)
            self.assertEqual(scenario_summary["premium_fleet_credit_allowed"], 5)
            self.assertIn("premium_sla_fleet_credit_blocked_no_current_work", csv_text)
            self.assertIn("premium_primary_adaptation_credit_blocked_no_current_work", csv_text)
            self.assertIn("premium_fleet_credit_allowed", csv_text)
            self.assertIn("Premium Reward Guard Telemetry", report_text)
            self.assertIn("premium_sla_pressure", report_text)
            self.assertIn("2", report_text)
            self.assertIn("3", report_text)
            self.assertIn("5", report_text)

    def test_routepremium_action_quality_and_route_split_telemetry(self) -> None:
        accumulator = EpisodeMetricAccumulator(
            scenario_id="route_disruption_congestion",
            seed=92,
            episode_index=0,
        )

        accumulator.observe(
            reward=-0.2,
            info={
                "raw_action": {"discrete": 31},
                "projected_action": {
                    "discrete": {
                        "dispatch": "dispatch",
                        "route": "shortest",
                        "mode": "primary_fleet",
                        "reorder": "emergency",
                    }
                },
                "reward_components": {
                    "dispatch_attempted": 1,
                    "dispatch_success_count": 0,
                    "dispatch_no_unassigned_orders": 1,
                    "dispatch_no_vehicle_available": 0,
                    "dispatch_route_failure": 1,
                    "dispatch_already_assigned_count": 3,
                    "current_dispatch_work_count": 0,
                    "current_dispatch_work_signal": 0,
                    "dispatch_progress_credit": 0,
                },
            },
        )
        accumulator.observe(
            reward=0.2,
            info={
                "raw_action": {"discrete": 37},
                "projected_action": {
                    "discrete": {
                        "dispatch": "dispatch",
                        "route": "low_congestion",
                        "mode": "primary_fleet",
                        "reorder": "conservative",
                    }
                },
                "reward_components": {
                    "dispatch_attempted": 1,
                    "dispatch_success_count": 1,
                    "dispatch_progress_credit": 0.1,
                },
            },
        )
        accumulator.observe(
            reward=0.0,
            info={
                "raw_action": {"discrete": 8},
                "projected_action": {
                    "discrete": {
                        "dispatch": "hold",
                        "route": "low_congestion",
                        "mode": "secondary_fleet",
                        "reorder": "none",
                    }
                },
                "reward_components": {
                    "dispatch_attempted": 0,
                    "dispatch_success_count": 0,
                },
            },
        )

        episode = accumulator.finish()
        summary = summarize_episodes("route_disruption_congestion", [episode])

        self.assertEqual(episode.action_attempts_by_id["31"], 1)
        self.assertEqual(episode.action_dispatch_by_id["31"], 1)
        self.assertEqual(episode.action_successes_by_id["37"], 1)
        self.assertEqual(episode.action_failed_noop_by_id["31"], 1)
        self.assertEqual(episode.action_no_unassigned_by_id["31"], 1)
        self.assertEqual(episode.action_no_current_work_by_id["31"], 1)
        self.assertEqual(episode.action_already_assigned_by_id["31"], 1)
        self.assertEqual(episode.action_route_failure_by_id["31"], 1)
        self.assertAlmostEqual(episode.action_dispatch_success_ratio_by_id["31"], 0.0)
        self.assertAlmostEqual(episode.action_dispatch_success_ratio_by_id["37"], 1.0)
        self.assertEqual(episode.route_distribution_failed_dispatch["shortest"], 1)
        self.assertEqual(episode.route_distribution_successful_dispatch["low_congestion"], 1)
        self.assertEqual(episode.route_distribution_hold["low_congestion"], 1)
        self.assertAlmostEqual(episode.route_dispatch_success_ratio_by_route["shortest"], 0.0)
        self.assertAlmostEqual(episode.route_dispatch_success_ratio_by_route["low_congestion"], 1.0)
        self.assertEqual(summary["scenario_action_attempts_by_id"]["31"], 1)
        self.assertEqual(summary["scenario_action_successes_by_id"]["37"], 1)
        self.assertEqual(summary["scenario_action_failed_noop_by_id"]["31"], 1)
        self.assertEqual(summary["action31_attempts"], 1)
        self.assertEqual(summary["action31_successes"], 0)
        self.assertEqual(summary["action31_failed_noop"], 1)
        self.assertEqual(summary["action31_no_unassigned"], 1)
        self.assertEqual(summary["action31_no_current_work"], 1)
        self.assertEqual(summary["action31_route_failure"], 1)
        self.assertEqual(summary["scenario_route_distribution_failed_dispatch"]["shortest"], 1)
        self.assertEqual(summary["scenario_route_distribution_successful_dispatch"]["low_congestion"], 1)
        self.assertEqual(summary["scenario_route_distribution_hold"]["low_congestion"], 1)
        self.assertEqual(summary["top_failed_noop_actions"][0]["action_id"], 31)

    def test_healthy_in_transit_order_is_active_but_not_stuck_at_finite_horizon(self) -> None:
        accumulator = EpisodeMetricAccumulator(
            scenario_id="baseline_normal",
            seed=50,
            episode_index=8,
        )
        accumulator.observe(
            reward=0.0,
            info={
                "snapshot": {
                    "time": 86_400.0,
                    "orders": {
                        "order-stuck": {
                            "status": "in_transit",
                            "assigned_vehicle_id": "vehicle-7",
                            "pickup_time": 84_000.0,
                            "delivery_time": None,
                            "due_time": 90_000.0,
                            "is_late": False,
                        },
                        "order-done": {
                            "status": "delivered",
                            "assigned_vehicle_id": None,
                            "pickup_time": 1_000.0,
                            "delivery_time": 2_000.0,
                            "due_time": 3_000.0,
                            "is_late": False,
                        },
                    },
                    "vehicles": {
                        "vehicle-7": {
                            "tier": "secondary",
                            "asset_kind": "truck",
                            "load_units": 3.0,
                            "utilization": 0.6,
                            "current_node_id": "node-3",
                        },
                    },
                },
            },
        )

        episode = accumulator.finish()
        summary = summarize_episodes("baseline_normal", [episode])
        episode_json = episode.to_dict()

        expected_example = {
            "scenario_id": "baseline_normal",
            "seed": 50,
            "episode_index": 8,
            "order_id": "order-stuck",
            "status": "in_transit",
            "assigned_vehicle_id": "vehicle-7",
            "pickup_time": 84_000.0,
            "delivery_time": None,
            "due_time": 90_000.0,
            "is_late": False,
            "final_snapshot_time": 86_400.0,
            "seconds_to_due": 3_600.0,
            "vehicle_tier": "secondary",
            "vehicle_asset_kind": "truck",
            "vehicle_load_units": 3.0,
            "vehicle_utilization": 0.6,
            "vehicle_current_node_id": "node-3",
        }
        self.assertEqual(episode.active_assigned_in_transit_orders, 1)
        self.assertEqual(episode.stuck_assigned_in_transit_orders, 0)
        self.assertEqual(
            episode_json["active_assigned_in_transit_order_examples"],
            [expected_example],
        )
        self.assertEqual(episode_json["stuck_assigned_in_transit_order_examples"], [])
        self.assertEqual(summary["active_assigned_in_transit_orders"], 1)
        self.assertEqual(summary["stuck_assigned_in_transit_orders"], 0)
        self.assertEqual(
            summary["active_assigned_in_transit_order_examples"],
            [expected_example],
        )
        self.assertEqual(summary["stuck_assigned_in_transit_order_examples"], [])
        self.assertEqual(
            evaluate_thresholds(summary, {"fail_on_stuck_assigned_in_transit_orders": True}),
            [],
        )

    def test_overdue_active_order_counts_as_stuck_failure(self) -> None:
        accumulator = EpisodeMetricAccumulator(
            scenario_id="baseline_normal",
            seed=51,
            episode_index=9,
        )
        accumulator.observe(
            reward=0.0,
            info={
                "snapshot": {
                    "time": 86_400.0,
                    "orders": {
                        "order-overdue": {
                            "status": "assigned",
                            "assigned_vehicle_id": "vehicle-8",
                            "pickup_time": None,
                            "delivery_time": None,
                            "due_time": 85_000.0,
                            "is_late": False,
                        },
                    },
                    "vehicles": {
                        "vehicle-8": {
                            "tier": "primary",
                            "asset_kind": "truck",
                            "load_units": 1.0,
                            "utilization": 0.2,
                            "current_node_id": "node-8",
                        },
                    },
                },
            },
        )

        episode = accumulator.finish()
        summary = summarize_episodes("baseline_normal", [episode])

        self.assertEqual(episode.active_assigned_in_transit_orders, 1)
        self.assertEqual(episode.stuck_assigned_in_transit_orders, 1)
        self.assertEqual(summary["active_assigned_in_transit_orders"], 1)
        self.assertEqual(summary["stuck_assigned_in_transit_orders"], 1)
        self.assertEqual(
            evaluate_thresholds(summary, {"fail_on_stuck_assigned_in_transit_orders": True}),
            ["stuck assigned/in-transit orders present: 1"],
        )

    def test_stale_active_order_without_due_time_counts_as_stuck_failure(self) -> None:
        accumulator = EpisodeMetricAccumulator(
            scenario_id="baseline_normal",
            seed=52,
            episode_index=10,
        )
        accumulator.observe(
            reward=0.0,
            info={
                "snapshot": {
                    "time": 86_400.0,
                    "orders": {
                        "order-stale": {
                            "status": "in_transit",
                            "assigned_vehicle_id": "vehicle-9",
                            "pickup_time": 50_000.0,
                            "delivery_time": None,
                            "due_time": None,
                            "is_late": False,
                        },
                    },
                    "vehicles": {"vehicle-9": {"tier": "secondary"}},
                },
            },
        )

        episode = accumulator.finish()
        summary = summarize_episodes("baseline_normal", [episode])

        self.assertEqual(episode.active_assigned_in_transit_orders, 1)
        self.assertEqual(episode.stuck_assigned_in_transit_orders, 1)
        self.assertEqual(summary["active_assigned_in_transit_orders"], 1)
        self.assertEqual(summary["stuck_assigned_in_transit_orders"], 1)
        self.assertEqual(
            evaluate_thresholds(summary, {"fail_on_stuck_assigned_in_transit_orders": True}),
            ["stuck assigned/in-transit orders present: 1"],
        )

    def test_mixed_route_candidate_quality_telemetry_summarizes(self) -> None:
        accumulator = EpisodeMetricAccumulator(
            scenario_id="mixed_stress",
            seed=202,
            episode_index=0,
        )
        accumulator.observe(
            reward=0.1,
            info={
                "raw_action": {"discrete": 44},
                "projected_action": {
                    "discrete": {
                        "dispatch": "dispatch",
                        "route": "high_resilience",
                        "mode": "secondary_fleet",
                        "reorder": "conservative",
                    }
                },
                "reward_components": {
                    "dispatch_attempted": 1,
                    "dispatch_success_count": 1,
                    "shortest_route_candidate_score": 0.50,
                    "low_congestion_route_candidate_score": 0.20,
                    "high_resilience_route_candidate_score": 0.30,
                    "selected_route_candidate_score": 0.30,
                    "best_route_candidate_score": 0.50,
                    "selected_route_candidate_score_gap": 0.20,
                    "selected_route_candidate_near_best": 1.0,
                    "selected_route_candidate_action_gate": 1.0,
                    "route_candidate_alignment_credit": 0.004,
                    "route_candidate_mismatch_penalty": 0.0,
                    "route_candidate_useful_work_factor": 1.0,
                    "route_adaptation_pressure": 0.70,
                    "route_disruption_reliability_pressure": 0.55,
                    "route_congestion_cost_pressure": 0.60,
                },
            },
        )

        episode = accumulator.finish()
        summary = summarize_episodes("mixed_stress", [episode])

        self.assertAlmostEqual(episode.route_candidate_score_mean_by_route["shortest"], 0.50)
        self.assertAlmostEqual(episode.route_candidate_score_mean_by_route["high_resilience"], 0.30)
        self.assertAlmostEqual(episode.selected_route_candidate_score_mean_by_route["high_resilience"], 0.30)
        self.assertEqual(episode.selected_route_rank_counts["2"], 1)
        self.assertEqual(episode.selected_route_best_count, 0)
        self.assertEqual(episode.selected_route_near_best_count, 1)
        self.assertAlmostEqual(episode.selected_route_margin_to_best_mean, 0.20)
        self.assertEqual(episode.shortest_near_best_but_not_selected_count, 1)
        self.assertEqual(episode.high_resilience_selected_when_shortest_near_best_count, 1)
        self.assertEqual(episode.high_resilience_selected_when_not_best_count, 1)
        self.assertEqual(episode.mixed_route_overconservative_candidate_count, 1)
        self.assertAlmostEqual(summary["route_candidate_score_mean_by_route"]["shortest"], 0.50)
        self.assertAlmostEqual(summary["selected_route_candidate_score_mean_by_route"]["high_resilience"], 0.30)
        self.assertEqual(summary["selected_route_rank_counts"]["2"], 1)
        self.assertEqual(summary["shortest_near_best_but_not_selected_count"], 1)
        self.assertEqual(summary["high_resilience_selected_when_shortest_near_best_count"], 1)
        self.assertEqual(summary["high_resilience_selected_when_not_best_count"], 1)
        self.assertEqual(summary["mixed_route_overconservative_candidate_count"], 1)

    def test_premium_responsiveness_telemetry_summarizes(self) -> None:
        accumulator = EpisodeMetricAccumulator(
            scenario_id="premium_sla_pressure",
            seed=102,
            episode_index=0,
        )
        accumulator.observe(
            reward=-0.1,
            info={
                "raw_action": {"discrete": 0},
                "projected_action": {
                    "discrete": {
                        "dispatch": "hold",
                        "route": "shortest",
                        "mode": "primary_fleet",
                        "reorder": "none",
                    }
                },
                "reward_components": {
                    "premium_sla_pressure": 0.8,
                    "pending_work_pressure": 0.9,
                    "emergency_gate": 0.6,
                    "inventory_shortfall": 0.4,
                    "backlog_age_pressure": 0.2,
                    "lateness_risk": 0.1,
                    "premium_sla_fleet_credit_blocked_no_current_work": 1,
                    "premium_primary_adaptation_credit_blocked_no_current_work": 1,
                },
            },
        )
        accumulator.observe(
            reward=0.2,
            info={
                "raw_action": {"discrete": 39},
                "projected_action": {
                    "discrete": {
                        "dispatch": "dispatch",
                        "route": "low_congestion",
                        "mode": "primary_fleet",
                        "reorder": "emergency",
                    }
                },
                "reward_components": {
                    "dispatch_attempted": 1,
                    "dispatch_success_count": 1,
                    "premium_sla_pressure": 0.8,
                    "pending_work_pressure": 0.9,
                    "current_dispatch_work_count": 2,
                    "current_dispatch_work_signal": 1,
                    "emergency_gate": 0.6,
                    "emergency_useful_factor": 1.0,
                    "premium_fleet_credit_allowed": 1,
                },
            },
        )

        episode = accumulator.finish()
        summary = summarize_episodes("premium_sla_pressure", [episode])

        self.assertEqual(episode.premium_pressure_steps, 2)
        self.assertEqual(episode.premium_hold_steps, 1)
        self.assertEqual(episode.premium_dispatch_steps, 1)
        self.assertEqual(episode.premium_hold_under_useful_dispatch_opportunity, 1)
        self.assertEqual(episode.premium_dispatch_under_useful_dispatch_opportunity, 1)
        self.assertEqual(episode.premium_reorder_none_steps, 1)
        self.assertEqual(episode.premium_reorder_emergency_steps, 1)
        self.assertEqual(episode.premium_useful_reorder_opportunity_steps, 2)
        self.assertEqual(episode.premium_reorder_missed_opportunity_steps, 1)
        self.assertEqual(episode.hold_primary_under_premium_pressure, 1)
        self.assertEqual(episode.dispatch_primary_under_premium_pressure, 1)
        self.assertEqual(episode.premium_sla_fleet_credit_blocked_no_current_work, 1)
        self.assertEqual(episode.premium_primary_adaptation_credit_blocked_no_current_work, 1)
        self.assertEqual(episode.premium_fleet_credit_allowed, 1)
        self.assertEqual(summary["premium_pressure_steps"], 2)
        self.assertAlmostEqual(summary["premium_hold_rate"], 0.5)
        self.assertAlmostEqual(summary["premium_dispatch_rate"], 0.5)
        self.assertAlmostEqual(summary["premium_missed_useful_dispatch_rate"], 0.5)
        self.assertAlmostEqual(summary["premium_missed_useful_reorder_rate"], 0.5)
        self.assertAlmostEqual(summary["hold_primary_under_premium_pressure_rate"], 0.5)

    def test_outputs_include_routepremium_mixed_diagnostic_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "reports"
            scenario = ScenarioConfig(
                scenario_id="mixed_stress",
                name="Mixed Stress",
                description="Synthetic route premium mixed telemetry fixture.",
                seeds=[202],
                episode_count=1,
                environment_overrides={},
                expected_behavior="fixture",
                pass_fail_thresholds={},
            )
            accumulator = EpisodeMetricAccumulator(
                scenario_id="mixed_stress",
                seed=202,
                episode_index=0,
            )
            accumulator.observe(
                reward=0.0,
                info={
                    "raw_action": {"discrete": 31},
                    "projected_action": {
                        "discrete": {
                            "dispatch": "dispatch",
                            "route": "shortest",
                            "mode": "primary_fleet",
                            "reorder": "emergency",
                        }
                    },
                    "reward_components": {
                        "dispatch_attempted": 1,
                        "dispatch_success_count": 0,
                        "dispatch_no_unassigned_orders": 1,
                        "current_dispatch_work_count": 0,
                        "current_dispatch_work_signal": 0,
                        "premium_sla_pressure": 0.7,
                        "pending_work_pressure": 0.8,
                        "emergency_gate": 0.6,
                        "shortest_route_candidate_score": 0.4,
                        "low_congestion_route_candidate_score": 0.2,
                        "high_resilience_route_candidate_score": 0.5,
                        "selected_route_candidate_score": 0.4,
                        "best_route_candidate_score": 0.5,
                        "selected_route_candidate_score_gap": 0.1,
                    },
                },
            )
            episode = accumulator.finish()

            write_evaluation_outputs(
                output_dir=output_dir,
                checkpoint_path=Path("candidate.pt"),
                scenarios=[scenario],
                episode_rows=[episode],
                override_capabilities={},
            )

            episode_json = json.loads((output_dir / "episode_metrics.jsonl").read_text(encoding="utf-8"))
            summary_json = json.loads((output_dir / "scenario_summary.json").read_text(encoding="utf-8"))
            csv_text = (output_dir / "scenario_summary.csv").read_text(encoding="utf-8")
            report_text = (output_dir / "real_world_evaluation_report.md").read_text(encoding="utf-8")
            scenario_summary = summary_json["scenarios"][0]

            self.assertIn("action_attempts_by_id", episode_json)
            self.assertIn("route_distribution_failed_dispatch", episode_json)
            self.assertIn("route_candidate_score_mean_by_route", episode_json)
            self.assertIn("premium_hold_under_useful_dispatch_opportunity", episode_json)
            self.assertIn("action31_attempts", scenario_summary)
            self.assertIn("top_failed_noop_actions", scenario_summary)
            self.assertIn("route_successful_dispatch_json", scenario_summary)
            self.assertIn("route_failed_dispatch_json", scenario_summary)
            self.assertIn("action31_attempts", csv_text)
            self.assertIn("action31_successes", csv_text)
            self.assertIn("action31_failed_noop", csv_text)
            self.assertIn("action31_dispatch_success_ratio", csv_text)
            self.assertIn("top_failed_noop_actions_json", csv_text)
            self.assertIn("route_successful_dispatch_json", csv_text)
            self.assertIn("route_failed_dispatch_json", csv_text)
            self.assertIn("premium_hold_under_useful_dispatch_opportunity", csv_text)
            self.assertIn("premium_reorder_missed_opportunity_steps", csv_text)
            self.assertIn("shortest_near_best_but_not_selected_count", csv_text)
            self.assertIn("high_resilience_selected_when_shortest_near_best_count", csv_text)
            self.assertIn("Route/Premium/Mixed Diagnostic Telemetry", report_text)
            self.assertIn("action31", report_text)
            self.assertIn("route_candidate_score_mean_by_route", report_text)
            self.assertIn("selected_route_candidate_score_mean_by_route", report_text)
            self.assertIn("selected_route_rank_counts", report_text)
            self.assertIn("selected_route_margin_to_best_mean", report_text)
            self.assertIn("selected_route_best_count", report_text)
            self.assertIn("selected_route_near_best_count", report_text)
            self.assertIn("high_resilience_selected_when_not_best_count", report_text)
            self.assertIn("mixed_route_overconservative_candidate_count", report_text)
            self.assertIn("premium_pressure_steps", report_text)
            self.assertIn("premium_hold_rate", report_text)
            self.assertIn("premium_dispatch_rate", report_text)
            self.assertIn("premium_no_vehicle_steps", report_text)
            self.assertIn("premium_already_assigned_steps", report_text)
            self.assertIn("premium_no_unassigned_steps", report_text)
            self.assertIn("premium_reorder_none_steps", report_text)
            self.assertIn("premium_reorder_conservative_steps", report_text)
            self.assertIn("premium_reorder_aggressive_steps", report_text)
            self.assertIn("premium_reorder_emergency_steps", report_text)
            self.assertIn("premium_service_pressure_steps", report_text)
            self.assertIn("premium_late_or_at_risk_backlog_steps", report_text)
            self.assertIn("hold_primary_under_premium_pressure_rate", report_text)
            self.assertIn("mixed_stress remains a watch", report_text)

    def test_evaluation_outputs_mark_global_hard_blocker_failure_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "reports"
            scenario = ScenarioConfig(
                scenario_id="premium_sla_pressure",
                name="Premium SLA",
                description="Synthetic hard-blocker report fixture.",
                seeds=[1],
                episode_count=1,
                environment_overrides={},
                expected_behavior="fixture",
                pass_fail_thresholds={
                    "min_service_level": 0.93,
                    "min_dispatch_success_per_attempt": 0.50,
                },
            )
            episode = EpisodeMetricAccumulator(
                scenario_id="premium_sla_pressure",
                seed=1,
                episode_index=0,
            ).finish()
            episode.service_level = 0.96
            episode.dispatch_success_per_attempt = 0.95
            episode.mean_step_delivered_delta = 0.75
            episode.delivered_delta = 0.75
            episode.no_work_positive_dqn_local = 1

            write_evaluation_outputs(
                output_dir=output_dir,
                checkpoint_path=Path("candidate.pt"),
                scenarios=[scenario],
                episode_rows=[episode],
                override_capabilities={},
            )

            summary_json = json.loads((output_dir / "scenario_summary.json").read_text(encoding="utf-8"))
            csv_text = (output_dir / "scenario_summary.csv").read_text(encoding="utf-8")
            report_text = (output_dir / "real_world_evaluation_report.md").read_text(encoding="utf-8")
            scenario_summary = summary_json["scenarios"][0]

            self.assertEqual(scenario_summary["scenario_threshold_verdict"], "PASS")
            self.assertEqual(scenario_summary["hard_blocker_verdict"], "FAIL")
            self.assertEqual(scenario_summary["verdict"], "FAIL")
            self.assertIn("no-work positive dqn_local present: 1", scenario_summary["hard_blocker_failures"])
            self.assertIn("hard_blocker_verdict", csv_text)
            self.assertIn("scenario_threshold_verdict", csv_text)
            self.assertIn("Global hard blockers override scenario threshold pass", report_text)
            self.assertIn("delivered_delta is a backward-compatible alias for mean_step_delivered_delta", report_text)

    def test_scenario_summary_reports_top_action_family_concentration(self) -> None:
        accumulator = EpisodeMetricAccumulator(scenario_id="baseline", seed=1, episode_index=0)

        for action_id, route, mode, reorder in (
            (24, "shortest", "secondary_fleet", "none"),
            (24, "shortest", "secondary_fleet", "none"),
            (24, "shortest", "secondary_fleet", "none"),
            (25, "shortest", "secondary_fleet", "conservative"),
            (32, "low_congestion", "secondary_fleet", "none"),
        ):
            accumulator.observe(
                reward=0.1,
                info={
                    "raw_action": {"discrete": action_id},
                    "projected_action": {
                        "discrete": {
                            "dispatch": "dispatch",
                            "route": route,
                            "mode": mode,
                            "reorder": reorder,
                        }
                    },
                    "reward_components": {
                        "dispatch_attempted": 1,
                        "dispatch_success_count": 1,
                        "dispatch_progress_credit": 0.1,
                    },
                },
            )
        episode = accumulator.finish()
        summary = summarize_episodes("baseline", [episode])

        self.assertEqual(summary["action_24_count"], 3)
        self.assertEqual(summary["action_25_count"], 1)
        self.assertAlmostEqual(summary["action_24_percentage"], 0.6)
        self.assertAlmostEqual(summary["action_25_percentage"], 0.2)
        self.assertAlmostEqual(summary["action_24_25_concentration"], 0.8)
        self.assertEqual(summary["top_action_ids"][0]["action_id"], 24)
        self.assertEqual(summary["top_action_ids"][0]["count"], 3)
        self.assertEqual(summary["route_distribution"]["shortest"], 4)
        self.assertEqual(summary["fleet_distribution"]["secondary_fleet"], 5)
        self.assertEqual(summary["reorder_mode_distribution"]["conservative"], 1)
        self.assertEqual(summary["dispatch_success_by_action_id"]["24"], 3)

    def test_reward_component_diagnostics_are_reported_by_action_and_family(self) -> None:
        accumulator = EpisodeMetricAccumulator(
            scenario_id="route_disruption_congestion",
            seed=41,
            episode_index=0,
        )

        accumulator.observe(
            reward=-0.2,
            info={
                "raw_action": {"discrete": 41},
                "projected_action": {
                    "discrete": {
                        "dispatch": "dispatch",
                        "route": "high_resilience",
                        "mode": "secondary_fleet",
                        "reorder": "conservative",
                    }
                },
                "reward_components": {
                    "dispatch_attempted": 1,
                    "dispatch_success_count": 0,
                    "current_dispatch_work_count": 0,
                    "current_dispatch_work_signal": 0,
                    "dispatch_no_unassigned_orders": 1,
                    "dqn_local": -0.22,
                    "no_current_or_unassigned_dispatch_penalty": 0.25,
                    "route_candidate_alignment_credit": 0.01,
                    "route_candidate_mismatch_penalty": 0.04,
                    "selected_route_candidate_score_gap": 0.20,
                    "route_candidate_useful_work_factor": 0.0,
                    "high_resilience_no_useful_work_credit_blocked": 1.0,
                    "candidate_alignment_blocked_no_useful_work": 1.0,
                },
            },
        )
        accumulator.observe(
            reward=-0.1,
            info={
                "raw_action": {"discrete": 9},
                "projected_action": {
                    "discrete": {
                        "dispatch": "hold",
                        "route": "low_congestion",
                        "mode": "secondary_fleet",
                        "reorder": "conservative",
                    }
                },
                "reward_components": {
                    "dqn_local": -0.05,
                    "hold_under_lateness_pressure": 0.80,
                    "hold_under_lateness_pressure_penalty": 0.088,
                    "hold_timing_degradation_pressure": 0.60,
                },
            },
        )
        accumulator.observe(
            reward=-0.3,
            info={
                "raw_action": {"discrete": 32},
                "projected_action": {
                    "discrete": {
                        "dispatch": "dispatch",
                        "route": "low_congestion",
                        "mode": "secondary_fleet",
                        "reorder": "none",
                    }
                },
                "reward_components": {
                    "dispatch_attempted": 1,
                    "dispatch_success_count": 0,
                    "current_dispatch_work_count": 0,
                    "current_dispatch_work_signal": 0,
                    "dispatch_no_unassigned_orders": 1,
                    "dqn_local": -0.34,
                    "no_current_or_unassigned_dispatch_penalty": 0.28,
                    "low_congestion_no_useful_work_credit_blocked": 1.0,
                    "low_congestion_adaptation_credit": 0.0,
                    "route_candidate_useful_work_factor": 0.0,
                    "candidate_alignment_blocked_no_useful_work": 1.0,
                },
            },
        )

        episode = accumulator.finish()
        summary = summarize_episodes("route_disruption_congestion", [episode])
        episode_json = episode.to_dict()

        self.assertAlmostEqual(
            episode_json["reward_component_sum_by_action_id"]["41"][
                "no_current_or_unassigned_dispatch_penalty"
            ],
            0.25,
        )
        self.assertAlmostEqual(
            episode_json["reward_component_mean_by_action_id"]["9"][
                "hold_under_lateness_pressure_penalty"
            ],
            0.088,
        )
        self.assertAlmostEqual(
            summary["reward_component_mean_by_action_id"]["41"]["route_candidate_mismatch_penalty"],
            0.04,
        )
        self.assertAlmostEqual(
            summary["reward_component_sum_by_action_family"][
                "dispatch:high_resilience:secondary_fleet:conservative"
            ]["candidate_alignment_blocked_no_useful_work"],
            1.0,
        )
        self.assertAlmostEqual(
            summary["reward_component_mean_by_action_family"][
                "dispatch:high_resilience:secondary_fleet:conservative"
            ]["high_resilience_no_useful_work_credit_blocked"],
            1.0,
        )
        self.assertAlmostEqual(
            summary["reward_component_mean_by_action_family"][
                "hold:low_congestion:secondary_fleet:conservative"
            ]["hold_under_lateness_pressure"],
            0.80,
        )
        self.assertAlmostEqual(
            summary["reward_component_mean_by_action_id"]["9"]["hold_timing_degradation_pressure"],
            0.60,
        )
        self.assertAlmostEqual(
            summary["reward_component_mean_by_action_id"]["32"][
                "low_congestion_no_useful_work_credit_blocked"
            ],
            1.0,
        )
        self.assertAlmostEqual(
            summary["reward_component_sum_by_action_family"][
                "dispatch:low_congestion:secondary_fleet:none"
            ]["low_congestion_no_useful_work_credit_blocked"],
            1.0,
        )

    def test_action_concentration_alone_is_warning_not_exploit(self) -> None:
        summary = {
            "action_24_25_concentration": 0.75,
            "service_level": 0.96,
            "true_lateness_pressure": 0.0,
            "delivered_delta": 0.60,
            "dispatch_success_per_attempt": 0.95,
            "fake_dispatch_credit": 0,
            "route_failure_positive_dispatch_credit": 0,
        }
        thresholds = {
            "max_action_24_25_concentration_warn": 0.40,
            "max_action_24_25_concentration_fail": 0.40,
            "require_operational_degradation_for_action_concentration_fail": True,
            "fail_on_fake_dispatch_credit": True,
            "fail_on_route_failure_positive_dispatch_credit": True,
        }

        self.assertEqual(evaluate_thresholds(summary, thresholds), [])
        self.assertIn(
            "action_24_25_concentration 0.750 > warning 0.400",
            evaluate_warnings(summary, thresholds),
        )

    def test_action_concentration_with_operational_degradation_can_fail(self) -> None:
        summary = {
            "action_24_25_concentration": 0.75,
            "service_level": 0.40,
            "true_lateness_pressure": 0.35,
            "delivered_delta": 0.20,
            "dispatch_success_per_attempt": 0.45,
        }
        failures = evaluate_thresholds(
            summary,
            {
                "max_action_24_25_concentration_fail": 0.40,
                "require_operational_degradation_for_action_concentration_fail": True,
                "min_service_level": 0.93,
                "max_true_lateness_pressure": 0.10,
                "min_delivered_delta": 0.45,
                "min_dispatch_success_per_attempt": 0.70,
            },
        )

        self.assertTrue(any("action_24_25_concentration" in failure for failure in failures))

    def test_mixed_stress_service_collapse_is_not_clean_pass(self) -> None:
        summary = {
            "service_level": 0.399,
            "true_lateness_pressure": 0.374,
            "delivered_delta": 1.050,
            "dispatch_success_per_attempt": 1.0,
            "action_24_25_concentration": 0.247,
        }
        failures = evaluate_thresholds(
            summary,
            {
                "min_service_level": 0.70,
                "max_true_lateness_pressure": 0.20,
                "min_dispatch_success_per_attempt": 0.50,
                "max_action_24_25_concentration_warn": 0.50,
            },
        )

        self.assertIn("service_level 0.399 < 0.700", failures)
        self.assertIn("true_lateness_pressure 0.374 > 0.200", failures)

    def test_demand_spike_lateness_collapse_is_flagged(self) -> None:
        summary = {
            "service_level": 0.414,
            "true_lateness_pressure": 0.390,
            "delivered_delta": 1.450,
            "dispatch_success_per_attempt": 1.0,
            "action_24_25_concentration": 0.247,
        }
        failures = evaluate_thresholds(
            summary,
            {
                "min_service_level": 0.60,
                "max_true_lateness_pressure": 0.25,
                "min_dispatch_success_per_attempt": 0.50,
                "max_action_24_25_concentration_warn": 0.50,
            },
        )

        self.assertIn("service_level 0.414 < 0.600", failures)
        self.assertIn("true_lateness_pressure 0.390 > 0.250", failures)

    def test_high_dispatch_success_passes_only_when_operational_quality_is_healthy(self) -> None:
        healthy = {
            "service_level": 0.955,
            "true_lateness_pressure": 0.010,
            "delivered_delta": 0.55,
            "dispatch_success_per_attempt": 0.95,
            "action_24_25_concentration": 0.35,
        }
        collapsed = {
            **healthy,
            "service_level": 0.399,
            "true_lateness_pressure": 0.374,
            "dispatch_success_per_attempt": 1.0,
        }
        thresholds = {
            "min_service_level": 0.70,
            "max_true_lateness_pressure": 0.20,
            "min_delivered_delta": 0.45,
            "min_dispatch_success_per_attempt": 0.50,
        }

        self.assertEqual(evaluate_thresholds(healthy, thresholds), [])
        self.assertTrue(evaluate_thresholds(collapsed, thresholds))

    def test_action_concentration_threshold_requires_operational_degradation_when_configured(self) -> None:
        thresholds = {
            "max_action_24_25_concentration_fail": 0.40,
            "require_operational_degradation_for_action_concentration_fail": True,
            "min_service_level": 0.93,
            "max_true_lateness_pressure": 0.05,
            "min_delivered_delta": 0.45,
            "min_dispatch_success_per_attempt": 0.70,
        }
        concentrated_but_healthy = {
            "action_24_25_concentration": 0.75,
            "service_level": 0.95,
            "true_lateness_pressure": 0.0,
            "delivered_delta": 0.60,
            "dispatch_success_per_attempt": 0.90,
        }
        concentrated_and_bad = {
            **concentrated_but_healthy,
            "service_level": 0.80,
        }

        self.assertFalse(
            any("action_24_25_concentration" in item for item in evaluate_thresholds(concentrated_but_healthy, thresholds))
        )
        self.assertTrue(
            any("action_24_25_concentration" in item for item in evaluate_thresholds(concentrated_and_bad, thresholds))
        )

    def test_fake_credit_remains_hard_fail_after_action_diagnostics(self) -> None:
        summary = {
            "action_24_25_concentration": 0.0,
            "fake_dispatch_credit": 1,
            "route_failure_positive_dispatch_credit": 1,
        }
        failures = evaluate_thresholds(
            summary,
            {
                "fail_on_fake_dispatch_credit": True,
                "fail_on_route_failure_positive_dispatch_credit": True,
            },
        )

        self.assertIn("fake dispatch credit present: 1", failures)
        self.assertIn("route failure with positive dispatch credit present: 1", failures)

    def test_mixed_route_failure_success_and_credit_is_warning_not_hard_exploit(self) -> None:
        accumulator = EpisodeMetricAccumulator(scenario_id="baseline", seed=1, episode_index=0)

        accumulator.observe(
            reward=0.1,
            info={
                "reward_components": {
                    "dispatch_route_failure": 1,
                    "dispatch_success_count": 2,
                    "dispatch_progress_credit": 0.2,
                }
            },
        )
        episode = accumulator.finish()
        summary = summarize_episodes("baseline", [episode])

        self.assertEqual(episode.route_failures, 1)
        self.assertEqual(episode.fake_dispatch_credit, 0)
        self.assertEqual(episode.route_failure_positive_dispatch_credit, 0)
        self.assertEqual(episode.mixed_success_route_failure_steps, 1)
        self.assertEqual(summary["route_failure_positive_dispatch_credit"], 0)
        self.assertEqual(summary["mixed_success_route_failure_steps"], 1)
        self.assertEqual(
            evaluate_thresholds(summary, {"fail_on_route_failure_positive_dispatch_credit": True}),
            [],
        )
        self.assertIn(
            "mixed_success_route_failure_steps present: 1",
            evaluate_warnings(summary, {}),
        )

    def test_route_failure_zero_success_and_positive_credit_remains_hard_exploit(self) -> None:
        accumulator = EpisodeMetricAccumulator(scenario_id="baseline", seed=1, episode_index=0)

        accumulator.observe(
            reward=0.1,
            info={
                "reward_components": {
                    "route_feasibility_failed_count": 1,
                    "dispatch_success_count": 0,
                    "dispatch_progress_credit": 0.2,
                }
            },
        )
        episode = accumulator.finish()
        summary = summarize_episodes("baseline", [episode])
        failures = evaluate_thresholds(
            summary,
            {
                "fail_on_fake_dispatch_credit": True,
                "fail_on_route_failure_positive_dispatch_credit": True,
            },
        )

        self.assertEqual(episode.route_failures, 1)
        self.assertEqual(episode.fake_dispatch_credit, 1)
        self.assertEqual(episode.route_failure_positive_dispatch_credit, 1)
        self.assertEqual(episode.mixed_success_route_failure_steps, 0)
        self.assertIn("fake dispatch credit present: 1", failures)
        self.assertIn("route failure with positive dispatch credit present: 1", failures)

    def test_route_failure_without_positive_credit_is_counted_but_not_exploit(self) -> None:
        accumulator = EpisodeMetricAccumulator(scenario_id="baseline", seed=1, episode_index=0)

        accumulator.observe(
            reward=0.1,
            info={
                "reward_components": {
                    "dispatch_route_failure": 1,
                    "dispatch_success_count": 0,
                    "dispatch_progress_credit": 0.0,
                }
            },
        )
        episode = accumulator.finish()
        summary = summarize_episodes("baseline", [episode])

        self.assertEqual(episode.route_failures, 1)
        self.assertEqual(episode.fake_dispatch_credit, 0)
        self.assertEqual(episode.route_failure_positive_dispatch_credit, 0)
        self.assertEqual(episode.mixed_success_route_failure_steps, 0)
        self.assertEqual(
            evaluate_thresholds(
                summary,
                {
                    "fail_on_fake_dispatch_credit": True,
                    "fail_on_route_failure_positive_dispatch_credit": True,
                },
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
