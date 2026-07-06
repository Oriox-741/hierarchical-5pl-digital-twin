from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

import src.learn.train_joint_torch as train_joint_torch
from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv
from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.observation_builder import OBSERVATION_DIM, OBSERVATION_FEATURE_INDEX
from src.learn.curriculum import CurriculumConfig, CurriculumSample, CurriculumSampler
from src.learn.joint_buffers import DQNReplayBuffer, JointRewardBlendConfig, JointTransition
from src.learn.joint_metrics import JointMetricsAggregator
from src.learn.train_joint_torch import (
    TraceLoggingConfig,
    TrainingTimingWindow,
    TrainingState,
    configure_torch_threading,
    _save_final_artifacts_if_training_advanced,
    _should_write_final_training_outputs,
    _should_save_periodic_checkpoint,
    append_training_metrics_jsonl,
    attach_curriculum_telemetry,
    build_environment_config,
    configure_environment_info_mode_for_trace_logging,
    reset_environment_for_episode,
    parse_args,
    update_dqn,
    validate_resume_metrics_checkpoint_alignment,
)
from src.think.joint_policies import FLAT_DQN_ARCHITECTURE, HIERARCHICAL_DQN_ARCHITECTURE, JointPolicyBundle


BASE_TRAINING_CONFIG = {
    "mdp_contract_version": "physical_reality_v5_route_candidate_visibility",
    "shared_global_parameters": {
        "observation_dim": OBSERVATION_DIM,
        "max_steps": 4,
        "decision_interval_seconds": 300,
        "max_extra_dispatch_budget": 4,
        "feasibility_gated_macro_budget": True,
    },
    "reward_physics": {
        "planned_replenishment_cost_weight": 0.12,
        "excess_inventory_penalty": 0.15,
        "infeasible_dispatch_penalty_weight": 0.04,
        "flow_enablement_weight": 0.08,
        "dispatch_progress_weight": 0.05,
    },
}


class TrainJointCurriculumPlumbingTests(unittest.TestCase):
    def _allow_fine_tune_init_paths(self, *paths: Path):
        normalized_paths = frozenset(str(path).replace("\\", "/").lower() for path in paths)
        return patch.object(train_joint_torch, "APPROVED_FINE_TUNE_INIT_CHECKPOINTS", normalized_paths)

    def test_trace_disabled_training_uses_compact_info_and_reset_preserves_mode(self) -> None:
        base_env_config = build_environment_config(BASE_TRAINING_CONFIG, max_steps=4, seed=7)

        configure_environment_info_mode_for_trace_logging(
            base_env_config,
            TraceLoggingConfig(enabled=False),
        )

        self.assertEqual(base_env_config.info_mode, "compact")
        env = FivePLDigitalTwinEnv(deepcopy(base_env_config))
        try:
            reset_environment_for_episode(
                env=env,
                base_env_config=base_env_config,
                curriculum_sampler=CurriculumSampler(CurriculumConfig.disabled(seed=7)),
                seed=7,
                episode_index=0,
            )

            self.assertEqual(env.config.info_mode, "compact")
            _observation, _reward, _terminated, _truncated, info = env.step(_neutral_joint_action())
            self.assertNotIn("snapshot", info)
            self.assertIn("reward_components", info)
        finally:
            env.close()

    def test_trace_enabled_training_keeps_full_info_mode(self) -> None:
        base_env_config = build_environment_config(BASE_TRAINING_CONFIG, max_steps=4, seed=7)

        configure_environment_info_mode_for_trace_logging(
            base_env_config,
            TraceLoggingConfig(enabled=True),
        )

        self.assertEqual(base_env_config.info_mode, "full")

    def test_disabled_curriculum_reset_is_noop_and_clears_stale_stress_state(self) -> None:
        base_env_config = build_environment_config(BASE_TRAINING_CONFIG, max_steps=4, seed=7)
        env = FivePLDigitalTwinEnv(deepcopy(base_env_config))
        env.scenario_stress_state["holding_cost_multiplier"] = 2.5
        try:
            observation, sample = reset_environment_for_episode(
                env=env,
                base_env_config=base_env_config,
                curriculum_sampler=CurriculumSampler(CurriculumConfig.disabled(seed=7)),
                seed=7,
                episode_index=0,
            )

            self.assertEqual(observation.shape, (OBSERVATION_DIM,))
            self.assertFalse(sample.enabled)
            self.assertEqual(sample.environment_overrides, {})
            self.assertEqual(env.scenario_stress_state["holding_cost_multiplier"], 1.0)
            self.assertEqual(
                float(observation[OBSERVATION_FEATURE_INDEX["normalized_holding_cost_pressure"]]),
                0.0,
            )
            self.assertEqual(env.config.excess_inventory_penalty_weight, base_env_config.excess_inventory_penalty_weight)
        finally:
            env.close()

    def test_enabled_curriculum_applies_stage_overrides_after_reset(self) -> None:
        base_env_config = build_environment_config(BASE_TRAINING_CONFIG, max_steps=4, seed=7)
        env = FivePLDigitalTwinEnv(deepcopy(base_env_config))
        sampler = CurriculumSampler(CurriculumConfig(enabled=True, stage="high_holding_cost", seed=7))
        try:
            observation, sample = reset_environment_for_episode(
                env=env,
                base_env_config=base_env_config,
                curriculum_sampler=sampler,
                seed=7,
                episode_index=1,
            )

            self.assertEqual(observation.shape, (OBSERVATION_DIM,))
            self.assertTrue(sample.enabled)
            self.assertEqual(sample.stage, "high_holding_cost")
            self.assertGreater(sample.environment_overrides["holding_cost_multiplier"], 1.0)
            self.assertGreater(env.config.excess_inventory_penalty_weight, base_env_config.excess_inventory_penalty_weight)
            self.assertEqual(env.scenario_stress_state["holding_cost_multiplier"], sample.environment_overrides["holding_cost_multiplier"])
            self.assertTrue(sample.capability_matrix)
        finally:
            env.close()

    def test_curriculum_reset_does_not_mutate_base_environment_config(self) -> None:
        base_env_config = build_environment_config(BASE_TRAINING_CONFIG, max_steps=4, seed=7)
        env = FivePLDigitalTwinEnv(deepcopy(base_env_config))
        sampler = CurriculumSampler(CurriculumConfig(enabled=True, stage="vehicle_scarcity", seed=7))
        try:
            reset_environment_for_episode(
                env=env,
                base_env_config=base_env_config,
                curriculum_sampler=sampler,
                seed=7,
                episode_index=0,
            )

            self.assertEqual(base_env_config.excess_inventory_penalty_weight, 0.15)
            self.assertEqual(base_env_config.random_seed, 7)
        finally:
            env.close()

    def test_curriculum_reset_does_not_deepcopy_db_snapshots(self) -> None:
        base_env_config = build_environment_config(BASE_TRAINING_CONFIG, max_steps=4, seed=7)
        snapshot = _DeepcopyTrapDict({"speed_mps": 15.0, "battery_pct": 0.9})
        base_env_config.db_snapshots = (snapshot,)
        env = FivePLDigitalTwinEnv(EnvironmentConfig(max_steps=4, rolling_demand_enabled=False))
        try:
            reset_environment_for_episode(
                env=env,
                base_env_config=base_env_config,
                curriculum_sampler=CurriculumSampler(CurriculumConfig.disabled(seed=7)),
                seed=11,
                episode_index=0,
            )

            self.assertIsNot(env.config, base_env_config)
            self.assertIs(env.config.db_snapshots[0], snapshot)
            self.assertEqual(base_env_config.random_seed, 7)
            self.assertEqual(env.config.random_seed, 11)
        finally:
            env.close()

    def test_reset_environment_uses_global_step_for_scheduled_curriculum(self) -> None:
        base_env_config = build_environment_config(BASE_TRAINING_CONFIG, max_steps=4, seed=7)
        env = FivePLDigitalTwinEnv(deepcopy(base_env_config))
        config = CurriculumConfig.from_mapping(
            {
                "enabled": True,
                "stage": "normal_v4",
                "seed": 7,
                "stage_schedule": [
                    {"name": "normal_v4", "steps": 1},
                    {"name": "route_disruption", "steps": 1},
                ],
                "randomization": {"enabled": False, "ranges_profile": "fixed_medium"},
            }
        )
        sampler = CurriculumSampler(config, total_timesteps=100)
        try:
            observation, sample = reset_environment_for_episode(
                env=env,
                base_env_config=base_env_config,
                curriculum_sampler=sampler,
                seed=9,
                episode_index=2,
                global_step=75,
            )

            self.assertEqual(observation.shape, (OBSERVATION_DIM,))
            self.assertEqual(sample.stage, "route_disruption")
            self.assertEqual(sample.effective_stage, "route_disruption")
            self.assertGreater(sample.environment_overrides["route_disruption_probability"], 0.0)
            self.assertGreater(env.scenario_stress_state["route_disruption_probability"], 0.0)
        finally:
            env.close()

    def test_attach_curriculum_telemetry_adds_compact_trace_fields(self) -> None:
        sample = CurriculumSample(
            enabled=True,
            stage="route_disruption",
            effective_stage="route_disruption",
            seed=42,
            episode_index=3,
            baseline_anchor_used=False,
            environment_overrides={"route_disruption_probability": 0.25, "congestion_multiplier": 1.75},
        )
        info: dict[str, object] = {}

        attach_curriculum_telemetry(info, sample)

        self.assertEqual(info["curriculum_enabled"], True)
        self.assertEqual(info["curriculum_stage"], "route_disruption")
        self.assertEqual(info["curriculum_episode_index"], 3)
        self.assertEqual(
            info["sampled_stress_overrides"],
            {"congestion_multiplier": 1.75, "route_disruption_probability": 0.25},
        )
        self.assertIn("curriculum", info)

    def test_attach_curriculum_telemetry_is_noop_when_disabled(self) -> None:
        sample = CurriculumSampler(CurriculumConfig.disabled(seed=1)).sample(0)
        info: dict[str, object] = {}

        attach_curriculum_telemetry(info, sample)

        self.assertEqual(info, {})

    def test_training_metrics_jsonl_includes_curriculum_fields_when_enabled(self) -> None:
        sample = CurriculumSample(
            enabled=True,
            stage="demand_spike",
            effective_stage="demand_spike",
            seed=42,
            episode_index=5,
            baseline_anchor_used=False,
            environment_overrides={"rolling_demand_probability_multiplier": 1.5},
        )
        metrics = JointMetricsAggregator()
        with TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "training_metrics.jsonl"

            append_training_metrics_jsonl(metrics, metrics_path, global_step=10, curriculum_sample=sample)

            row = json.loads(metrics_path.read_text(encoding="utf-8").strip())

        self.assertEqual(row["global_step"], 10)
        self.assertEqual(row["curriculum_stage"], "demand_spike")
        self.assertEqual(row["curriculum_episode_index"], 5)
        self.assertEqual(row["sampled_stress_overrides"], {"rolling_demand_probability_multiplier": 1.5})

    def test_joint_metrics_emit_action_quality_telemetry_by_action_and_family(self) -> None:
        metrics = JointMetricsAggregator()
        metrics.record_transition(
            _dqn_transition(
                25,
                reward_dqn_local=-0.22,
                info={
                    "curriculum_stage": "route_disruption",
                    "dispatch_success_count": 0.0,
                    "dispatch_no_unassigned_orders": 1.0,
                    "reward_components": {
                        "current_dispatch_work_count": 0.0,
                        "current_dispatch_work_signal": 0.0,
                        "dqn_local": -0.22,
                    },
                },
            )
        )
        metrics.record_transition(
            _dqn_transition(
                39,
                reward_dqn_local=0.18,
                info={
                    "curriculum_stage": "premium_sla",
                    "dispatch_success_count": 1.0,
                    "reward_components": {
                        "current_dispatch_work_count": 1.0,
                        "current_dispatch_work_signal": 1.0,
                        "dqn_local": 0.18,
                    },
                },
            )
        )

        row = metrics.snapshot(global_step=2).as_json_dict()

        self.assertEqual(row["action_failed_noop_by_id"]["25"], 1)
        self.assertEqual(row["action_no_current_work_by_id"]["25"], 1)
        self.assertEqual(row["action_no_unassigned_by_id"]["25"], 1)
        self.assertEqual(row["action_dispatch_attempts_by_id"]["25"], 1)
        self.assertEqual(row["action_dispatch_attempts_by_id"]["39"], 1)
        self.assertEqual(row["action_dispatch_successes_by_id"].get("25", 0), 0)
        self.assertEqual(row["action_dispatch_successes_by_id"]["39"], 1)
        self.assertEqual(row["action_dispatch_success_ratio_by_id"]["25"], 0.0)
        self.assertEqual(row["action_dispatch_success_ratio_by_id"]["39"], 1.0)
        self.assertEqual(
            row["dispatch_success_per_attempt_by_stage_action"]["route_disruption:25"],
            0.0,
        )
        self.assertEqual(
            row["dispatch_success_per_attempt_by_stage_action"]["premium_sla:39"],
            1.0,
        )
        self.assertGreater(row["top_action_concentration"], 0.0)
        self.assertIn(
            "dispatch:shortest:secondary_fleet:conservative",
            row["mean_dqn_local_reward_by_action_family"],
        )
        self.assertAlmostEqual(
            row["mean_dqn_local_reward_by_action_family"]["dispatch:shortest:secondary_fleet:conservative"],
            -0.22,
        )

    def test_joint_metrics_read_no_unassigned_exposure_from_reward_components(self) -> None:
        metrics = JointMetricsAggregator()
        metrics.record_transition(
            _dqn_transition(
                41,
                reward_dqn_local=-0.19,
                info={
                    "curriculum_stage": "route_disruption",
                    "reward_components": {
                        "dispatch_success_count": 0.0,
                        "dispatch_no_unassigned_orders": 1.0,
                        "current_dispatch_work_count": 0.0,
                        "current_dispatch_work_signal": 0.0,
                        "dqn_local": -0.19,
                    },
                },
            )
        )

        row = metrics.snapshot(global_step=1).as_json_dict()

        self.assertEqual(row["action_dispatch_attempts_by_id"]["41"], 1)
        self.assertEqual(row["action_failed_noop_by_id"]["41"], 1)
        self.assertEqual(row["action_no_current_work_by_id"]["41"], 1)
        self.assertEqual(row["action_no_unassigned_by_id"]["41"], 1)

    def test_joint_metrics_emit_v21_lateness_timing_telemetry(self) -> None:
        metrics = JointMetricsAggregator()
        metrics.record_transition(
            _dqn_transition(
                0,
                reward_dqn_local=-0.12,
                info={
                    "curriculum_stage": "route_disruption",
                    "reward_components": {
                        "true_lateness_pressure": 0.60,
                        "hold_under_lateness_pressure": 0.75,
                        "useful_dispatch_opportunity_missed": 1.0,
                    },
                },
            )
        )
        metrics.record_transition(
            _dqn_transition(
                28,
                reward_dqn_local=0.24,
                info={
                    "curriculum_stage": "premium_sla",
                    "dispatch_success_count": 1.0,
                    "reward_components": {
                        "current_dispatch_work_count": 1.0,
                        "current_dispatch_work_signal": 1.0,
                        "true_lateness_pressure": 0.40,
                        "useful_dispatch_lateness_urgency_credit": 0.08,
                        "primary_fleet_timing_justification_credit": 0.05,
                        "secondary_fleet_lateness_exposure": 0.0,
                        "dqn_local": 0.24,
                    },
                },
            )
        )
        metrics.record_transition(
            _dqn_transition(
                24,
                reward_dqn_local=0.05,
                info={
                    "curriculum_stage": "lead_time_delay",
                    "dispatch_success_count": 1.0,
                    "reward_components": {
                        "current_dispatch_work_count": 1.0,
                        "current_dispatch_work_signal": 1.0,
                        "true_lateness_pressure": 0.25,
                        "useful_dispatch_lateness_urgency_credit": 0.04,
                        "secondary_fleet_lateness_exposure": 0.50,
                        "dqn_local": 0.05,
                    },
                },
            )
        )

        row = metrics.snapshot(global_step=3).as_json_dict()

        for field in (
            "lateness_pressure_by_stage_action",
            "useful_dispatch_opportunity_missed_by_action",
            "primary_fleet_useful_dispatch_by_scenario",
            "hold_under_lateness_pressure_by_scenario",
            "secondary_fleet_lateness_exposure",
            "mean_useful_dispatch_lateness_urgency_credit_by_action_family",
            "mean_primary_fleet_timing_justification_credit_by_action_family",
        ):
            self.assertIn(field, row)

        self.assertEqual(row["useful_dispatch_opportunity_missed_by_action"]["0"], 1)
        self.assertEqual(row["hold_under_lateness_pressure_by_scenario"]["route_disruption"], 1)
        self.assertEqual(row["primary_fleet_useful_dispatch_by_scenario"]["premium_sla"], 1)
        self.assertAlmostEqual(row["secondary_fleet_lateness_exposure"], 0.50)
        self.assertAlmostEqual(row["lateness_pressure_by_stage_action"]["route_disruption:0"], 0.60)
        self.assertAlmostEqual(row["lateness_pressure_by_stage_action"]["premium_sla:28"], 0.40)
        self.assertAlmostEqual(
            row["mean_useful_dispatch_lateness_urgency_credit_by_action_family"][
                "dispatch:shortest:primary_fleet:none"
            ],
            0.08,
        )
        self.assertAlmostEqual(
            row["mean_primary_fleet_timing_justification_credit_by_action_family"][
                "dispatch:shortest:primary_fleet:none"
            ],
            0.05,
        )

    def test_training_metrics_jsonl_includes_native_timing_fields_when_supplied(self) -> None:
        timing = TrainingTimingWindow(
            started_at=10.0,
            window_started_at=12.0,
            process_started_at=1.0,
            process_window_started_at=1.5,
            start_step=2,
            window_start_step=6,
            cpu_logical_cores=8,
            torch_num_threads=4,
            torch_num_interop_threads=2,
        )
        timing.seconds_policy_select_window = 0.10
        timing.seconds_env_step_window = 0.20
        timing.seconds_buffer_add_window = 0.03
        timing.seconds_ppo_update_window = 0.40
        timing.seconds_dqn_update_window = 0.50
        timing.seconds_metrics_window = 0.06
        timing.seconds_checkpoint_window = 0.07
        timing.dqn_update_count_window = 1
        timing.dqn_gradient_steps_window = 2
        timing.dqn_replay_sample_seconds_window = 0.08
        timing.dqn_tensor_build_seconds_window = 0.09
        timing.dqn_forward_seconds_window = 0.11
        timing.dqn_loss_seconds_window = 0.12
        timing.dqn_backward_seconds_window = 0.13
        timing.dqn_optimizer_seconds_window = 0.14
        timing.dqn_target_update_seconds_window = 0.15
        timing.dqn_validation_seconds_window = 0.16
        timing.dqn_misc_seconds_window = 0.17
        timing.dqn_sample_count_window = 128
        timing.dqn_replay_size = 128
        timing.dqn_batch_size = 64
        timing.dqn_gradient_steps_per_rollout_effective = 2
        timing.dqn_update_trigger_reason = "unit_test"
        with TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "training_metrics.jsonl"

            append_training_metrics_jsonl(
                JointMetricsAggregator(),
                metrics_path,
                global_step=10,
                timing_fields=timing.as_metrics_fields(global_step=10, now=16.0, process_now=4.0),
            )
            row = json.loads(metrics_path.read_text(encoding="utf-8").strip())

        expected_fields = {
            "elapsed_seconds_total",
            "fps_total",
            "fps_window",
            "process_cpu_seconds_total",
            "process_cpu_seconds_window",
            "cpu_parallelism_total",
            "cpu_parallelism_window",
            "cpu_utilization_pct_total",
            "cpu_utilization_pct_window",
            "cpu_logical_cores",
            "torch_num_threads",
            "torch_num_interop_threads",
            "seconds_policy_select_window",
            "seconds_env_step_window",
            "seconds_buffer_add_window",
            "seconds_ppo_update_window",
            "seconds_dqn_update_window",
            "seconds_metrics_window",
            "seconds_checkpoint_window",
            "steps_window",
            "dqn_update_count_window",
            "dqn_gradient_steps_window",
            "dqn_replay_sample_seconds_window",
            "dqn_tensor_build_seconds_window",
            "dqn_forward_seconds_window",
            "dqn_loss_seconds_window",
            "dqn_backward_seconds_window",
            "dqn_optimizer_seconds_window",
            "dqn_target_update_seconds_window",
            "dqn_validation_seconds_window",
            "dqn_misc_seconds_window",
            "dqn_sample_count_window",
            "dqn_forward_avg_ms_per_grad_step_window",
            "dqn_backward_avg_ms_per_grad_step_window",
            "dqn_optimizer_avg_ms_per_grad_step_window",
            "dqn_validation_avg_ms_per_grad_step_window",
            "dqn_total_avg_ms_per_grad_step_window",
            "dqn_samples_per_second_window",
            "dqn_gradient_steps_per_second_window",
            "dqn_replay_size",
            "dqn_batch_size",
            "dqn_gradient_steps_per_rollout_effective",
            "dqn_update_trigger_reason",
        }
        self.assertTrue(expected_fields.issubset(row))
        for field in expected_fields - {"dqn_update_trigger_reason"}:
            self.assertIsInstance(row[field], (int, float))
            self.assertGreaterEqual(row[field], 0.0)
        self.assertEqual(row["dqn_update_trigger_reason"], "unit_test")
        self.assertEqual(row["steps_window"], 4)
        self.assertAlmostEqual(row["elapsed_seconds_total"], 6.0)
        self.assertAlmostEqual(row["fps_total"], 8.0 / 6.0)
        self.assertAlmostEqual(row["fps_window"], 1.0)
        self.assertAlmostEqual(row["process_cpu_seconds_total"], 3.0)
        self.assertAlmostEqual(row["process_cpu_seconds_window"], 2.5)
        self.assertAlmostEqual(row["cpu_parallelism_total"], 0.5)
        self.assertAlmostEqual(row["cpu_parallelism_window"], 0.625)
        self.assertAlmostEqual(row["seconds_policy_select_window"], 0.10)
        self.assertAlmostEqual(row["seconds_env_step_window"], 0.20)
        self.assertEqual(row["cpu_logical_cores"], 8)
        self.assertEqual(row["torch_num_threads"], 4)
        self.assertEqual(row["torch_num_interop_threads"], 2)
        self.assertEqual(row["dqn_update_count_window"], 1)
        self.assertEqual(row["dqn_gradient_steps_window"], 2)
        self.assertEqual(row["dqn_sample_count_window"], 128)
        self.assertAlmostEqual(row["dqn_forward_avg_ms_per_grad_step_window"], 55.0)
        self.assertAlmostEqual(row["dqn_backward_avg_ms_per_grad_step_window"], 65.0)
        self.assertAlmostEqual(row["dqn_optimizer_avg_ms_per_grad_step_window"], 70.0)
        self.assertAlmostEqual(row["dqn_validation_avg_ms_per_grad_step_window"], 80.0)
        self.assertAlmostEqual(row["dqn_total_avg_ms_per_grad_step_window"], 250.0)
        self.assertEqual(row["dqn_replay_size"], 128)
        self.assertEqual(row["dqn_batch_size"], 64)

    def test_torch_threading_cli_defaults_do_not_override_runtime_threads(self) -> None:
        args = parse_args([])
        calls: list[tuple[str, int]] = []

        actual = configure_torch_threading(
            num_threads=args.torch_num_threads,
            num_interop_threads=args.torch_num_interop_threads,
            set_num_threads=lambda value: calls.append(("threads", value)),
            set_num_interop_threads=lambda value: calls.append(("interop", value)),
            get_num_threads=lambda: 4,
            get_num_interop_threads=lambda: 4,
        )

        self.assertEqual(calls, [])
        self.assertEqual(actual, (4, 4))

    def test_torch_threading_cli_overrides_are_validated_and_applied(self) -> None:
        args = parse_args(["--torch-num-threads", "8", "--torch-num-interop-threads", "1"])
        calls: list[tuple[str, int]] = []
        current = {"threads": 4, "interop": 4}

        def set_threads(value: int) -> None:
            calls.append(("threads", value))
            current["threads"] = value

        def set_interop(value: int) -> None:
            calls.append(("interop", value))
            current["interop"] = value

        actual = configure_torch_threading(
            num_threads=args.torch_num_threads,
            num_interop_threads=args.torch_num_interop_threads,
            set_num_threads=set_threads,
            set_num_interop_threads=set_interop,
            get_num_threads=lambda: current["threads"],
            get_num_interop_threads=lambda: current["interop"],
        )

        self.assertEqual(calls, [("threads", 8), ("interop", 1)])
        self.assertEqual(actual, (8, 1))
        with self.assertRaises(SystemExit):
            parse_args(["--torch-num-threads", "0"])

    def test_training_metrics_jsonl_collects_native_timing_window_directly(self) -> None:
        timing = TrainingTimingWindow(started_at=0.0, window_started_at=0.0, start_step=0, window_start_step=1)
        timing.seconds_policy_select_window = 0.01
        timing.seconds_env_step_window = 0.02

        with TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "training_metrics.jsonl"

            append_training_metrics_jsonl(
                JointMetricsAggregator(),
                metrics_path,
                global_step=5,
                timing_window=timing,
            )
            row = json.loads(metrics_path.read_text(encoding="utf-8").strip())

        self.assertEqual(row["steps_window"], 4)
        self.assertGreaterEqual(row["elapsed_seconds_total"], 0.0)
        self.assertGreaterEqual(row["fps_total"], 0.0)
        self.assertGreaterEqual(row["fps_window"], 0.0)
        self.assertGreaterEqual(row["seconds_metrics_window"], 0.0)
        self.assertAlmostEqual(row["seconds_policy_select_window"], 0.01)
        self.assertAlmostEqual(row["seconds_env_step_window"], 0.02)

    def test_timing_window_reset_clears_dqn_window_counters(self) -> None:
        timing = TrainingTimingWindow(
            started_at=0.0,
            window_started_at=0.0,
            process_started_at=0.0,
            process_window_started_at=0.0,
        )
        timing.dqn_update_count_window = 1
        timing.dqn_gradient_steps_window = 3
        timing.dqn_replay_sample_seconds_window = 0.1
        timing.dqn_update_trigger_reason = "rollout_full"

        timing.reset_window(global_step=42, now=10.0, process_now=4.0)
        fields = timing.as_metrics_fields(global_step=42, now=12.0, process_now=5.0)

        self.assertEqual(fields["steps_window"], 0.0)
        self.assertEqual(fields["dqn_update_count_window"], 0.0)
        self.assertEqual(fields["dqn_gradient_steps_window"], 0.0)
        self.assertEqual(fields["dqn_replay_sample_seconds_window"], 0.0)
        self.assertEqual(fields["dqn_update_trigger_reason"], "")
        self.assertAlmostEqual(fields["process_cpu_seconds_window"], 1.0)

    def test_teacher_retention_config_is_disabled_by_default_and_requires_explicit_enable(self) -> None:
        self.assertTrue(hasattr(train_joint_torch, "teacher_retention_config_from_training_config"))

        config = train_joint_torch.teacher_retention_config_from_training_config({})
        args = parse_args([])

        self.assertFalse(config.enabled)
        self.assertFalse(args.enable_teacher_retention)
        self.assertIsNone(args.teacher_production_checkpoint)
        self.assertIsNone(args.teacher_candidate_checkpoint)

        with self.assertRaisesRegex(ValueError, "explicit_enable"):
            train_joint_torch.teacher_retention_config_from_training_config(
                {
                    "teacher_retention": {
                        "enabled": True,
                        "production_teacher_checkpoint": "production.pt",
                    }
                }
            )

        enabled = train_joint_torch.teacher_retention_config_from_training_config(
            {
                "teacher_retention": {
                    "enabled": True,
                    "explicit_enable": True,
                    "production_teacher_checkpoint": "production.pt",
                    "candidate_teacher_checkpoint": "candidate.pt",
                    "loss_weight": 0.12,
                    "max_loss": 0.3,
                }
            }
        )

        self.assertTrue(enabled.enabled)
        self.assertEqual(enabled.production_teacher_checkpoint, Path("production.pt"))
        self.assertEqual(enabled.candidate_teacher_checkpoint, Path("candidate.pt"))
        self.assertAlmostEqual(enabled.loss_weight, 0.12)
        self.assertAlmostEqual(enabled.max_loss, 0.3)

    def test_teacher_checkpoint_loads_read_only_and_rejects_contract_or_version_mismatch(self) -> None:
        self.assertTrue(hasattr(train_joint_torch, "load_teacher_policy_checkpoint"))
        source = JointPolicyBundle().to("cpu")

        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "teacher.pt"
            checkpoint = source.build_checkpoint(
                global_step=250_000,
                config={"mdp_contract_version": train_joint_torch.MDP_CONTRACT_VERSION},
                extra={},
            )
            torch.save(checkpoint, checkpoint_path)
            before_hash = _sha256(checkpoint_path)

            teacher = train_joint_torch.load_teacher_policy_checkpoint(
                checkpoint_path,
                device=torch.device("cpu"),
                teacher_name="candidate_250k",
            )

            self.assertEqual(_sha256(checkpoint_path), before_hash)
            self.assertEqual(teacher.name, "candidate_250k")
            self.assertEqual(teacher.checkpoint_path, checkpoint_path)
            self.assertFalse(teacher.policy.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in teacher.policy.parameters()))

            stale_contract_path = Path(tmp) / "stale_contract.pt"
            stale_contract = deepcopy(checkpoint)
            stale_contract["config"] = {"mdp_contract_version": "physical_reality_v4"}
            torch.save(stale_contract, stale_contract_path)
            with self.assertRaisesRegex(ValueError, "contract mismatch|Checkpoint MDP contract mismatch"):
                train_joint_torch.load_teacher_policy_checkpoint(
                    stale_contract_path,
                    device=torch.device("cpu"),
                    teacher_name="stale",
                )

            stale_version_path = Path(tmp) / "stale_version.pt"
            stale_version = deepcopy(checkpoint)
            stale_version["checkpoint_version"] = "torch_joint_policy_v0"
            torch.save(stale_version, stale_version_path)
            with self.assertRaisesRegex(ValueError, "checkpoint_version"):
                train_joint_torch.load_teacher_policy_checkpoint(
                    stale_version_path,
                    device=torch.device("cpu"),
                    teacher_name="stale_version",
                )

    def test_teacher_state_bank_records_gate_critical_scenario_counts_and_samples(self) -> None:
        self.assertTrue(hasattr(train_joint_torch, "TeacherRetentionStateBank"))
        bank = train_joint_torch.TeacherRetentionStateBank(max_per_scenario=2, observation_dim=OBSERVATION_DIM)
        stage_aliases = {
            "normal_v4": "baseline",
            "lead_time_delay": "lead_time",
            "route_disruption": "route_disruption",
            "premium_sla": "premium_sla",
            "high_holding_cost": "high_holding",
            "vehicle_scarcity": "vehicle_scarcity",
            "mixed_stress": "mixed_stress",
            "demand_spike": "demand_spike",
        }

        for index, (stage, scenario) in enumerate(stage_aliases.items()):
            bank.record(torch.full((OBSERVATION_DIM,), float(index)), stage)
            bank.record(torch.full((OBSERVATION_DIM,), float(index + 100)), scenario)
            bank.record(torch.full((OBSERVATION_DIM,), float(index + 200)), stage)

        with self.assertRaisesRegex(ValueError, "observation_dim"):
            bank.record(torch.zeros(OBSERVATION_DIM + 1), "normal_v4")

        self.assertEqual(bank.scenario_counts(), {scenario: 2 for scenario in stage_aliases.values()})
        sample = bank.sample(batch_size=16, device=torch.device("cpu"))

        self.assertEqual(tuple(sample.observations.shape), (16, OBSERVATION_DIM))
        self.assertEqual(set(sample.scenario_ids), set(stage_aliases.values()))
        self.assertEqual(sample.scenario_counts, {scenario: 2 for scenario in stage_aliases.values()})

    def test_teacher_retention_loss_is_finite_bounded_and_detects_bad_action_33_drift(self) -> None:
        self.assertTrue(hasattr(train_joint_torch, "compute_dqn_teacher_retention_loss"))
        config = train_joint_torch.teacher_retention_config_from_training_config(
            {
                "teacher_retention": {
                    "enabled": True,
                    "explicit_enable": True,
                    "production_teacher_checkpoint": "production.pt",
                    "candidate_teacher_checkpoint": "candidate.pt",
                    "loss_weight": 1.0,
                    "cross_entropy_weight": 1.0,
                    "kl_weight": 1.0,
                    "bad_pocket_margin_weight": 1.0,
                    "bad_pocket_margin": 0.25,
                    "max_loss": 0.25,
                    "temperature": 1.0,
                }
            }
        )
        teacher_q = torch.zeros((3, DISCRETE_ACTION_COUNT), dtype=torch.float32)
        teacher_q[:, 29] = 4.0
        matching_student_q = teacher_q.clone().detach().requires_grad_(True)
        bad_student_q = torch.zeros((3, DISCRETE_ACTION_COUNT), dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            bad_student_q[:, 29] = -2.0
            bad_student_q[:, 33] = 7.0

        matching = train_joint_torch.compute_dqn_teacher_retention_loss(
            student_q=matching_student_q,
            teacher_q=teacher_q,
            config=config,
            scenario_ids=("lead_time", "lead_time", "route_disruption"),
        )
        bad = train_joint_torch.compute_dqn_teacher_retention_loss(
            student_q=bad_student_q,
            teacher_q=teacher_q,
            config=config,
            scenario_ids=("lead_time", "lead_time", "route_disruption"),
        )

        self.assertTrue(torch.isfinite(bad.loss))
        self.assertLessEqual(float(bad.loss.detach().item()), config.max_loss)
        self.assertGreater(bad.metrics["teacher_dqn_kl"], matching.metrics["teacher_dqn_kl"])
        self.assertGreater(bad.metrics["teacher_dqn_ce"], matching.metrics["teacher_dqn_ce"])
        self.assertEqual(matching.metrics["teacher_dqn_action_match_rate"], 1.0)
        self.assertEqual(bad.metrics["teacher_dqn_action_match_rate"], 0.0)
        self.assertEqual(bad.metrics["gate_critical_action_drift"], 1.0)
        self.assertLess(bad.bad_pocket_margin_by_action["33"], 0.0)
        bad.loss.backward()
        self.assertIsNotNone(bad_student_q.grad)
        self.assertTrue(torch.isfinite(bad_student_q.grad).all())

    def test_teacher_retention_does_not_penalize_teacher_agreed_valid_bad_pocket_dispatch(self) -> None:
        self.assertTrue(hasattr(train_joint_torch, "compute_dqn_teacher_retention_loss"))
        config = train_joint_torch.teacher_retention_config_from_training_config(
            {
                "teacher_retention": {
                    "enabled": True,
                    "explicit_enable": True,
                    "candidate_teacher_checkpoint": "candidate.pt",
                    "loss_weight": 1.0,
                    "cross_entropy_weight": 1.0,
                    "kl_weight": 0.0,
                    "bad_pocket_margin_weight": 1.0,
                    "bad_pocket_margin": 0.25,
                    "max_loss": 1.0,
                }
            }
        )
        teacher_q = torch.zeros((2, DISCRETE_ACTION_COUNT), dtype=torch.float32)
        teacher_q[:, 33] = 5.0
        student_q = teacher_q.clone().detach().requires_grad_(True)

        result = train_joint_torch.compute_dqn_teacher_retention_loss(
            student_q=student_q,
            teacher_q=teacher_q,
            config=config,
            scenario_ids=("lead_time", "lead_time"),
        )

        self.assertLess(float(result.loss.detach().item()), 0.05)
        self.assertEqual(result.metrics["teacher_dqn_action_match_rate"], 1.0)
        self.assertEqual(result.metrics["gate_critical_action_drift"], 0.0)
        self.assertGreaterEqual(result.bad_pocket_margin_by_action["33"], 0.0)

    def test_teacher_retention_telemetry_is_aggregated_in_timing_metrics_and_resets(self) -> None:
        self.assertTrue(hasattr(train_joint_torch, "record_teacher_retention_metrics"))
        timing = TrainingTimingWindow(
            started_at=0.0,
            window_started_at=0.0,
            process_started_at=0.0,
            process_window_started_at=0.0,
        )
        metrics = {
            "teacher_dqn_kl": 0.20,
            "teacher_dqn_ce": 0.40,
            "teacher_dqn_action_match_rate": 0.75,
            "teacher_retention_loss": 0.10,
            "gate_critical_action_drift": 0.25,
            "bad_pocket_margin_by_action": {"32": 0.5, "33": -0.3, "41": 0.1, "45": 0.0, "46": -0.2},
            "teacher_bank_scenario_counts": {"lead_time": 3, "route_disruption": 1},
        }

        train_joint_torch.record_teacher_retention_metrics(timing, metrics)
        fields = timing.as_metrics_fields(global_step=4, now=1.0, process_now=1.0)

        self.assertAlmostEqual(fields["teacher_dqn_kl_window"], 0.20)
        self.assertAlmostEqual(fields["teacher_dqn_ce_window"], 0.40)
        self.assertAlmostEqual(fields["teacher_dqn_action_match_rate_window"], 0.75)
        self.assertAlmostEqual(fields["teacher_retention_loss_window"], 0.10)
        self.assertAlmostEqual(fields["gate_critical_action_drift_window"], 0.25)
        self.assertEqual(fields["bad_pocket_margin_by_action_window"]["33"], -0.3)
        self.assertEqual(fields["teacher_bank_scenario_counts"], {"lead_time": 3, "route_disruption": 1})

        timing.reset_window(global_step=4, now=1.0, process_now=1.0)
        reset_fields = timing.as_metrics_fields(global_step=4, now=1.5, process_now=1.5)

        self.assertEqual(reset_fields["teacher_dqn_kl_window"], 0.0)
        self.assertEqual(reset_fields["teacher_dqn_action_match_rate_window"], 0.0)
        self.assertEqual(reset_fields["teacher_retention_loss_window"], 0.0)
        self.assertEqual(reset_fields["gate_critical_action_drift_window"], 0.0)
        self.assertEqual(reset_fields["bad_pocket_margin_by_action_window"], {})
        self.assertEqual(reset_fields["teacher_bank_scenario_counts"], {})

    def test_update_dqn_records_window_timing_without_changing_update_count(self) -> None:
        torch.manual_seed(123)
        policies = JointPolicyBundle().to("cpu")
        optimizer = torch.optim.Adam(policies.dqn_q_network.parameters(), lr=1e-4)
        replay = DQNReplayBuffer(
            capacity=16,
            learning_starts=0,
            device="cpu",
            reward_blend=JointRewardBlendConfig(),
        )
        for index in range(8):
            replay.add(_dqn_transition(index))
        state = TrainingState()
        timing = TrainingTimingWindow(
            started_at=0.0,
            window_started_at=0.0,
            process_started_at=0.0,
            process_window_started_at=0.0,
        )

        loss = update_dqn(
            policies=policies,
            optimizer=optimizer,
            replay=replay,
            batch_size=4,
            gradient_steps=2,
            gamma=0.99,
            target_update_interval=1,
            tau=1.0,
            state=state,
            timing_window=timing,
            trigger_reason="unit_test",
        )

        self.assertIsInstance(loss, float)
        self.assertEqual(state.dqn_update_count, 2)
        self.assertEqual(timing.dqn_update_count_window, 1)
        self.assertEqual(timing.dqn_gradient_steps_window, 2)
        self.assertEqual(timing.dqn_sample_count_window, 8)
        self.assertEqual(timing.dqn_batch_size, 4)
        self.assertEqual(timing.dqn_replay_size, 8)
        self.assertEqual(timing.dqn_gradient_steps_per_rollout_effective, 2)
        self.assertEqual(timing.dqn_update_trigger_reason, "unit_test")
        timing_fields = timing.as_metrics_fields(global_step=8)
        self.assertEqual(timing_fields["teacher_retention_loss_window"], 0.0)
        self.assertEqual(timing_fields["teacher_bank_scenario_counts"], {})
        self.assertGreaterEqual(timing.seconds_dqn_update_window, 0.0)
        self.assertGreaterEqual(timing.dqn_replay_sample_seconds_window, 0.0)
        self.assertGreaterEqual(timing.dqn_tensor_build_seconds_window, 0.0)
        self.assertGreaterEqual(timing.dqn_forward_seconds_window, 0.0)
        self.assertGreaterEqual(timing.dqn_loss_seconds_window, 0.0)
        self.assertGreaterEqual(timing.dqn_backward_seconds_window, 0.0)
        self.assertGreaterEqual(timing.dqn_optimizer_seconds_window, 0.0)
        self.assertGreaterEqual(timing.dqn_target_update_seconds_window, 0.0)
        self.assertGreaterEqual(timing.dqn_validation_seconds_window, 0.0)

    def test_periodic_checkpoint_is_skipped_at_final_boundary(self) -> None:
        self.assertTrue(
            _should_save_periodic_checkpoint(
                global_step=9_000,
                next_checkpoint_step=9_000,
                total_timesteps=10_000,
            )
        )
        self.assertFalse(
            _should_save_periodic_checkpoint(
                global_step=10_000,
                next_checkpoint_step=10_000,
                total_timesteps=10_000,
            )
        )

    def test_completed_resume_does_not_rewrite_final_training_outputs(self) -> None:
        self.assertTrue(
            _should_write_final_training_outputs(
                start_step=7_000,
                global_step=10_000,
            )
        )
        self.assertFalse(
            _should_write_final_training_outputs(
                start_step=10_000,
                global_step=10_000,
            )
        )

    def test_completed_resume_does_not_call_outer_final_artifact_save(self) -> None:
        calls = 0

        def record_save() -> None:
            nonlocal calls
            calls += 1

        state = TrainingState(global_step=10_000)
        saved = _save_final_artifacts_if_training_advanced(
            start_step=10_000,
            state=state,
            save=record_save,
        )

        self.assertFalse(saved)
        self.assertEqual(calls, 0)

        saved = _save_final_artifacts_if_training_advanced(
            start_step=9_000,
            state=state,
            save=record_save,
        )

        self.assertTrue(saved)
        self.assertEqual(calls, 1)

    def test_resume_metrics_ahead_of_checkpoint_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "training_metrics.jsonl"
            metrics_path.write_text(
                '{"global_step":700000}\n{"global_step":712704}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "metrics.*ahead.*checkpoint"):
                validate_resume_metrics_checkpoint_alignment(metrics_path, checkpoint_global_step=700000)

    def test_resume_metrics_at_or_behind_checkpoint_is_allowed(self) -> None:
        with TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "training_metrics.jsonl"
            metrics_path.write_text(
                '{"global_step":696000}\n{"global_step":700000}\n',
                encoding="utf-8",
            )

            validate_resume_metrics_checkpoint_alignment(metrics_path, checkpoint_global_step=700000)

    def test_parse_args_accepts_init_from_joint_checkpoint_and_rejects_resume_combo(self) -> None:
        args = parse_args(["--init-from-joint-checkpoint", "parent.pt"])

        self.assertEqual(args.init_from_joint_checkpoint, "parent.pt")
        self.assertIsNone(args.resume)
        with self.assertRaises(SystemExit):
            parse_args(["--resume", "resume.pt", "--init-from-joint-checkpoint", "parent.pt"])

    def test_fine_tune_init_loads_weights_without_resume_state_or_optimizers(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "parent.pt"
            source = JointPolicyBundle()
            target = JointPolicyBundle()
            with torch.no_grad():
                for parameter in source.ppo_actor.parameters():
                    parameter.fill_(0.123)
            torch.save(
                source.build_checkpoint(
                    global_step=1_000_000,
                    config={
                        "mdp_contract_version": "physical_reality_v5_route_candidate_visibility",
                        "shared_global_parameters": {
                            "environment_id": "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean",
                            "observation_dim": OBSERVATION_DIM,
                        },
                    },
                ),
                checkpoint_path,
            )
            state = TrainingState(global_step=0, episode_count=0, dqn_update_count=0)
            ppo_optimizer = torch.optim.Adam(target.ppo_actor.parameters(), lr=0.001)
            dqn_optimizer = torch.optim.Adam(target.dqn_q_network.parameters(), lr=0.001)

            with self._allow_fine_tune_init_paths(checkpoint_path):
                metadata = train_joint_torch.load_fine_tune_initial_checkpoint(
                    checkpoint_path,
                    policies=target,
                    state=state,
                    device=torch.device("cpu"),
                )

            self.assertEqual(state.global_step, 0)
            self.assertEqual(state.episode_count, 0)
            self.assertEqual(state.dqn_update_count, 0)
            self.assertEqual(ppo_optimizer.state_dict()["state"], {})
            self.assertEqual(dqn_optimizer.state_dict()["state"], {})
            self.assertEqual(metadata["initialized_from_global_step"], 1_000_000)
            self.assertEqual(metadata["replay_restore_status"], "fresh_replay_by_design")
            self.assertEqual(
                metadata["initialized_from_env_id"],
                "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean",
            )
            self.assertEqual(metadata["initialized_from_contract"], "physical_reality_v5_route_candidate_visibility")
            self.assertEqual(metadata["fine_tune_parent_checkpoint"], str(checkpoint_path))
            self.assertTrue(
                torch.equal(
                    next(source.ppo_actor.parameters()),
                    next(target.ppo_actor.parameters()),
                )
            )

    def test_fine_tune_init_rejects_incompatible_action_count(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "bad_action_count.pt"
            source = JointPolicyBundle()
            checkpoint = source.build_checkpoint(
                global_step=1_000_000,
                config={
                    "mdp_contract_version": "physical_reality_v5_route_candidate_visibility",
                    "shared_global_parameters": {"observation_dim": OBSERVATION_DIM},
                },
            )
            checkpoint["discrete_action_count"] = DISCRETE_ACTION_COUNT + 1
            torch.save(checkpoint, checkpoint_path)

            with self._allow_fine_tune_init_paths(checkpoint_path):
                with self.assertRaisesRegex(ValueError, "discrete_action_count mismatch"):
                    train_joint_torch.load_fine_tune_initial_checkpoint(
                        checkpoint_path,
                        policies=JointPolicyBundle(),
                        state=TrainingState(),
                        device=torch.device("cpu"),
                    )

    def test_fine_tune_init_rejects_incompatible_contract_and_dimension(self) -> None:
        with TemporaryDirectory() as tmp:
            source = JointPolicyBundle()
            stale_contract_path = Path(tmp) / "stale_contract.pt"
            stale_contract = source.build_checkpoint(
                global_step=1_000_000,
                config={
                    "mdp_contract_version": "physical_reality_v4_real_world_stress_visibility",
                    "shared_global_parameters": {"observation_dim": OBSERVATION_DIM},
                },
            )
            torch.save(stale_contract, stale_contract_path)
            with self._allow_fine_tune_init_paths(stale_contract_path):
                with self.assertRaisesRegex(ValueError, "Checkpoint MDP contract mismatch"):
                    train_joint_torch.load_fine_tune_initial_checkpoint(
                        stale_contract_path,
                        policies=JointPolicyBundle(),
                        state=TrainingState(),
                        device=torch.device("cpu"),
                    )

            stale_dim_path = Path(tmp) / "stale_dim.pt"
            stale_dim = source.build_checkpoint(
                global_step=1_000_000,
                config={
                    "mdp_contract_version": "physical_reality_v5_route_candidate_visibility",
                    "shared_global_parameters": {"observation_dim": OBSERVATION_DIM},
                },
            )
            stale_dim["observation_dim"] = OBSERVATION_DIM - 1
            torch.save(stale_dim, stale_dim_path)
            with self._allow_fine_tune_init_paths(stale_dim_path):
                with self.assertRaisesRegex(ValueError, "observation_dim mismatch"):
                    train_joint_torch.load_fine_tune_initial_checkpoint(
                        stale_dim_path,
                        policies=JointPolicyBundle(),
                        state=TrainingState(),
                        device=torch.device("cpu"),
                    )

    def test_fine_tune_init_path_guard_rejects_invalid_old_artifact_paths(self) -> None:
        invalid_paths = (
            Path("models/checkpoints/joint_torch_v5_clean_cold_start_1m/joint_torch_latest.pt"),
            Path("models/checkpoints/joint_torch_v5_clean_cold_start_1m_after_rewardfix/joint_torch_latest.pt"),
            Path("models/checkpoints/some_invalid_700k_artifact/joint_torch_latest.pt"),
            Path("models/checkpoints/dqn1024_probe/joint_torch_latest.pt"),
        )

        for path in invalid_paths:
            with self.subTest(path=str(path)):
                with self.assertRaisesRegex(ValueError, "disallowed fine-tune initialization checkpoint path"):
                    train_joint_torch.validate_fine_tune_initial_checkpoint_path(path)

    def test_fine_tune_init_path_guard_requires_approved_parent_checkpoint(self) -> None:
        approved_parent = Path(
            "models/checkpoints/"
            "joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/"
            "joint_torch_latest.pt"
        )
        approved_production_parent = Path(
            "models/production/"
            "joint_torch_v5_balanced_retention_ft_200k_20260608/"
            "joint_torch_latest.pt"
        )
        train_joint_torch.validate_fine_tune_initial_checkpoint_path(approved_parent)
        train_joint_torch.validate_fine_tune_initial_checkpoint_path(approved_production_parent)

        with self.assertRaisesRegex(ValueError, "not approved"):
            train_joint_torch.validate_fine_tune_initial_checkpoint_path(
                Path("models/checkpoints/another_v5_compatible_run/joint_torch_latest.pt")
            )

    def test_checkpoint_payload_includes_fine_tune_initialization_metadata(self) -> None:
        parent = "models/checkpoints/parent/joint_torch_latest.pt"
        policies = JointPolicyBundle()
        optimizer = torch.optim.Adam(policies.ppo_actor.parameters(), lr=0.001)
        dqn_optimizer = torch.optim.Adam(policies.dqn_q_network.parameters(), lr=0.001)
        payload = train_joint_torch.build_checkpoint_payload(
            policies=policies,
            ppo_optimizer=optimizer,
            dqn_optimizer=dqn_optimizer,
            state=TrainingState(global_step=25),
            config={
                "mdp_contract_version": "physical_reality_v5_route_candidate_visibility",
                "fine_tune_initialization": {
                    "initialized_from_checkpoint": parent,
                    "initialized_from_env_id": "joint_curriculum_v5_clean_cold_start_1m_after_rewardfix_perfclean",
                    "initialized_from_global_step": 1_000_000,
                    "initialized_from_contract": "physical_reality_v5_route_candidate_visibility",
                    "fine_tune_parent_checkpoint": parent,
                },
            },
        )

        self.assertEqual(payload["initialized_from_checkpoint"], parent)
        self.assertEqual(payload["fine_tune_parent_checkpoint"], parent)
        self.assertEqual(payload["initialized_from_global_step"], 1_000_000)

    def test_checkpoint_payload_includes_replay_state_rng_state_and_resume_metadata(self) -> None:
        policies = JointPolicyBundle()
        optimizer = torch.optim.Adam(policies.ppo_actor.parameters(), lr=0.001)
        dqn_optimizer = torch.optim.Adam(policies.dqn_q_network.parameters(), lr=0.001)
        replay = _filled_replay(capacity=4, transition_count=6)
        rng_state = train_joint_torch.capture_rng_state()

        payload = train_joint_torch.build_checkpoint_payload(
            policies=policies,
            ppo_optimizer=optimizer,
            dqn_optimizer=dqn_optimizer,
            state=TrainingState(global_step=25, episode_count=3, dqn_update_count=7),
            config={"mdp_contract_version": "physical_reality_v5_route_candidate_visibility"},
            dqn_replay=replay,
            rng_state=rng_state,
        )

        self.assertIn(train_joint_torch.DQN_REPLAY_STATE_KEY, payload)
        self.assertIn(train_joint_torch.RNG_STATE_KEY, payload)
        self.assertIn(train_joint_torch.RESUME_STATE_METADATA_KEY, payload)
        self.assertEqual(payload[train_joint_torch.DQN_REPLAY_STATE_KEY]["size"], len(replay))
        self.assertEqual(payload[train_joint_torch.DQN_REPLAY_STATE_KEY]["position"], 2)
        resume_state = payload[train_joint_torch.RESUME_STATE_METADATA_KEY]
        self.assertEqual(resume_state["resume_state_version"], "joint_trainer_resume_state_v1")
        self.assertTrue(resume_state["dqn_replay_buffer_included"])
        self.assertEqual(resume_state["dqn_replay_size"], len(replay))
        self.assertTrue(resume_state["rng_state_included"])
        self.assertEqual(resume_state["global_step"], 25)

    def test_hierarchical_dqn_architecture_is_read_from_config(self) -> None:
        config = {"torch_training": {"dqn_architecture": HIERARCHICAL_DQN_ARCHITECTURE}}

        self.assertEqual(HIERARCHICAL_DQN_ARCHITECTURE, train_joint_torch._dqn_architecture_from_config(config))
        self.assertEqual(FLAT_DQN_ARCHITECTURE, train_joint_torch._dqn_architecture_from_config({}))
        with self.assertRaisesRegex(ValueError, "unsupported dqn_architecture"):
            train_joint_torch._dqn_architecture_from_config({"torch_training": {"dqn_architecture": "flat_but_spicy"}})

    def test_hierarchical_checkpoint_payload_includes_architecture_and_exact_resume_metadata(self) -> None:
        policies = JointPolicyBundle(dqn_architecture=HIERARCHICAL_DQN_ARCHITECTURE)
        optimizer = torch.optim.Adam(policies.ppo_actor.parameters(), lr=0.001)
        dqn_optimizer = torch.optim.Adam(policies.dqn_q_network.parameters(), lr=0.001)
        replay = _filled_replay(capacity=4, transition_count=6)

        payload = train_joint_torch.build_checkpoint_payload(
            policies=policies,
            ppo_optimizer=optimizer,
            dqn_optimizer=dqn_optimizer,
            state=TrainingState(global_step=25),
            config={"mdp_contract_version": "physical_reality_v5_route_candidate_visibility"},
            dqn_replay=replay,
        )

        self.assertEqual(HIERARCHICAL_DQN_ARCHITECTURE, payload["dqn_architecture"])
        self.assertEqual(DISCRETE_ACTION_COUNT, payload["external_discrete_action_count"])
        self.assertEqual(2, payload["internal_heads"]["dispatch"])
        self.assertIn(train_joint_torch.DQN_REPLAY_STATE_KEY, payload)
        self.assertIn(train_joint_torch.RNG_STATE_KEY, payload)
        self.assertTrue(payload[train_joint_torch.RESUME_STATE_METADATA_KEY]["exact_resume_capable"])

    def test_checkpoint_load_rejects_flat_hierarchical_architecture_mismatch(self) -> None:
        flat_checkpoint = JointPolicyBundle(dqn_architecture=FLAT_DQN_ARCHITECTURE).build_checkpoint(
            global_step=5,
            config={"mdp_contract_version": "physical_reality_v5_route_candidate_visibility"},
        )
        hierarchical_checkpoint = JointPolicyBundle(dqn_architecture=HIERARCHICAL_DQN_ARCHITECTURE).build_checkpoint(
            global_step=5,
            config={"mdp_contract_version": "physical_reality_v5_route_candidate_visibility"},
        )

        with self.assertRaisesRegex(ValueError, "dqn_architecture mismatch"):
            JointPolicyBundle(dqn_architecture=HIERARCHICAL_DQN_ARCHITECTURE).load_checkpoint_state(flat_checkpoint)
        with self.assertRaisesRegex(ValueError, "dqn_architecture mismatch"):
            JointPolicyBundle(dqn_architecture=FLAT_DQN_ARCHITECTURE).load_checkpoint_state(hierarchical_checkpoint)

    def test_checkpoint_payload_includes_teacher_retention_state_bank_when_present(self) -> None:
        policies = JointPolicyBundle()
        optimizer = torch.optim.Adam(policies.ppo_actor.parameters(), lr=0.001)
        dqn_optimizer = torch.optim.Adam(policies.dqn_q_network.parameters(), lr=0.001)
        replay = _filled_replay(capacity=4, transition_count=4)
        teacher_bank = train_joint_torch.TeacherRetentionStateBank(
            max_per_scenario=2,
            observation_dim=OBSERVATION_DIM,
        )
        teacher_bank.record(torch.ones(OBSERVATION_DIM), "lead_time_delay")
        teacher_bank.record(torch.full((OBSERVATION_DIM,), 2.0), "route_disruption")

        payload = train_joint_torch.build_checkpoint_payload(
            policies=policies,
            ppo_optimizer=optimizer,
            dqn_optimizer=dqn_optimizer,
            state=TrainingState(global_step=4),
            config={"mdp_contract_version": "physical_reality_v5_route_candidate_visibility"},
            dqn_replay=replay,
            teacher_retention_state_bank=teacher_bank,
        )

        self.assertIn(train_joint_torch.TEACHER_RETENTION_STATE_BANK_KEY, payload)
        resume_state = payload[train_joint_torch.RESUME_STATE_METADATA_KEY]
        self.assertTrue(resume_state["teacher_retention_state_bank_included"])
        self.assertEqual(resume_state["teacher_retention_state_bank_size"], 2)
        self.assertEqual(
            resume_state["teacher_retention_state_bank_scenario_counts"],
            {"lead_time": 1, "route_disruption": 1},
        )

    def test_periodic_checkpoint_save_writes_replay_state_to_joint_latest_only(self) -> None:
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            policies = JointPolicyBundle()
            optimizer = torch.optim.Adam(policies.ppo_actor.parameters(), lr=0.001)
            dqn_optimizer = torch.optim.Adam(policies.dqn_q_network.parameters(), lr=0.001)
            replay = _filled_replay(capacity=4, transition_count=6)
            state = TrainingState(global_step=100, episode_count=2, dqn_update_count=3)

            train_joint_torch.save_periodic_checkpoint(
                policies=policies,
                ppo_optimizer=optimizer,
                dqn_optimizer=dqn_optimizer,
                state=state,
                config={"mdp_contract_version": "physical_reality_v5_route_candidate_visibility"},
                dqn_replay=replay,
                output_dir=output_dir,
            )

            latest = torch.load(output_dir / "joint_torch_latest.pt", map_location="cpu", weights_only=False)
            ppo_artifact = torch.load(output_dir / "ppo_torch_joint_100.pt", map_location="cpu", weights_only=False)

            self.assertIn(train_joint_torch.DQN_REPLAY_STATE_KEY, latest)
            self.assertIn(train_joint_torch.RNG_STATE_KEY, latest)
            self.assertEqual(latest[train_joint_torch.DQN_REPLAY_STATE_KEY]["size"], len(replay))
            self.assertTrue(latest[train_joint_torch.RESUME_STATE_METADATA_KEY]["exact_resume_capable"])
            self.assertNotIn(train_joint_torch.DQN_REPLAY_STATE_KEY, ppo_artifact)
            self.assertFalse(ppo_artifact[train_joint_torch.RESUME_STATE_METADATA_KEY]["exact_resume_capable"])

    def test_load_training_checkpoint_restores_replay_rng_and_global_step(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "joint_torch_latest.pt"
            source_policies = JointPolicyBundle()
            target_policies = JointPolicyBundle()
            source_ppo_optimizer = torch.optim.Adam(source_policies.ppo_actor.parameters(), lr=0.001)
            source_dqn_optimizer = torch.optim.Adam(source_policies.dqn_q_network.parameters(), lr=0.001)
            target_ppo_optimizer = torch.optim.Adam(target_policies.ppo_actor.parameters(), lr=0.001)
            target_dqn_optimizer = torch.optim.Adam(target_policies.dqn_q_network.parameters(), lr=0.001)
            source_replay = _filled_replay(capacity=4, transition_count=6)
            target_replay = DQNReplayBuffer(
                capacity=4,
                learning_starts=0,
                device="cpu",
                reward_blend=JointRewardBlendConfig(),
            )

            random.seed(11)
            np.random.seed(22)
            torch.manual_seed(33)
            rng_state = train_joint_torch.capture_rng_state()
            expected_random = random.random()
            expected_numpy = float(np.random.random())
            expected_torch = float(torch.rand(1).item())
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)

            torch.save(
                train_joint_torch.build_checkpoint_payload(
                    policies=source_policies,
                    ppo_optimizer=source_ppo_optimizer,
                    dqn_optimizer=source_dqn_optimizer,
                    state=TrainingState(global_step=12_345, episode_count=9, dqn_update_count=17),
                    config={"mdp_contract_version": "physical_reality_v5_route_candidate_visibility"},
                    dqn_replay=source_replay,
                    rng_state=rng_state,
                ),
                checkpoint_path,
            )
            restored_state = TrainingState()

            metadata = train_joint_torch.load_training_checkpoint(
                checkpoint_path,
                policies=target_policies,
                ppo_optimizer=target_ppo_optimizer,
                dqn_optimizer=target_dqn_optimizer,
                state=restored_state,
                dqn_replay=target_replay,
                device=torch.device("cpu"),
                allow_empty_replay_resume=False,
            )

            self.assertEqual(restored_state.global_step, 12_345)
            self.assertEqual(restored_state.episode_count, 9)
            self.assertEqual(restored_state.dqn_update_count, 17)
            self.assertEqual(len(target_replay), len(source_replay))
            self.assertEqual(target_replay.total_added, source_replay.total_added)
            indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
            source_batch = source_replay._sample_at_indices(indices)
            target_batch = target_replay._sample_at_indices(indices)
            self.assertTrue(torch.equal(target_batch.observations, source_batch.observations))
            self.assertTrue(torch.equal(target_batch.actions, source_batch.actions))
            self.assertTrue(torch.equal(target_batch.rewards, source_batch.rewards))
            self.assertTrue(torch.equal(target_batch.next_observations, source_batch.next_observations))
            self.assertTrue(torch.equal(target_batch.dones, source_batch.dones))
            self.assertTrue(metadata["dqn_replay_restored"])
            self.assertEqual(metadata["dqn_replay_size"], len(source_replay))
            self.assertTrue(metadata["rng_state_restored"])
            self.assertAlmostEqual(random.random(), expected_random)
            self.assertAlmostEqual(float(np.random.random()), expected_numpy)
            self.assertAlmostEqual(float(torch.rand(1).item()), expected_torch)

    def test_load_training_checkpoint_restores_teacher_retention_state_bank_when_required(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "joint_torch_latest.pt"
            source_policies = JointPolicyBundle()
            target_policies = JointPolicyBundle()
            source_ppo_optimizer = torch.optim.Adam(source_policies.ppo_actor.parameters(), lr=0.001)
            source_dqn_optimizer = torch.optim.Adam(source_policies.dqn_q_network.parameters(), lr=0.001)
            target_ppo_optimizer = torch.optim.Adam(target_policies.ppo_actor.parameters(), lr=0.001)
            target_dqn_optimizer = torch.optim.Adam(target_policies.dqn_q_network.parameters(), lr=0.001)
            source_replay = _filled_replay(capacity=4, transition_count=4)
            target_replay = DQNReplayBuffer(
                capacity=4,
                learning_starts=0,
                device="cpu",
                reward_blend=JointRewardBlendConfig(),
            )
            source_bank = train_joint_torch.TeacherRetentionStateBank(
                max_per_scenario=2,
                observation_dim=OBSERVATION_DIM,
            )
            source_bank.record(torch.full((OBSERVATION_DIM,), 3.0), "lead_time_delay")
            source_bank.record(torch.full((OBSERVATION_DIM,), 4.0), "route_disruption")
            target_bank = train_joint_torch.TeacherRetentionStateBank(
                max_per_scenario=2,
                observation_dim=OBSERVATION_DIM,
            )
            payload = train_joint_torch.build_checkpoint_payload(
                policies=source_policies,
                ppo_optimizer=source_ppo_optimizer,
                dqn_optimizer=source_dqn_optimizer,
                state=TrainingState(global_step=4),
                config={"mdp_contract_version": "physical_reality_v5_route_candidate_visibility"},
                dqn_replay=source_replay,
                teacher_retention_state_bank=source_bank,
            )
            torch.save(payload, checkpoint_path)

            metadata = train_joint_torch.load_training_checkpoint(
                checkpoint_path,
                policies=target_policies,
                ppo_optimizer=target_ppo_optimizer,
                dqn_optimizer=target_dqn_optimizer,
                state=TrainingState(),
                dqn_replay=target_replay,
                device=torch.device("cpu"),
                teacher_retention_state_bank=target_bank,
                require_teacher_retention_state=True,
            )

            self.assertEqual(metadata["teacher_retention_state_status"], "restored")
            self.assertEqual(target_bank.scenario_counts(), {"lead_time": 1, "route_disruption": 1})

            missing_payload = dict(payload)
            missing_payload.pop(train_joint_torch.TEACHER_RETENTION_STATE_BANK_KEY)
            torch.save(missing_payload, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "teacher retention state bank is missing"):
                train_joint_torch.load_training_checkpoint(
                    checkpoint_path,
                    policies=JointPolicyBundle(),
                    ppo_optimizer=torch.optim.Adam(JointPolicyBundle().ppo_actor.parameters(), lr=0.001),
                    dqn_optimizer=torch.optim.Adam(JointPolicyBundle().dqn_q_network.parameters(), lr=0.001),
                    state=TrainingState(),
                    dqn_replay=DQNReplayBuffer(
                        capacity=4,
                        learning_starts=0,
                        device="cpu",
                        reward_blend=JointRewardBlendConfig(),
                    ),
                    device=torch.device("cpu"),
                    teacher_retention_state_bank=train_joint_torch.TeacherRetentionStateBank(
                        max_per_scenario=2,
                        observation_dim=OBSERVATION_DIM,
                    ),
                    require_teacher_retention_state=True,
                )

    def test_load_training_checkpoint_rejects_missing_replay_unless_explicitly_allowed(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "old_warm_resume_only.pt"
            policies = JointPolicyBundle()
            ppo_optimizer = torch.optim.Adam(policies.ppo_actor.parameters(), lr=0.001)
            dqn_optimizer = torch.optim.Adam(policies.dqn_q_network.parameters(), lr=0.001)
            torch.save(
                train_joint_torch.build_checkpoint_payload(
                    policies=policies,
                    ppo_optimizer=ppo_optimizer,
                    dqn_optimizer=dqn_optimizer,
                    state=TrainingState(global_step=250_000),
                    config={"mdp_contract_version": "physical_reality_v5_route_candidate_visibility"},
                    include_replay_state=False,
                    include_rng_state=False,
                ),
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "DQN replay buffer state"):
                train_joint_torch.load_training_checkpoint(
                    checkpoint_path,
                    policies=JointPolicyBundle(),
                    ppo_optimizer=torch.optim.Adam(JointPolicyBundle().ppo_actor.parameters(), lr=0.001),
                    dqn_optimizer=torch.optim.Adam(JointPolicyBundle().dqn_q_network.parameters(), lr=0.001),
                    state=TrainingState(),
                    dqn_replay=DQNReplayBuffer(
                        capacity=4,
                        learning_starts=0,
                        device="cpu",
                        reward_blend=JointRewardBlendConfig(),
                    ),
                    device=torch.device("cpu"),
                    allow_empty_replay_resume=False,
                )

            replay = DQNReplayBuffer(
                capacity=4,
                learning_starts=0,
                device="cpu",
                reward_blend=JointRewardBlendConfig(),
            )
            state = TrainingState()
            metadata = train_joint_torch.load_training_checkpoint(
                checkpoint_path,
                policies=JointPolicyBundle(),
                ppo_optimizer=torch.optim.Adam(JointPolicyBundle().ppo_actor.parameters(), lr=0.001),
                dqn_optimizer=torch.optim.Adam(JointPolicyBundle().dqn_q_network.parameters(), lr=0.001),
                state=state,
                dqn_replay=replay,
                device=torch.device("cpu"),
                allow_empty_replay_resume=True,
            )

            self.assertEqual(state.global_step, 250_000)
            self.assertEqual(len(replay), 0)
            self.assertFalse(metadata["dqn_replay_restored"])
            self.assertEqual(metadata["replay_restore_status"], "explicit_empty_replay_warm_resume")


def _neutral_joint_action() -> dict[str, object]:
    return {
        "continuous": [-1.0, -1.0, 0.0, -1.0, -1.0],
        "discrete": 0,
    }


def _filled_replay(*, capacity: int, transition_count: int) -> DQNReplayBuffer:
    replay = DQNReplayBuffer(
        capacity=capacity,
        learning_starts=0,
        device="cpu",
        reward_blend=JointRewardBlendConfig(),
    )
    for index in range(transition_count):
        replay.add(_dqn_transition(index))
    return replay


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dqn_transition(
    index: int,
    *,
    reward_dqn_local: float = 0.1,
    info: dict[str, object] | None = None,
) -> JointTransition:
    return JointTransition.from_reward_components(
        observation=torch.full((OBSERVATION_DIM,), float(index) / 10.0, dtype=torch.float32),
        ppo_action=torch.zeros(CONTINUOUS_ACTION_DIM, dtype=torch.float32),
        ppo_log_prob=torch.tensor(0.0, dtype=torch.float32),
        ppo_value=torch.tensor(0.0, dtype=torch.float32),
        dqn_action=torch.tensor(index % DISCRETE_ACTION_COUNT, dtype=torch.long),
        reward_total=0.1,
        reward_global=0.1,
        reward_ppo_local=0.0,
        reward_dqn_local=reward_dqn_local,
        reward_blend=JointRewardBlendConfig(),
        next_observation=torch.full((OBSERVATION_DIM,), float(index + 1) / 10.0, dtype=torch.float32),
        terminated=False,
        truncated=False,
        projected=False,
        blocked=False,
        info=info,
    )


class _DeepcopyTrapDict(dict[str, float]):
    def __deepcopy__(self, memo: dict[object, object]) -> "_DeepcopyTrapDict":
        raise AssertionError("db_snapshots should not be deep-copied during episode reset")


if __name__ == "__main__":
    unittest.main()
