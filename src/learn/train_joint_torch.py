"""Raw PyTorch synchronized PPO+DQN joint MARL trainer."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Final
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.env_5pl import EnvironmentConfig, FivePLDigitalTwinEnv
from src.act.observation_builder import OBSERVATION_DIM
from src.eval.scenario_overrides import apply_config_overrides, apply_environment_overrides, capability_matrix_as_dict
from src.learn.curriculum import CurriculumSample, CurriculumSampler, curriculum_config_from_training_config
from src.learn.episode_store import EpisodeStore, EpisodeTransition
from src.learn.joint_buffers import (
    DQNReplayBatch,
    DQNReplayBuffer,
    JointRewardBlendConfig,
    JointTransition,
    PPORolloutBuffer,
)
from src.learn.joint_metrics import JointMetricsAggregator
from src.learn.model_registry import ModelRegistry
from src.learn.teacher_retention import (
    TEACHER_RETENTION_STATE_BANK_KEY,
    TeacherRetentionConfig,
    TeacherRetentionRuntime,
    TeacherRetentionStateBank,
    compute_dqn_teacher_retention_loss,
    load_teacher_policy_checkpoint,
    record_teacher_retention_metrics,
    teacher_retention_config_from_training_config,
)
from src.learn.hierarchical_dqn_initialization import (
    HIERARCHICAL_INIT_METHOD_KEY,
    hierarchical_dqn_initialization_config_from_training_config,
    load_hierarchical_distillation_initial_checkpoint,
)
from src.sense.db_pool import DatabasePool
from src.think.joint_policies import (
    FLAT_DQN_ARCHITECTURE,
    HIERARCHICAL_DQN_ARCHITECTURE,
    SUPPORTED_DQN_ARCHITECTURES,
    JointPolicyBundle,
)


PPO_POLICY_ID: Final[str] = "ppo_torch_joint"
PPO_AGENT_ID: Final[str] = "ppo_strategic_controller"
PPO_AGENT_ROLE: Final[str] = "continuous_control"
DQN_POLICY_ID: Final[str] = "dqn_torch_joint"
DQN_AGENT_ID: Final[str] = "dqn_tactical_controller"
DQN_AGENT_ROLE: Final[str] = "tactical_dispatch"
MDP_CONTRACT_VERSION: Final[str] = "physical_reality_v5_route_candidate_visibility"
MDP_CONTRACT_MISMATCH_MESSAGE: Final[str] = (
    "Checkpoint MDP contract mismatch: refusing to resume old-policy weights under physical_reality_v5_route_candidate_visibility. "
    "Start from cold state."
)
FINE_TUNE_INITIALIZATION_KEY: Final[str] = "fine_tune_initialization"
DQN_REPLAY_STATE_KEY: Final[str] = "dqn_replay_buffer_state"
RNG_STATE_KEY: Final[str] = "rng_state"
RESUME_STATE_METADATA_KEY: Final[str] = "resume_state"
RESUME_STATE_VERSION: Final[str] = "joint_trainer_resume_state_v1"
RNG_STATE_VERSION: Final[str] = "torch_joint_rng_state_v1"
FINE_TUNE_INIT_METADATA_KEYS: Final[tuple[str, ...]] = (
    "initialized_from_checkpoint",
    "initialized_from_env_id",
    "initialized_from_global_step",
    "initialized_from_contract",
    "fine_tune_parent_checkpoint",
    HIERARCHICAL_INIT_METHOD_KEY,
    "hierarchical_teacher_checkpoint",
    "hierarchical_init_source_dqn_architecture",
    "hierarchical_init_target_dqn_architecture",
    "hierarchical_init_ppo_loaded_from_teacher",
    "hierarchical_init_dqn_trunk_loaded_from_teacher",
    "hierarchical_init_dqn_trunk_tensor_count",
    "hierarchical_distillation_loss",
    "hierarchical_distillation_raw_loss",
    "hierarchical_distillation_q_loss",
    "hierarchical_distillation_action_ce",
    "hierarchical_distillation_action_match_rate",
    "hierarchical_distillation_teacher_entropy",
    "hierarchical_distillation_initial_loss",
    "hierarchical_distillation_steps",
    "hierarchical_distillation_observation_sample_count",
    "hierarchical_distillation_batch_size",
)
APPROVED_FINE_TUNE_INIT_CHECKPOINTS: Final[frozenset[str]] = frozenset(
    {
        "models/checkpoints/"
        "joint_torch_v5_clean_cold_start_1m_after_rewardfix_perfclean_20260608/"
        "joint_torch_latest.pt",
        "models/production/"
        "joint_torch_v5_balanced_retention_ft_200k_20260608/"
        "joint_torch_latest.pt",
    }
)
DISALLOWED_FINE_TUNE_INIT_PATH_PARTS: Final[frozenset[str]] = frozenset(
    {
        "joint_torch_v5_clean_cold_start_1m",
        "joint_torch_v5_clean_cold_start_1m_after_rewardfix",
    }
)
DISALLOWED_FINE_TUNE_INIT_PATH_TOKENS: Final[tuple[str, ...]] = (
    "700k",
    "physical_reality_v3",
    "physical_reality_v4",
    "dqn1024",
    "dqn512",
)


@dataclass(frozen=True, slots=True)
class TraceLoggingConfig:
    """Trace logging controls consumed by the joint trainer."""

    enabled: bool = True
    flush_interval: int = 5_000
    sample_interval: int = 1
    environment_id: str = "joint_rolling_5pl_288"
    team_id: str = "joint_from_scratch"

    def __post_init__(self) -> None:
        if self.flush_interval <= 0:
            raise ValueError("flush_interval must be positive.")
        if not self.environment_id:
            raise ValueError("environment_id must not be empty.")
        if not self.team_id:
            raise ValueError("team_id must not be empty.")


@dataclass(slots=True)
class JointTraceBatchBuffer:
    """Accumulate joint trace rows and flush through `EpisodeStore.write_many`."""

    store: EpisodeStore
    flush_interval: int
    _pending: list[EpisodeTransition] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.flush_interval <= 0:
            raise ValueError("flush_interval must be positive.")

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def extend(self, rows: Iterable[EpisodeTransition]) -> None:
        self._pending.extend(rows)

    async def maybe_flush(self) -> int:
        if len(self._pending) < self.flush_interval:
            return 0
        return await self.flush()

    async def flush(self) -> int:
        if not self._pending:
            return 0
        batch = list(self._pending)
        self._pending.clear()
        try:
            return await self.store.write_many(batch)
        except BaseException:
            self._pending[:0] = batch
            raise


class JointTraceRecorder:
    """Trainer-facing trace sink with optional PostgreSQL persistence."""

    def __init__(self, buffer: JointTraceBatchBuffer | None, *, config: TraceLoggingConfig) -> None:
        if config.enabled and buffer is None:
            raise ValueError("enabled trace logging requires a JointTraceBatchBuffer.")
        self._buffer = buffer
        self.config = config

    @classmethod
    def enabled(cls, store: EpisodeStore, *, config: TraceLoggingConfig) -> JointTraceRecorder:
        if not config.enabled:
            raise ValueError("TraceLoggingConfig.enabled must be true for JointTraceRecorder.enabled().")
        return cls(JointTraceBatchBuffer(store=store, flush_interval=config.flush_interval), config=config)

    @classmethod
    def disabled(cls, *, config: TraceLoggingConfig | None = None) -> JointTraceRecorder:
        disabled_config = config or TraceLoggingConfig(enabled=False)
        if disabled_config.enabled:
            disabled_config = TraceLoggingConfig(
                enabled=False,
                flush_interval=disabled_config.flush_interval,
                sample_interval=disabled_config.sample_interval,
                environment_id=disabled_config.environment_id,
                team_id=disabled_config.team_id,
            )
        return cls(None, config=disabled_config)

    @property
    def enabled_for_writes(self) -> bool:
        return self._buffer is not None and self.config.enabled

    @property
    def pending_count(self) -> int:
        if self._buffer is None:
            return 0
        return self._buffer.pending_count

    def should_record_step(self, global_step: int) -> bool:
        if self._buffer is None:
            return False
        if self.config.sample_interval <= 1:
            return True
        return global_step % self.config.sample_interval == 0

    async def record_transition(self, transition: JointTransition, *, recorded_at: datetime | None = None) -> int:
        if self._buffer is None:
            return 0
        self._buffer.extend(
            build_joint_trace_rows(
                transition,
                recorded_at=recorded_at,
                environment_id=self.config.environment_id,
                team_id=self.config.team_id,
            )
        )
        return await self._buffer.maybe_flush()

    async def record_transitions(self, transitions: Iterable[JointTransition]) -> int:
        if self._buffer is None:
            return 0
        self._buffer.extend(
            build_joint_trace_batch(
                transitions,
                environment_id=self.config.environment_id,
                team_id=self.config.team_id,
            )
        )
        return await self._buffer.maybe_flush()

    async def flush(self) -> int:
        if self._buffer is None:
            return 0
        return await self._buffer.flush()


@dataclass(slots=True)
class TrainingState:
    global_step: int = 0
    episode_count: int = 0
    dqn_update_count: int = 0
    last_ppo_loss: float | None = None
    last_dqn_loss: float | None = None


@dataclass(slots=True)
class TrainingTimingWindow:
    started_at: float
    window_started_at: float
    process_started_at: float = field(default_factory=time.process_time)
    process_window_started_at: float = field(default_factory=time.process_time)
    cpu_logical_cores: int = field(default_factory=lambda: max(1, int(os.cpu_count() or 1)))
    torch_num_threads: int = field(default_factory=torch.get_num_threads)
    torch_num_interop_threads: int = field(default_factory=torch.get_num_interop_threads)
    start_step: int = 0
    window_start_step: int = 0
    seconds_policy_select_window: float = 0.0
    seconds_env_step_window: float = 0.0
    seconds_buffer_add_window: float = 0.0
    seconds_ppo_update_window: float = 0.0
    seconds_dqn_update_window: float = 0.0
    seconds_metrics_window: float = 0.0
    seconds_checkpoint_window: float = 0.0
    dqn_update_count_window: int = 0
    dqn_gradient_steps_window: int = 0
    dqn_replay_sample_seconds_window: float = 0.0
    dqn_tensor_build_seconds_window: float = 0.0
    dqn_forward_seconds_window: float = 0.0
    dqn_loss_seconds_window: float = 0.0
    dqn_backward_seconds_window: float = 0.0
    dqn_optimizer_seconds_window: float = 0.0
    dqn_target_update_seconds_window: float = 0.0
    dqn_validation_seconds_window: float = 0.0
    dqn_misc_seconds_window: float = 0.0
    dqn_sample_count_window: int = 0
    dqn_replay_size: int = 0
    dqn_batch_size: int = 0
    dqn_gradient_steps_per_rollout_effective: int = 0
    dqn_update_trigger_reason: str = ""
    teacher_retention_updates_window: int = 0
    teacher_dqn_kl_sum_window: float = 0.0
    teacher_dqn_ce_sum_window: float = 0.0
    teacher_dqn_action_match_rate_sum_window: float = 0.0
    teacher_retention_loss_sum_window: float = 0.0
    gate_critical_action_drift_sum_window: float = 0.0
    bad_pocket_margin_sum_by_action_window: dict[str, float] = field(default_factory=dict)
    bad_pocket_margin_count_by_action_window: dict[str, int] = field(default_factory=dict)
    teacher_bank_scenario_counts: dict[str, int] = field(default_factory=dict)
    ppo_update_count_window: int = 0
    ppo_minibatch_steps_window: int = 0
    ppo_forward_backward_seconds_window: float = 0.0
    ppo_optimizer_seconds_window: float = 0.0

    def as_metrics_fields(
        self,
        *,
        global_step: int,
        now: float | None = None,
        process_now: float | None = None,
    ) -> dict[str, float | str]:
        current = time.perf_counter() if now is None else float(now)
        current_process = time.process_time() if process_now is None else float(process_now)
        elapsed_total = max(0.0, current - self.started_at)
        elapsed_window = max(0.0, current - self.window_started_at)
        process_elapsed_total = max(0.0, current_process - self.process_started_at)
        process_elapsed_window = max(0.0, current_process - self.process_window_started_at)
        total_steps = max(0, int(global_step) - int(self.start_step))
        window_steps = max(0, int(global_step) - int(self.window_start_step))
        cpu_logical_cores = max(1, int(self.cpu_logical_cores))
        cpu_parallelism_total = process_elapsed_total / elapsed_total if elapsed_total > 0.0 else 0.0
        cpu_parallelism_window = process_elapsed_window / elapsed_window if elapsed_window > 0.0 else 0.0
        dqn_update_seconds = max(0.0, self.seconds_dqn_update_window)
        dqn_sample_count = max(0, int(self.dqn_sample_count_window))
        dqn_gradient_steps = max(0, int(self.dqn_gradient_steps_window))
        dqn_avg_denominator = float(dqn_gradient_steps) if dqn_gradient_steps > 0 else 0.0
        teacher_updates = max(0, int(self.teacher_retention_updates_window))
        teacher_denominator = float(teacher_updates) if teacher_updates > 0 else 0.0
        bad_pocket_margins = {
            action_id: float(total) / float(max(1, self.bad_pocket_margin_count_by_action_window.get(action_id, 0)))
            for action_id, total in sorted(self.bad_pocket_margin_sum_by_action_window.items())
            if self.bad_pocket_margin_count_by_action_window.get(action_id, 0) > 0
        }
        return {
            "elapsed_seconds_total": float(elapsed_total),
            "fps_total": float(total_steps / elapsed_total) if elapsed_total > 0.0 else 0.0,
            "fps_window": float(window_steps / elapsed_window) if elapsed_window > 0.0 else 0.0,
            "process_cpu_seconds_total": float(process_elapsed_total),
            "process_cpu_seconds_window": float(process_elapsed_window),
            "cpu_parallelism_total": float(cpu_parallelism_total),
            "cpu_parallelism_window": float(cpu_parallelism_window),
            "cpu_utilization_pct_total": float((cpu_parallelism_total / cpu_logical_cores) * 100.0),
            "cpu_utilization_pct_window": float((cpu_parallelism_window / cpu_logical_cores) * 100.0),
            "cpu_logical_cores": float(cpu_logical_cores),
            "torch_num_threads": float(max(0, int(self.torch_num_threads))),
            "torch_num_interop_threads": float(max(0, int(self.torch_num_interop_threads))),
            "seconds_policy_select_window": float(max(0.0, self.seconds_policy_select_window)),
            "seconds_env_step_window": float(max(0.0, self.seconds_env_step_window)),
            "seconds_buffer_add_window": float(max(0.0, self.seconds_buffer_add_window)),
            "seconds_ppo_update_window": float(max(0.0, self.seconds_ppo_update_window)),
            "seconds_dqn_update_window": float(max(0.0, self.seconds_dqn_update_window)),
            "seconds_metrics_window": float(max(0.0, self.seconds_metrics_window)),
            "seconds_checkpoint_window": float(max(0.0, self.seconds_checkpoint_window)),
            "steps_window": float(window_steps),
            "dqn_update_count_window": float(max(0, int(self.dqn_update_count_window))),
            "dqn_gradient_steps_window": float(max(0, int(self.dqn_gradient_steps_window))),
            "dqn_replay_sample_seconds_window": float(max(0.0, self.dqn_replay_sample_seconds_window)),
            "dqn_tensor_build_seconds_window": float(max(0.0, self.dqn_tensor_build_seconds_window)),
            "dqn_forward_seconds_window": float(max(0.0, self.dqn_forward_seconds_window)),
            "dqn_loss_seconds_window": float(max(0.0, self.dqn_loss_seconds_window)),
            "dqn_backward_seconds_window": float(max(0.0, self.dqn_backward_seconds_window)),
            "dqn_optimizer_seconds_window": float(max(0.0, self.dqn_optimizer_seconds_window)),
            "dqn_target_update_seconds_window": float(max(0.0, self.dqn_target_update_seconds_window)),
            "dqn_validation_seconds_window": float(max(0.0, self.dqn_validation_seconds_window)),
            "dqn_misc_seconds_window": float(max(0.0, self.dqn_misc_seconds_window)),
            "dqn_sample_count_window": float(dqn_sample_count),
            "dqn_forward_avg_ms_per_grad_step_window": _avg_ms(
                self.dqn_forward_seconds_window,
                denominator=dqn_avg_denominator,
            ),
            "dqn_backward_avg_ms_per_grad_step_window": _avg_ms(
                self.dqn_backward_seconds_window,
                denominator=dqn_avg_denominator,
            ),
            "dqn_optimizer_avg_ms_per_grad_step_window": _avg_ms(
                self.dqn_optimizer_seconds_window,
                denominator=dqn_avg_denominator,
            ),
            "dqn_validation_avg_ms_per_grad_step_window": _avg_ms(
                self.dqn_validation_seconds_window,
                denominator=dqn_avg_denominator,
            ),
            "dqn_total_avg_ms_per_grad_step_window": _avg_ms(
                self.seconds_dqn_update_window,
                denominator=dqn_avg_denominator,
            ),
            "dqn_samples_per_second_window": float(dqn_sample_count / dqn_update_seconds)
            if dqn_update_seconds > 0.0
            else 0.0,
            "dqn_gradient_steps_per_second_window": float(max(0, int(self.dqn_gradient_steps_window)) / dqn_update_seconds)
            if dqn_update_seconds > 0.0
            else 0.0,
            "dqn_replay_size": float(max(0, int(self.dqn_replay_size))),
            "dqn_batch_size": float(max(0, int(self.dqn_batch_size))),
            "dqn_gradient_steps_per_rollout_effective": float(
                max(0, int(self.dqn_gradient_steps_per_rollout_effective))
            ),
            "dqn_update_trigger_reason": str(self.dqn_update_trigger_reason),
            "teacher_retention_updates_window": float(teacher_updates),
            "teacher_dqn_kl_window": float(self.teacher_dqn_kl_sum_window / teacher_denominator)
            if teacher_denominator > 0.0
            else 0.0,
            "teacher_dqn_ce_window": float(self.teacher_dqn_ce_sum_window / teacher_denominator)
            if teacher_denominator > 0.0
            else 0.0,
            "teacher_dqn_action_match_rate_window": float(
                self.teacher_dqn_action_match_rate_sum_window / teacher_denominator
            )
            if teacher_denominator > 0.0
            else 0.0,
            "teacher_retention_loss_window": float(self.teacher_retention_loss_sum_window / teacher_denominator)
            if teacher_denominator > 0.0
            else 0.0,
            "gate_critical_action_drift_window": float(self.gate_critical_action_drift_sum_window / teacher_denominator)
            if teacher_denominator > 0.0
            else 0.0,
            "bad_pocket_margin_by_action_window": bad_pocket_margins,
            "teacher_bank_scenario_counts": dict(sorted(self.teacher_bank_scenario_counts.items())),
            "ppo_update_count_window": float(max(0, int(self.ppo_update_count_window))),
            "ppo_minibatch_steps_window": float(max(0, int(self.ppo_minibatch_steps_window))),
            "ppo_forward_backward_seconds_window": float(max(0.0, self.ppo_forward_backward_seconds_window)),
            "ppo_optimizer_seconds_window": float(max(0.0, self.ppo_optimizer_seconds_window)),
        }

    def reset_window(self, *, global_step: int, now: float | None = None, process_now: float | None = None) -> None:
        current = time.perf_counter() if now is None else float(now)
        current_process = time.process_time() if process_now is None else float(process_now)
        self.window_started_at = current
        self.process_window_started_at = current_process
        self.window_start_step = int(global_step)
        self.seconds_policy_select_window = 0.0
        self.seconds_env_step_window = 0.0
        self.seconds_buffer_add_window = 0.0
        self.seconds_ppo_update_window = 0.0
        self.seconds_dqn_update_window = 0.0
        self.seconds_metrics_window = 0.0
        self.seconds_checkpoint_window = 0.0
        self.dqn_update_count_window = 0
        self.dqn_gradient_steps_window = 0
        self.dqn_replay_sample_seconds_window = 0.0
        self.dqn_tensor_build_seconds_window = 0.0
        self.dqn_forward_seconds_window = 0.0
        self.dqn_loss_seconds_window = 0.0
        self.dqn_backward_seconds_window = 0.0
        self.dqn_optimizer_seconds_window = 0.0
        self.dqn_target_update_seconds_window = 0.0
        self.dqn_validation_seconds_window = 0.0
        self.dqn_misc_seconds_window = 0.0
        self.dqn_sample_count_window = 0
        self.dqn_replay_size = 0
        self.dqn_batch_size = 0
        self.dqn_gradient_steps_per_rollout_effective = 0
        self.dqn_update_trigger_reason = ""
        self.teacher_retention_updates_window = 0
        self.teacher_dqn_kl_sum_window = 0.0
        self.teacher_dqn_ce_sum_window = 0.0
        self.teacher_dqn_action_match_rate_sum_window = 0.0
        self.teacher_retention_loss_sum_window = 0.0
        self.gate_critical_action_drift_sum_window = 0.0
        self.bad_pocket_margin_sum_by_action_window.clear()
        self.bad_pocket_margin_count_by_action_window.clear()
        self.teacher_bank_scenario_counts.clear()
        self.ppo_update_count_window = 0
        self.ppo_minibatch_steps_window = 0
        self.ppo_forward_backward_seconds_window = 0.0
        self.ppo_optimizer_seconds_window = 0.0


def add_trace_logging_arguments(parser: argparse.ArgumentParser) -> None:
    """Register trace flags on the trainer CLI parser."""

    parser.add_argument(
        "--disable-trace-logging",
        action="store_true",
        help="Disable PostgreSQL EpisodeStore writes for joint MARL traces.",
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer.")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train synchronized PPO+DQN joint MARL policies with raw PyTorch.")
    parser.add_argument("--config", default="configs/training_joint.json", help="Path to unified joint training config.")
    parser.add_argument("--output-dir", default="models/checkpoints/joint_torch", help="Checkpoint/metrics directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or cuda:N.")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", default=None, help="Path to a joint_torch_latest.pt checkpoint to resume.")
    checkpoint_group.add_argument(
        "--init-from-joint-checkpoint",
        default=None,
        help="Load PPO/DQN weights from a joint checkpoint while starting a fresh fine-tune run.",
    )
    parser.add_argument(
        "--allow-empty-replay-resume",
        action="store_true",
        help=(
            "Permit --resume from a legacy checkpoint without DQN replay/RNG state. "
            "This is a warm-start diagnostic mode, not an exact continuation."
        ),
    )
    parser.add_argument("--final-ppo-path", default="models/checkpoints/ppo/ppo_torch_joint_final.pt")
    parser.add_argument("--final-dqn-path", default="models/checkpoints/dqn/dqn_torch_joint_final.pt")
    parser.add_argument("--skip-registry", action="store_true", help="Do not mark final checkpoints active.")
    parser.add_argument("--torch-num-threads", type=_positive_int, default=None, help="Override torch intra-op threads.")
    parser.add_argument(
        "--torch-num-interop-threads",
        type=_positive_int,
        default=None,
        help="Override torch inter-op threads.",
    )
    parser.add_argument(
        "--enable-teacher-retention",
        action="store_true",
        help="Enable explicit DQN teacher-retention behavior anchoring from configured teacher checkpoints.",
    )
    parser.add_argument(
        "--teacher-production-checkpoint",
        default=None,
        help="Read-only production teacher joint checkpoint for teacher retention.",
    )
    parser.add_argument(
        "--teacher-candidate-checkpoint",
        default=None,
        help="Read-only passing-candidate teacher joint checkpoint for teacher retention.",
    )
    add_trace_logging_arguments(parser)
    return parser.parse_args(argv)


def configure_torch_threading(
    *,
    num_threads: int | None,
    num_interop_threads: int | None,
    set_num_threads: Callable[[int], None] = torch.set_num_threads,
    set_num_interop_threads: Callable[[int], None] = torch.set_num_interop_threads,
    get_num_threads: Callable[[], int] = torch.get_num_threads,
    get_num_interop_threads: Callable[[], int] = torch.get_num_interop_threads,
) -> tuple[int, int]:
    """Apply optional Torch thread overrides and return actual runtime counts."""

    if num_threads is not None:
        if num_threads <= 0:
            raise ValueError("num_threads must be positive.")
        set_num_threads(int(num_threads))
    if num_interop_threads is not None:
        if num_interop_threads <= 0:
            raise ValueError("num_interop_threads must be positive.")
        set_num_interop_threads(int(num_interop_threads))
    return int(get_num_threads()), int(get_num_interop_threads())


def trace_logging_config_from_args(
    args: argparse.Namespace,
    *,
    flush_interval: int,
    sample_interval: int,
    environment_id: str,
    team_id: str,
) -> TraceLoggingConfig:
    """Build trace config from CLI args without opening a database connection."""

    return TraceLoggingConfig(
        enabled=not bool(getattr(args, "disable_trace_logging", False)),
        flush_interval=flush_interval,
        sample_interval=sample_interval,
        environment_id=environment_id,
        team_id=team_id,
    )


def configure_environment_info_mode_for_trace_logging(
    env_config: EnvironmentConfig,
    trace_cfg: TraceLoggingConfig,
) -> None:
    """Use compact step info only when trace snapshots are not being persisted."""

    env_config.info_mode = "full" if trace_cfg.enabled else "compact"


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_torch_threading(
        num_threads=getattr(args, "torch_num_threads", None),
        num_interop_threads=getattr(args, "torch_num_interop_threads", None),
    )
    config = load_config(Path(args.config))
    validate_active_mdp_contract(config)
    torch_cfg = _section(config, "torch_joint_training")
    shared_cfg = _section(config, "shared_global_parameters")
    ppo_cfg = _section(config, "ppo_hyperparameters")
    dqn_cfg = _section(config, "dqn_hyperparameters")

    seed = int(args.seed if args.seed is not None else shared_cfg.get("seed", 42))
    set_global_seed(seed)
    device = select_device(str(args.device if args.device != "auto" else torch_cfg.get("device", "auto")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "training_metrics.jsonl"

    total_timesteps = int(shared_cfg["total_timesteps"])
    max_steps = int(shared_cfg["max_steps"])
    reward_blend = build_reward_blend_config(torch_cfg)
    checkpoint_interval = int(torch_cfg["checkpoint_interval_steps"])
    metrics_interval = int(torch_cfg["metrics_interval_steps"])
    curriculum_config = curriculum_config_from_training_config(config, fallback_seed=seed)
    curriculum_sampler = CurriculumSampler(curriculum_config, total_timesteps=total_timesteps)

    trace_cfg = trace_logging_config_from_args(
        args,
        flush_interval=int(torch_cfg["trace_flush_interval"]),
        sample_interval=int(shared_cfg.get("trace_sample_interval", 1)),
        environment_id=str(shared_cfg.get("environment_id", "joint_rolling_5pl_288")),
        team_id=str(shared_cfg.get("team_id", "joint_from_scratch")),
    )
    base_env_config = build_environment_config(config, max_steps=max_steps, seed=seed)
    configure_environment_info_mode_for_trace_logging(base_env_config, trace_cfg)
    env = FivePLDigitalTwinEnv(replace(base_env_config))
    policies = JointPolicyBundle(dqn_architecture=_dqn_architecture_from_config(config)).to(device)
    ppo_optimizer = torch.optim.Adam(
        [
            *policies.ppo_feature_extractor.parameters(),
            *policies.ppo_actor.parameters(),
            *policies.ppo_critic.parameters(),
        ],
        lr=float(ppo_cfg["learning_rate"]),
    )
    dqn_optimizer = torch.optim.Adam(policies.dqn_q_network.parameters(), lr=float(dqn_cfg["learning_rate"]))
    ppo_rollout = PPORolloutBuffer(
        rollout_steps=int(torch_cfg["rollout_steps"]),
        device=device,
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        reward_blend=reward_blend,
    )
    dqn_replay = DQNReplayBuffer(
        capacity=int(torch_cfg["dqn_replay_buffer_size"]),
        learning_starts=int(torch_cfg["dqn_learning_starts"]),
        device=device,
        reward_blend=reward_blend,
    )
    hierarchical_init_config = hierarchical_dqn_initialization_config_from_training_config(config)
    teacher_retention_config = teacher_retention_config_from_training_config(config, args=args)
    teacher_retention_runtime = TeacherRetentionRuntime.from_config(teacher_retention_config, device=device)
    teacher_retention_state_bank = (
        TeacherRetentionStateBank(
            max_per_scenario=teacher_retention_config.state_bank_max_per_scenario,
            observation_dim=OBSERVATION_DIM,
        )
        if teacher_retention_config.enabled
        else None
    )
    metrics = JointMetricsAggregator()
    state = TrainingState()

    if args.resume:
        resume_metadata = load_training_checkpoint(
            Path(args.resume),
            policies=policies,
            ppo_optimizer=ppo_optimizer,
            dqn_optimizer=dqn_optimizer,
            state=state,
            dqn_replay=dqn_replay,
            device=device,
            allow_empty_replay_resume=bool(args.allow_empty_replay_resume),
            teacher_retention_state_bank=teacher_retention_state_bank,
            require_teacher_retention_state=teacher_retention_config.enabled,
        )
        validate_resume_metrics_checkpoint_alignment(metrics_path, checkpoint_global_step=state.global_step)
        print(
            f"resume={resume_metadata['resume_mode']} "
            f"replay_buffer={resume_metadata['replay_restore_status']} "
            f"replay_size={resume_metadata['dqn_replay_size']} "
            f"dqn_learning_starts={dqn_replay.learning_starts} "
            f"rng_state={resume_metadata['rng_restore_status']} "
            f"teacher_retention_state={resume_metadata['teacher_retention_state_status']} "
            "metrics_state=reinitialized",
            flush=True,
        )
    elif args.init_from_joint_checkpoint:
        init_checkpoint_path = Path(args.init_from_joint_checkpoint)
        if policies.dqn_architecture == HIERARCHICAL_DQN_ARCHITECTURE:
            if not hierarchical_init_config.enabled:
                raise ValueError(
                    "hierarchical_v1 init-from-joint-checkpoint requires explicit hierarchical_initialization."
                )
            validate_fine_tune_initial_checkpoint_path(init_checkpoint_path)
            if hierarchical_init_config.teacher_checkpoint is not None:
                expected_teacher = _normalize_checkpoint_guard_path(hierarchical_init_config.teacher_checkpoint)
                actual_teacher = _normalize_checkpoint_guard_path(init_checkpoint_path)
                if expected_teacher != actual_teacher:
                    raise ValueError(
                        "hierarchical_initialization.teacher_checkpoint must match --init-from-joint-checkpoint."
                    )
            hierarchical_init_config = hierarchical_init_config.with_teacher_checkpoint(init_checkpoint_path)
            init_metadata = load_hierarchical_distillation_initial_checkpoint(
                init_checkpoint_path,
                policies=policies,
                state=state,
                device=device,
                init_config=hierarchical_init_config,
            )
        else:
            init_metadata = load_fine_tune_initial_checkpoint(
                init_checkpoint_path,
                policies=policies,
                state=state,
                device=device,
            )
        existing_init = config.get(FINE_TUNE_INITIALIZATION_KEY, {})
        if not isinstance(existing_init, Mapping):
            existing_init = {}
        config[FINE_TUNE_INITIALIZATION_KEY] = {
            **dict(existing_init),
            **init_metadata,
        }
        print(
            "init_from_joint_checkpoint=true replay_buffer=empty optimizers=fresh "
            f"global_step=0 parent_global_step={init_metadata['initialized_from_global_step']} "
            f"parent_env_id={init_metadata['initialized_from_env_id']} "
            "metrics_state=fresh",
            flush=True,
        )
    else:
        print(
            f"global_step=0 mdp_contract_version={MDP_CONTRACT_VERSION} "
            "resume=none replay_buffer=empty optimizers=fresh",
            flush=True,
        )
    if curriculum_config.enabled:
        print(
            "curriculum_enabled=true "
            f"curriculum_stage={curriculum_config.stage} "
            f"curriculum_seed={curriculum_config.seed} "
            f"stage_progression={curriculum_sampler.stage_progression}",
            flush=True,
        )
    training_start_step = int(state.global_step)

    db_pool: DatabasePool | None = None
    trace_recorder: JointTraceRecorder
    if trace_cfg.enabled:
        db_pool = DatabasePool()
        await db_pool.startup()
        trace_recorder = JointTraceRecorder.enabled(EpisodeStore(db_pool), config=trace_cfg)
    else:
        trace_recorder = JointTraceRecorder.disabled(config=trace_cfg)

    try:
        await train_loop(
            env=env,
            policies=policies,
            ppo_optimizer=ppo_optimizer,
            dqn_optimizer=dqn_optimizer,
            ppo_rollout=ppo_rollout,
            dqn_replay=dqn_replay,
            metrics=metrics,
            teacher_retention_config=teacher_retention_config,
            teacher_retention_runtime=teacher_retention_runtime,
            teacher_retention_state_bank=teacher_retention_state_bank,
            trace_recorder=trace_recorder,
            state=state,
            config=config,
            output_dir=output_dir,
            metrics_path=metrics_path,
            base_env_config=base_env_config,
            curriculum_sampler=curriculum_sampler,
            total_timesteps=total_timesteps,
            seed=seed,
            reward_blend=reward_blend,
            ppo_epochs=int(torch_cfg["ppo_epochs"]),
            ppo_minibatch_size=int(torch_cfg["ppo_minibatch_size"]),
            ppo_clip_range=float(torch_cfg["ppo_clip_range"]),
            ppo_value_coef=float(torch_cfg["ppo_value_coef"]),
            ppo_entropy_coef=float(torch_cfg["ppo_entropy_coef"]),
            ppo_max_grad_norm=float(torch_cfg["ppo_max_grad_norm"]),
            dqn_batch_size=int(torch_cfg["dqn_batch_size"]),
            dqn_gradient_steps_per_rollout=int(torch_cfg["dqn_gradient_steps_per_rollout"]),
            dqn_target_update_interval=int(torch_cfg["dqn_target_update_interval"]),
            dqn_tau=float(torch_cfg["dqn_tau"]),
            dqn_gamma=float(dqn_cfg["gamma"]),
            exploration_initial=float(torch_cfg["dqn_exploration_initial_eps"]),
            exploration_final=float(torch_cfg["dqn_exploration_final_eps"]),
            exploration_fraction=float(torch_cfg["dqn_exploration_fraction"]),
            checkpoint_interval=checkpoint_interval,
            metrics_interval=metrics_interval,
        )
        await trace_recorder.flush()
        _save_final_artifacts_if_training_advanced(
            start_step=training_start_step,
            state=state,
            save=lambda: save_final_artifacts(
                policies=policies,
                ppo_optimizer=ppo_optimizer,
                dqn_optimizer=dqn_optimizer,
                state=state,
                config=config,
                metrics=metrics,
                dqn_replay=dqn_replay,
                teacher_retention_state_bank=teacher_retention_state_bank,
                output_dir=output_dir,
                final_ppo_path=Path(args.final_ppo_path),
                final_dqn_path=Path(args.final_dqn_path),
                register_active=not bool(args.skip_registry),
            ),
        )
    finally:
        await trace_recorder.flush()
        if db_pool is not None:
            await db_pool.shutdown()
        env.close()


async def train_loop(
    *,
    env: FivePLDigitalTwinEnv,
    policies: JointPolicyBundle,
    ppo_optimizer: torch.optim.Optimizer,
    dqn_optimizer: torch.optim.Optimizer,
    ppo_rollout: PPORolloutBuffer,
    dqn_replay: DQNReplayBuffer,
    metrics: JointMetricsAggregator,
    teacher_retention_config: TeacherRetentionConfig,
    teacher_retention_runtime: TeacherRetentionRuntime | None,
    teacher_retention_state_bank: TeacherRetentionStateBank | None,
    trace_recorder: JointTraceRecorder,
    state: TrainingState,
    config: dict[str, Any],
    output_dir: Path,
    metrics_path: Path,
    base_env_config: EnvironmentConfig,
    curriculum_sampler: CurriculumSampler,
    total_timesteps: int,
    seed: int,
    reward_blend: JointRewardBlendConfig,
    ppo_epochs: int,
    ppo_minibatch_size: int,
    ppo_clip_range: float,
    ppo_value_coef: float,
    ppo_entropy_coef: float,
    ppo_max_grad_norm: float,
    dqn_batch_size: int,
    dqn_gradient_steps_per_rollout: int,
    dqn_target_update_interval: int,
    dqn_tau: float,
    dqn_gamma: float,
    exploration_initial: float,
    exploration_final: float,
    exploration_fraction: float,
    checkpoint_interval: int,
    metrics_interval: int,
) -> None:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive.")

    loop_start_step = int(state.global_step)
    observation_np, curriculum_sample = reset_environment_for_episode(
        env=env,
        base_env_config=base_env_config,
        curriculum_sampler=curriculum_sampler,
        seed=seed + state.episode_count,
        episode_index=state.episode_count,
        global_step=state.global_step,
    )
    last_rollout_next_observation_np = observation_np
    last_rollout_terminated = False
    episode_id = uuid4()
    step_in_episode = 0
    next_checkpoint_step = _next_interval(state.global_step, checkpoint_interval)
    next_metrics_step = _next_interval(state.global_step, metrics_interval)
    timing_started = time.perf_counter()
    process_timing_started = time.process_time()
    timing = TrainingTimingWindow(
        started_at=timing_started,
        window_started_at=timing_started,
        process_started_at=process_timing_started,
        process_window_started_at=process_timing_started,
        start_step=state.global_step,
        window_start_step=state.global_step,
    )

    while state.global_step < total_timesteps:
        policy_started = time.perf_counter()
        observation = _observation_tensor(observation_np, device=next(policies.parameters()).device)
        epsilon = exploration_epsilon(
            state.global_step,
            total_timesteps=total_timesteps,
            initial=exploration_initial,
            final=exploration_final,
            fraction=exploration_fraction,
        )
        with torch.no_grad():
            ppo_output = policies.select_ppo_action(observation)
            dqn_output = policies.select_dqn_action(observation, epsilon=epsilon)

        continuous_action = ppo_output.action.squeeze(0).detach().cpu()
        discrete_action = dqn_output.action.squeeze(0).detach().cpu()
        timing.seconds_policy_select_window += time.perf_counter() - policy_started

        env_step_started = time.perf_counter()
        next_observation_np, reward, terminated, truncated, info = env.step(
            {
                "continuous": continuous_action.numpy(),
                "discrete": int(discrete_action.item()),
            }
        )
        timing.seconds_env_step_window += time.perf_counter() - env_step_started
        reward_components = _reward_components(info, fallback_total=float(reward))
        attach_curriculum_telemetry(info, curriculum_sample)
        last_rollout_next_observation_np = next_observation_np
        last_rollout_terminated = bool(terminated)
        buffer_started = time.perf_counter()
        transition = JointTransition.from_reward_components(
            observation=observation.squeeze(0).detach().cpu(),
            ppo_action=continuous_action,
            ppo_log_prob=ppo_output.log_prob.squeeze(0).detach().cpu(),
            ppo_value=ppo_output.value.squeeze(0).detach().cpu(),
            dqn_action=discrete_action,
            reward_total=float(reward),
            reward_global=reward_components["global"],
            reward_ppo_local=reward_components["ppo_local"],
            reward_dqn_local=reward_components["dqn_local"],
            reward_blend=reward_blend,
            next_observation=torch.as_tensor(next_observation_np, dtype=torch.float32).detach().cpu(),
            terminated=bool(terminated),
            truncated=bool(truncated),
            projected=bool(info.get("projected", False)),
            blocked=bool(info.get("blocked", False)),
            info=dict(info),
            joint_action_id=str(uuid4()),
            episode_id=str(episode_id),
            step_id=step_in_episode,
        )

        ppo_rollout.add(transition)
        dqn_replay.add(transition)
        if teacher_retention_state_bank is not None:
            teacher_retention_state_bank.record(transition.observation, curriculum_sample.effective_stage)
        metrics.record_transition(transition)
        if trace_recorder.should_record_step(state.global_step):
            await trace_recorder.record_transition(transition)
        timing.seconds_buffer_add_window += time.perf_counter() - buffer_started

        state.global_step += 1
        step_in_episode += 1

        if ppo_rollout.is_full:
            ppo_update_started = time.perf_counter()
            state.last_ppo_loss = update_ppo(
                policies=policies,
                optimizer=ppo_optimizer,
                rollout=ppo_rollout,
                next_observation=next_observation_np,
                terminated=bool(terminated),
                epochs=ppo_epochs,
                minibatch_size=ppo_minibatch_size,
                clip_range=ppo_clip_range,
                value_coef=ppo_value_coef,
                entropy_coef=ppo_entropy_coef,
                max_grad_norm=ppo_max_grad_norm,
                timing_window=timing,
            )
            timing.seconds_ppo_update_window += time.perf_counter() - ppo_update_started
            ppo_rollout.reset()
            if dqn_replay.can_sample:
                state.last_dqn_loss = update_dqn(
                    policies=policies,
                    optimizer=dqn_optimizer,
                    replay=dqn_replay,
                    batch_size=min(dqn_batch_size, len(dqn_replay)),
                    gradient_steps=dqn_gradient_steps_per_rollout,
                    gamma=dqn_gamma,
                    target_update_interval=dqn_target_update_interval,
                    tau=dqn_tau,
                    state=state,
                    timing_window=timing,
                    trigger_reason="rollout_full",
                    teacher_retention_runtime=teacher_retention_runtime,
                    teacher_retention_state_bank=teacher_retention_state_bank,
                    teacher_retention_batch_size=teacher_retention_config.state_bank_batch_size,
                )

        if _should_save_periodic_checkpoint(
            global_step=state.global_step,
            next_checkpoint_step=next_checkpoint_step,
            total_timesteps=total_timesteps,
        ):
            checkpoint_started = time.perf_counter()
            save_periodic_checkpoint(
                policies=policies,
                ppo_optimizer=ppo_optimizer,
                dqn_optimizer=dqn_optimizer,
                state=state,
                config=config,
                dqn_replay=dqn_replay,
                teacher_retention_state_bank=teacher_retention_state_bank,
                output_dir=output_dir,
            )
            timing.seconds_checkpoint_window += time.perf_counter() - checkpoint_started
            next_checkpoint_step += checkpoint_interval

        if state.global_step >= next_metrics_step and state.global_step < total_timesteps:
            snapshot = append_training_metrics_jsonl(
                metrics,
                metrics_path,
                global_step=state.global_step,
                curriculum_sample=curriculum_sample,
                timing_window=timing,
            )
            print(metrics.format_progress(global_step=state.global_step), flush=True)
            _validate_snapshot(snapshot.as_json_dict())
            timing.reset_window(global_step=state.global_step)
            next_metrics_step += metrics_interval

        done = bool(terminated or truncated)
        if done:
            state.episode_count += 1
            observation_np, curriculum_sample = reset_environment_for_episode(
                env=env,
                base_env_config=base_env_config,
                curriculum_sampler=curriculum_sampler,
                seed=seed + state.episode_count,
                episode_index=state.episode_count,
                global_step=state.global_step,
            )
            episode_id = uuid4()
            step_in_episode = 0
        else:
            observation_np = next_observation_np

    if len(ppo_rollout) > 0:
        ppo_update_started = time.perf_counter()
        state.last_ppo_loss = update_ppo(
            policies=policies,
            optimizer=ppo_optimizer,
            rollout=ppo_rollout,
            next_observation=last_rollout_next_observation_np,
            terminated=last_rollout_terminated,
            epochs=ppo_epochs,
            minibatch_size=ppo_minibatch_size,
            clip_range=ppo_clip_range,
            value_coef=ppo_value_coef,
            entropy_coef=ppo_entropy_coef,
            max_grad_norm=ppo_max_grad_norm,
            timing_window=timing,
        )
        timing.seconds_ppo_update_window += time.perf_counter() - ppo_update_started
        ppo_rollout.reset()
    if dqn_replay.can_sample:
        state.last_dqn_loss = update_dqn(
            policies=policies,
            optimizer=dqn_optimizer,
            replay=dqn_replay,
            batch_size=min(dqn_batch_size, len(dqn_replay)),
            gradient_steps=max(1, min(dqn_gradient_steps_per_rollout, len(dqn_replay))),
            gamma=dqn_gamma,
            target_update_interval=dqn_target_update_interval,
            tau=dqn_tau,
            state=state,
            timing_window=timing,
            trigger_reason="final_flush",
            teacher_retention_runtime=teacher_retention_runtime,
            teacher_retention_state_bank=teacher_retention_state_bank,
            teacher_retention_batch_size=teacher_retention_config.state_bank_batch_size,
        )
    if _should_write_final_training_outputs(start_step=loop_start_step, global_step=state.global_step):
        checkpoint_started = time.perf_counter()
        save_periodic_checkpoint(
            policies=policies,
            ppo_optimizer=ppo_optimizer,
            dqn_optimizer=dqn_optimizer,
            state=state,
            config=config,
            dqn_replay=dqn_replay,
            teacher_retention_state_bank=teacher_retention_state_bank,
            output_dir=output_dir,
        )
        timing.seconds_checkpoint_window += time.perf_counter() - checkpoint_started
        append_training_metrics_jsonl(
            metrics,
            metrics_path,
            global_step=state.global_step,
            curriculum_sample=curriculum_sample,
            timing_window=timing,
        )


def reset_environment_for_episode(
    *,
    env: FivePLDigitalTwinEnv,
    base_env_config: EnvironmentConfig,
    curriculum_sampler: CurriculumSampler,
    seed: int,
    episode_index: int,
    global_step: int | None = None,
) -> tuple[np.ndarray, CurriculumSample]:
    """Reset one training episode and apply deterministic curriculum overrides if enabled."""

    sample = curriculum_sampler.sample(episode_index, global_step=global_step)
    env.config = replace(base_env_config)
    env.config.random_seed = int(seed)
    config_matrix = {}
    if sample.enabled:
        config_matrix = apply_config_overrides(env.config, sample.environment_overrides)

    env.scenario_stress_state = env._default_scenario_stress_state()
    observation, _info = env.reset(seed=seed)
    if sample.enabled:
        env_matrix = apply_environment_overrides(env, sample.environment_overrides, config_matrix)
        sample = sample.with_capability_matrix(capability_matrix_as_dict(env_matrix))
        env._previous_snapshot = env.simulation.snapshot()
        observation = env._observation(snapshot=env._previous_snapshot)
    return observation, sample


def attach_curriculum_telemetry(info: dict[str, Any], sample: CurriculumSample) -> None:
    """Attach compact curriculum metadata to transition info for traces and audits."""

    if not sample.enabled:
        return
    info["curriculum_enabled"] = True
    info["curriculum_stage"] = sample.stage
    info["curriculum_effective_stage"] = sample.effective_stage
    info["curriculum_seed"] = sample.seed
    info["curriculum_episode_index"] = sample.episode_index
    info["baseline_anchor_used"] = sample.baseline_anchor_used
    info["sampled_stress_overrides"] = dict(sorted(sample.environment_overrides.items()))
    info["curriculum"] = sample.as_trace_info()


def append_training_metrics_jsonl(
    metrics: JointMetricsAggregator,
    path: str | Path,
    *,
    global_step: int | None = None,
    curriculum_sample: CurriculumSample | None = None,
    timing_fields: Mapping[str, float | str] | None = None,
    timing_window: TrainingTimingWindow | None = None,
):
    """Write training metrics with optional curriculum fields without changing disabled runs."""

    metrics_started = time.perf_counter() if timing_window is not None else None
    snapshot = metrics.snapshot(global_step=global_step)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = snapshot.as_json_dict()
    if curriculum_sample is not None and curriculum_sample.enabled:
        row.update(curriculum_sample.as_metrics_fields())
    if timing_window is not None and metrics_started is not None:
        timing_window.seconds_metrics_window += time.perf_counter() - metrics_started
        row.update(timing_window.as_metrics_fields(global_step=row["global_step"]))
    if timing_fields is not None:
        row.update(
            {
                key: value if isinstance(value, (str, Mapping, list, tuple)) else float(value)
                for key, value in timing_fields.items()
            }
        )
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
        handle.write("\n")
    return snapshot


def validate_resume_metrics_checkpoint_alignment(path: str | Path, *, checkpoint_global_step: int) -> None:
    """Reject silent same-output resumes when metrics have advanced past the checkpoint."""

    if checkpoint_global_step < 0:
        raise ValueError("checkpoint_global_step must not be negative.")
    output_path = Path(path)
    if not output_path.exists():
        return

    max_metrics_step: int | None = None
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"training metrics file {output_path} contains invalid JSON on line {line_number}."
                ) from exc
            if not isinstance(row, Mapping) or "global_step" not in row:
                continue
            metric_step = int(row["global_step"])
            max_metrics_step = metric_step if max_metrics_step is None else max(max_metrics_step, metric_step)

    if max_metrics_step is not None and max_metrics_step > checkpoint_global_step:
        raise ValueError(
            "training metrics are ahead of checkpoint; refusing silent same-output resume "
            f"(metrics global_step {max_metrics_step}, checkpoint global_step {checkpoint_global_step})."
        )


def update_ppo(
    *,
    policies: JointPolicyBundle,
    optimizer: torch.optim.Optimizer,
    rollout: PPORolloutBuffer,
    next_observation: np.ndarray,
    terminated: bool,
    epochs: int,
    minibatch_size: int,
    clip_range: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    timing_window: TrainingTimingWindow | None = None,
) -> float:
    device = next(policies.parameters()).device
    with torch.no_grad():
        next_value = policies.select_ppo_action(_observation_tensor(next_observation, device=device), deterministic=True).value.squeeze(0)
    rollout.compute_returns_and_advantages(last_value=next_value, last_terminated=terminated)

    if timing_window is not None:
        timing_window.ppo_update_count_window += 1
    losses: list[float] = []
    for _ in range(epochs):
        for batch in rollout.iter_minibatches(minibatch_size=minibatch_size):
            forward_backward_started = time.perf_counter()
            new_log_probs, entropy, values = policies.evaluate_ppo_actions(batch.observations, batch.actions)
            ratio = torch.exp(new_log_probs - batch.old_log_probs)
            policy_loss_1 = batch.advantages * ratio
            policy_loss_2 = batch.advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            value_loss = F.mse_loss(values, batch.returns)
            entropy_loss = -entropy.mean()
            loss = policy_loss + (value_coef * value_loss) + (entropy_coef * entropy_loss)
            _ensure_finite(loss, "ppo_loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [
                    *policies.ppo_feature_extractor.parameters(),
                    *policies.ppo_actor.parameters(),
                    *policies.ppo_critic.parameters(),
                ],
                max_norm=max_grad_norm,
                error_if_nonfinite=True,
            )
            _add_timing_attr(
                timing_window,
                "ppo_forward_backward_seconds_window",
                time.perf_counter() - forward_backward_started,
            )
            optimizer_started = time.perf_counter()
            optimizer.step()
            _add_timing_attr(timing_window, "ppo_optimizer_seconds_window", time.perf_counter() - optimizer_started)
            if timing_window is not None:
                timing_window.ppo_minibatch_steps_window += 1
            losses.append(float(loss.detach().cpu().item()))
    return sum(losses) / max(len(losses), 1)


def update_dqn(
    *,
    policies: JointPolicyBundle,
    optimizer: torch.optim.Optimizer,
    replay: DQNReplayBuffer,
    batch_size: int,
    gradient_steps: int,
    gamma: float,
    target_update_interval: int,
    tau: float,
    state: TrainingState,
    timing_window: TrainingTimingWindow | None = None,
    trigger_reason: str = "",
    teacher_retention_runtime: TeacherRetentionRuntime | None = None,
    teacher_retention_state_bank: TeacherRetentionStateBank | None = None,
    teacher_retention_batch_size: int = 0,
) -> float:
    update_started = time.perf_counter()
    component_seconds_before = _dqn_component_seconds(timing_window)
    if timing_window is not None:
        timing_window.dqn_update_count_window += 1
        timing_window.dqn_replay_size = len(replay)
        timing_window.dqn_batch_size = int(batch_size)
        timing_window.dqn_gradient_steps_per_rollout_effective = int(gradient_steps)
        timing_window.dqn_update_trigger_reason = trigger_reason

    losses: list[float] = []
    try:
        for _ in range(gradient_steps):
            batch_timing: dict[str, float] | None = {} if timing_window is not None else None
            batch = replay.sample(batch_size, timing=batch_timing)
            if timing_window is not None and batch_timing is not None:
                _merge_replay_timing(timing_window, batch_timing)

            loss = dqn_loss(policies=policies, batch=batch, gamma=gamma, timing_window=timing_window)
            if (
                teacher_retention_runtime is not None
                and teacher_retention_state_bank is not None
                and teacher_retention_state_bank.can_sample
                and teacher_retention_batch_size > 0
            ):
                retention_batch = teacher_retention_state_bank.sample(
                    batch_size=int(teacher_retention_batch_size),
                    device=batch.observations.device,
                )
                retention = teacher_retention_runtime.loss_for_batch(
                    student_policy=policies,
                    batch=retention_batch,
                )
                _ensure_finite(retention.loss, "teacher_retention_loss")
                loss = loss + retention.loss
                record_teacher_retention_metrics(timing_window, retention.metrics)

            validation_started = time.perf_counter()
            _ensure_finite(loss, "dqn_loss")
            _add_timing_attr(timing_window, "dqn_validation_seconds_window", time.perf_counter() - validation_started)

            backward_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policies.dqn_q_network.parameters(), max_norm=10.0, error_if_nonfinite=True)
            _add_timing_attr(timing_window, "dqn_backward_seconds_window", time.perf_counter() - backward_started)

            optimizer_started = time.perf_counter()
            optimizer.step()
            _add_timing_attr(timing_window, "dqn_optimizer_seconds_window", time.perf_counter() - optimizer_started)

            state.dqn_update_count += 1
            if timing_window is not None:
                timing_window.dqn_gradient_steps_window += 1
                timing_window.dqn_sample_count_window += int(batch_size)
            if tau < 1.0:
                target_update_started = time.perf_counter()
                policies.soft_update_dqn_target(tau)
                _add_timing_attr(
                    timing_window,
                    "dqn_target_update_seconds_window",
                    time.perf_counter() - target_update_started,
                )
            elif target_update_interval > 0 and state.dqn_update_count % target_update_interval == 0:
                target_update_started = time.perf_counter()
                policies.hard_update_dqn_target()
                _add_timing_attr(
                    timing_window,
                    "dqn_target_update_seconds_window",
                    time.perf_counter() - target_update_started,
                )
            losses.append(float(loss.detach().cpu().item()))
    finally:
        if timing_window is not None:
            update_elapsed = time.perf_counter() - update_started
            timing_window.seconds_dqn_update_window += max(0.0, update_elapsed)
            component_elapsed = max(0.0, _dqn_component_seconds(timing_window) - component_seconds_before)
            timing_window.dqn_misc_seconds_window += max(0.0, update_elapsed - component_elapsed)
    return sum(losses) / max(len(losses), 1)


def dqn_loss(
    *,
    policies: JointPolicyBundle,
    batch: DQNReplayBatch,
    gamma: float,
    timing_window: TrainingTimingWindow | None = None,
) -> Tensor:
    forward_started = time.perf_counter()
    q_values = policies.dqn_q_network(batch.observations)
    with torch.no_grad():
        next_q = policies.dqn_target_q_network(batch.next_observations).max(dim=1).values
    _add_timing_attr(timing_window, "dqn_forward_seconds_window", time.perf_counter() - forward_started)

    loss_started = time.perf_counter()
    current_q = q_values.gather(1, batch.actions.view(-1, 1)).squeeze(1)
    with torch.no_grad():
        not_done = (~batch.dones).to(dtype=torch.float32)
        target_q = batch.rewards + (gamma * next_q * not_done)
    loss = F.smooth_l1_loss(current_q, target_q)
    _add_timing_attr(timing_window, "dqn_loss_seconds_window", time.perf_counter() - loss_started)
    return loss


def _add_timing_attr(timing_window: TrainingTimingWindow | None, field_name: str, elapsed: float) -> None:
    if timing_window is None:
        return
    current = getattr(timing_window, field_name)
    setattr(timing_window, field_name, float(current) + max(0.0, float(elapsed)))


def _avg_ms(seconds: float, *, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float((max(0.0, float(seconds)) / denominator) * 1000.0)


def _merge_replay_timing(timing_window: TrainingTimingWindow, timing: Mapping[str, float]) -> None:
    for field_name in (
        "dqn_replay_sample_seconds_window",
        "dqn_tensor_build_seconds_window",
        "dqn_validation_seconds_window",
    ):
        if field_name in timing:
            _add_timing_attr(timing_window, field_name, float(timing[field_name]))


def _dqn_component_seconds(timing_window: TrainingTimingWindow | None) -> float:
    if timing_window is None:
        return 0.0
    return sum(
        max(0.0, float(getattr(timing_window, field_name)))
        for field_name in (
            "dqn_replay_sample_seconds_window",
            "dqn_tensor_build_seconds_window",
            "dqn_forward_seconds_window",
            "dqn_loss_seconds_window",
            "dqn_backward_seconds_window",
            "dqn_optimizer_seconds_window",
            "dqn_target_update_seconds_window",
            "dqn_validation_seconds_window",
        )
    )


def save_periodic_checkpoint(
    *,
    policies: JointPolicyBundle,
    ppo_optimizer: torch.optim.Optimizer,
    dqn_optimizer: torch.optim.Optimizer,
    state: TrainingState,
    config: dict[str, Any],
    dqn_replay: DQNReplayBuffer,
    teacher_retention_state_bank: TeacherRetentionStateBank | None = None,
    output_dir: Path,
) -> None:
    rng_state = capture_rng_state()
    eval_checkpoint = build_checkpoint_payload(
        policies=policies,
        ppo_optimizer=ppo_optimizer,
        dqn_optimizer=dqn_optimizer,
        state=state,
        config=config,
        include_replay_state=False,
        include_rng_state=False,
    )
    exact_checkpoint = build_checkpoint_payload(
        policies=policies,
        ppo_optimizer=ppo_optimizer,
        dqn_optimizer=dqn_optimizer,
        state=state,
        config=config,
        dqn_replay=dqn_replay,
        teacher_retention_state_bank=teacher_retention_state_bank,
        rng_state=rng_state,
    )
    _atomic_torch_save({**eval_checkpoint, "artifact_kind": "ppo"}, output_dir / f"ppo_torch_joint_{state.global_step}.pt")
    _atomic_torch_save({**eval_checkpoint, "artifact_kind": "dqn"}, output_dir / f"dqn_torch_joint_{state.global_step}.pt")
    _atomic_torch_save({**exact_checkpoint, "artifact_kind": "joint"}, output_dir / "joint_torch_latest.pt")


def save_final_artifacts(
    *,
    policies: JointPolicyBundle,
    ppo_optimizer: torch.optim.Optimizer,
    dqn_optimizer: torch.optim.Optimizer,
    state: TrainingState,
    config: dict[str, Any],
    metrics: JointMetricsAggregator,
    dqn_replay: DQNReplayBuffer,
    teacher_retention_state_bank: TeacherRetentionStateBank | None = None,
    output_dir: Path,
    final_ppo_path: Path = Path("models/checkpoints/ppo/ppo_torch_joint_final.pt"),
    final_dqn_path: Path = Path("models/checkpoints/dqn/dqn_torch_joint_final.pt"),
    register_active: bool = True,
) -> None:
    final_metrics = metrics.snapshot(global_step=state.global_step).as_json_dict()
    rng_state = capture_rng_state()
    eval_checkpoint = build_checkpoint_payload(
        policies=policies,
        ppo_optimizer=ppo_optimizer,
        dqn_optimizer=dqn_optimizer,
        state=state,
        config=config,
        extra={"final_metrics": final_metrics},
        include_replay_state=False,
        include_rng_state=False,
    )
    exact_checkpoint = build_checkpoint_payload(
        policies=policies,
        ppo_optimizer=ppo_optimizer,
        dqn_optimizer=dqn_optimizer,
        state=state,
        config=config,
        extra={"final_metrics": final_metrics},
        dqn_replay=dqn_replay,
        teacher_retention_state_bank=teacher_retention_state_bank,
        rng_state=rng_state,
    )
    ppo_final = final_ppo_path
    dqn_final = final_dqn_path
    ppo_final.parent.mkdir(parents=True, exist_ok=True)
    dqn_final.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save({**eval_checkpoint, "artifact_kind": "ppo_final"}, ppo_final)
    _atomic_torch_save({**eval_checkpoint, "artifact_kind": "dqn_final"}, dqn_final)
    _atomic_torch_save({**exact_checkpoint, "artifact_kind": "joint_final"}, output_dir / "joint_torch_latest.pt")

    snapshot = metrics.snapshot(global_step=state.global_step)
    if register_active:
        registry = ModelRegistry()
        registry.register(
            algorithm="ppo",
            path=ppo_final,
            agent_role=PPO_AGENT_ROLE,
            metrics={"mean_reward": snapshot.mean_ppo_local_reward, "service_level": snapshot.service_level},
            metadata={"policy_id": PPO_POLICY_ID, "global_step": state.global_step},
            status="active",
        )
        registry.register(
            algorithm="dqn",
            path=dqn_final,
            agent_role=DQN_AGENT_ROLE,
            metrics={"mean_reward": snapshot.mean_dqn_local_reward, "service_level": snapshot.service_level},
            metadata={"policy_id": DQN_POLICY_ID, "global_step": state.global_step},
            status="active",
        )


def build_checkpoint_payload(
    *,
    policies: JointPolicyBundle,
    ppo_optimizer: torch.optim.Optimizer,
    dqn_optimizer: torch.optim.Optimizer,
    state: TrainingState,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
    dqn_replay: DQNReplayBuffer | None = None,
    teacher_retention_state_bank: TeacherRetentionStateBank | None = None,
    rng_state: Mapping[str, Any] | None = None,
    include_replay_state: bool = True,
    include_rng_state: bool = True,
) -> dict[str, Any]:
    payload = policies.build_checkpoint(
        global_step=state.global_step,
        config=config,
        ppo_optimizer_state=ppo_optimizer.state_dict(),
        dqn_optimizer_state=dqn_optimizer.state_dict(),
        extra={
            "episode_count": state.episode_count,
            "dqn_update_count": state.dqn_update_count,
            "last_ppo_loss": state.last_ppo_loss,
            "last_dqn_loss": state.last_dqn_loss,
            **(extra or {}),
        },
    )
    replay_state: dict[str, Any] | None = None
    if include_replay_state and dqn_replay is not None:
        replay_state = dqn_replay.state_dict()
        payload[DQN_REPLAY_STATE_KEY] = replay_state
    if include_replay_state and teacher_retention_state_bank is not None:
        payload[TEACHER_RETENTION_STATE_BANK_KEY] = teacher_retention_state_bank.state_dict()
    rng_snapshot: Mapping[str, Any] | None = None
    if include_rng_state:
        rng_snapshot = rng_state if rng_state is not None else capture_rng_state()
        payload[RNG_STATE_KEY] = dict(rng_snapshot)
    payload[RESUME_STATE_METADATA_KEY] = _build_resume_state_metadata(
        state=state,
        dqn_replay=dqn_replay,
        replay_state_included=replay_state is not None,
        rng_state_included=rng_snapshot is not None,
        teacher_retention_state_bank=teacher_retention_state_bank,
    )
    init_metadata = config.get(FINE_TUNE_INITIALIZATION_KEY)
    if isinstance(init_metadata, Mapping):
        payload[FINE_TUNE_INITIALIZATION_KEY] = dict(init_metadata)
        for key in FINE_TUNE_INIT_METADATA_KEYS:
            if key in init_metadata:
                payload[key] = init_metadata[key]
    return payload


def _build_resume_state_metadata(
    *,
    state: TrainingState,
    dqn_replay: DQNReplayBuffer | None,
    replay_state_included: bool,
    rng_state_included: bool,
    teacher_retention_state_bank: TeacherRetentionStateBank | None = None,
) -> dict[str, Any]:
    replay_summary = dqn_replay.state_summary() if dqn_replay is not None else {}
    replay_size = int(replay_summary.get("size", 0))
    replay_total_added = int(replay_summary.get("total_added", 0))
    replay_capacity = int(replay_summary.get("capacity", 0))
    replay_position = int(replay_summary.get("position", 0))
    return {
        "resume_state_version": RESUME_STATE_VERSION,
        "global_step": int(state.global_step),
        "episode_count": int(state.episode_count),
        "dqn_update_count": int(state.dqn_update_count),
        "dqn_replay_buffer_included": bool(replay_state_included),
        "dqn_replay_size": replay_size if replay_state_included else 0,
        "dqn_replay_total_added": replay_total_added if replay_state_included else 0,
        "dqn_replay_capacity": replay_capacity if replay_state_included else 0,
        "dqn_replay_position": replay_position if replay_state_included else 0,
        "dqn_replay_learning_starts": int(replay_summary.get("learning_starts", 0)) if replay_state_included else 0,
        "teacher_retention_state_bank_included": bool(
            replay_state_included and teacher_retention_state_bank is not None
        ),
        "teacher_retention_state_bank_size": int(teacher_retention_state_bank.total_size)
        if replay_state_included and teacher_retention_state_bank is not None
        else 0,
        "teacher_retention_state_bank_scenario_counts": teacher_retention_state_bank.scenario_counts()
        if replay_state_included and teacher_retention_state_bank is not None
        else {},
        "rng_state_included": bool(rng_state_included),
        "exact_resume_capable": bool(replay_state_included and rng_state_included),
    }


def load_training_checkpoint(
    path: Path,
    *,
    policies: JointPolicyBundle,
    ppo_optimizer: torch.optim.Optimizer,
    dqn_optimizer: torch.optim.Optimizer,
    state: TrainingState,
    dqn_replay: DQNReplayBuffer,
    device: torch.device,
    allow_empty_replay_resume: bool = False,
    teacher_retention_state_bank: TeacherRetentionStateBank | None = None,
    require_teacher_retention_state: bool = False,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary.")
    validate_checkpoint_mdp_contract(checkpoint)
    policies.load_checkpoint_state(checkpoint)
    optimizer_states = checkpoint.get("optimizer_state_dicts", {})
    if isinstance(optimizer_states, Mapping):
        if optimizer_states.get("ppo") is not None:
            ppo_optimizer.load_state_dict(optimizer_states["ppo"])
        if optimizer_states.get("dqn") is not None:
            dqn_optimizer.load_state_dict(optimizer_states["dqn"])
    state.global_step = int(checkpoint.get("global_step", 0))
    extra = checkpoint.get("extra", {})
    if isinstance(extra, Mapping):
        state.episode_count = int(extra.get("episode_count", 0))
        state.dqn_update_count = int(extra.get("dqn_update_count", 0))
        state.last_ppo_loss = _optional_float(extra.get("last_ppo_loss"))
        state.last_dqn_loss = _optional_float(extra.get("last_dqn_loss"))
    replay_state = checkpoint.get(DQN_REPLAY_STATE_KEY)
    replay_restored = False
    replay_restore_status = "missing"
    if isinstance(replay_state, Mapping):
        dqn_replay.load_state_dict(replay_state)
        replay_restored = True
        replay_restore_status = "restored"
    elif allow_empty_replay_resume:
        replay_restore_status = "explicit_empty_replay_warm_resume"
    else:
        raise ValueError(
            "DQN replay buffer state is missing from resume checkpoint; exact continuation requires "
            f"{DQN_REPLAY_STATE_KEY!r}. Pass --allow-empty-replay-resume only for diagnostic warm-start runs."
        )

    rng_state = checkpoint.get(RNG_STATE_KEY)
    rng_restored = False
    rng_restore_status = "missing"
    if isinstance(rng_state, Mapping):
        restore_rng_state(rng_state)
        rng_restored = True
        rng_restore_status = "restored"
    elif allow_empty_replay_resume:
        rng_restore_status = "explicit_fresh_rng_warm_resume"
    else:
        raise ValueError(
            "RNG state is missing from resume checkpoint; exact continuation requires "
            f"{RNG_STATE_KEY!r}. Pass --allow-empty-replay-resume only for diagnostic warm-start runs."
        )
    teacher_state_status = "disabled"
    if teacher_retention_state_bank is not None:
        teacher_state = checkpoint.get(TEACHER_RETENTION_STATE_BANK_KEY)
        if isinstance(teacher_state, Mapping):
            teacher_retention_state_bank.load_state_dict(teacher_state)
            teacher_state_status = "restored"
        elif require_teacher_retention_state:
            raise ValueError(
                "teacher retention state bank is missing from resume checkpoint; exact teacher-retention "
                f"continuation requires {TEACHER_RETENTION_STATE_BANK_KEY!r}."
            )
        else:
            teacher_state_status = "missing"
    return {
        "resume_mode": "exact" if replay_restored and rng_restored else "warm_start",
        "dqn_replay_restored": replay_restored,
        "dqn_replay_size": len(dqn_replay),
        "dqn_replay_total_added": dqn_replay.total_added,
        "replay_restore_status": replay_restore_status,
        "rng_state_restored": rng_restored,
        "rng_restore_status": rng_restore_status,
        "teacher_retention_state_status": teacher_state_status,
        "global_step": state.global_step,
    }


def load_fine_tune_initial_checkpoint(
    path: Path,
    *,
    policies: JointPolicyBundle,
    state: TrainingState,
    device: torch.device,
) -> dict[str, Any]:
    """Load model weights from a parent checkpoint without resuming trainer state."""

    validate_fine_tune_initial_checkpoint_path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary.")
    validate_checkpoint_mdp_contract(checkpoint)
    policies.load_checkpoint_state(checkpoint)
    state.global_step = 0
    state.episode_count = 0
    state.dqn_update_count = 0
    state.last_ppo_loss = None
    state.last_dqn_loss = None

    config = checkpoint.get("config")
    shared_cfg = config.get("shared_global_parameters", {}) if isinstance(config, Mapping) else {}
    source_env_id = (
        str(shared_cfg.get("environment_id", ""))
        if isinstance(shared_cfg, Mapping)
        else ""
    )
    source_contract = config.get("mdp_contract_version") if isinstance(config, Mapping) else None
    return {
        "initialized_from_checkpoint": str(path),
        "initialized_from_env_id": source_env_id,
        "initialized_from_global_step": int(checkpoint.get("global_step", 0)),
        "initialized_from_contract": source_contract,
        "fine_tune_parent_checkpoint": str(path),
        "resume_mode": "init_from_joint_checkpoint",
        "replay_restore_status": "fresh_replay_by_design",
    }


def validate_fine_tune_initial_checkpoint_path(path: Path) -> None:
    """Reject known invalid source checkpoint locations for targeted fine-tunes."""

    normalized = _normalize_checkpoint_guard_path(path)
    absolute_normalized = _normalize_checkpoint_guard_path(path.resolve(strict=False))
    parts = {part.lower() for part in path.parts}
    disallowed_parts = sorted(DISALLOWED_FINE_TUNE_INIT_PATH_PARTS & parts)
    if disallowed_parts:
        raise ValueError(f"disallowed fine-tune initialization checkpoint path component: {disallowed_parts[0]}")
    for token in DISALLOWED_FINE_TUNE_INIT_PATH_TOKENS:
        if token in normalized:
            raise ValueError(f"disallowed fine-tune initialization checkpoint path token: {token}")
    approved = _approved_fine_tune_initial_checkpoint_paths()
    if normalized not in approved and absolute_normalized not in approved:
        expected = ", ".join(sorted(APPROVED_FINE_TUNE_INIT_CHECKPOINTS))
        raise ValueError(
            "disallowed fine-tune initialization checkpoint path: "
            f"not approved for this targeted fine-tune. Expected one of: {expected}"
        )


def _approved_fine_tune_initial_checkpoint_paths() -> frozenset[str]:
    approved: set[str] = set()
    for checkpoint_path in APPROVED_FINE_TUNE_INIT_CHECKPOINTS:
        path = Path(checkpoint_path)
        approved.add(_normalize_checkpoint_guard_path(path))
        approved.add(_normalize_checkpoint_guard_path(path.resolve(strict=False)))
    return frozenset(approved)


def _normalize_checkpoint_guard_path(path: Path) -> str:
    return str(path).replace("\\", "/").lower().lstrip("./")


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Write checkpoint through a same-directory temp file, then atomically replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}")
    try:
        torch.save(dict(payload), temp_path)
        file_fd = os.open(str(temp_path), os.O_RDWR)
        try:
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        os.replace(temp_path, path)
        if hasattr(os, "O_RDONLY"):
            try:
                directory_fd = os.open(str(path.parent), os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def build_joint_trace_rows(
    transition: JointTransition,
    *,
    recorded_at: datetime | None = None,
    environment_id: str,
    team_id: str,
) -> tuple[EpisodeTransition, EpisodeTransition]:
    """Create PPO and DQN role rows for one synchronized joint transition."""

    base_recorded_at = _as_utc(recorded_at or datetime.now(timezone.utc))
    episode_id = _stable_uuid("episode", transition.episode_id)
    joint_action_id = _stable_uuid("joint-action", transition.joint_action_id)
    observation = _tensor_payload(transition.observation, expected_dim=OBSERVATION_DIM)
    next_observation = _tensor_payload(transition.next_observation, expected_dim=OBSERVATION_DIM)
    ppo_action = _tensor_payload(transition.ppo_action, expected_dim=CONTINUOUS_ACTION_DIM)
    dqn_action = _discrete_action(transition.dqn_action)
    info = _trace_info(transition, joint_action_id=joint_action_id)
    projected_action = _projected_action(transition.info)

    ppo_row = EpisodeTransition(
        episode_id=episode_id,
        step_id=transition.step_id,
        recorded_at=base_recorded_at,
        observation=observation,
        action={"continuous": ppo_action["vector"], "discrete_counterpart": dqn_action},
        reward=transition.reward_ppo_train,
        next_observation=next_observation,
        projected_action=projected_action,
        terminated=transition.terminated,
        truncated=transition.truncated,
        policy_id=PPO_POLICY_ID,
        environment_id=environment_id,
        agent_id=PPO_AGENT_ID,
        agent_role=PPO_AGENT_ROLE,
        team_id=team_id,
        joint_action_id=joint_action_id,
        local_reward=transition.reward_ppo_local,
        global_reward=transition.reward_global,
        ppo_local_reward=transition.reward_ppo_local,
        dqn_local_reward=transition.reward_dqn_local,
        info=info,
    )
    dqn_row = EpisodeTransition(
        episode_id=episode_id,
        step_id=transition.step_id,
        recorded_at=base_recorded_at + timedelta(microseconds=1),
        observation=observation,
        action={"discrete": dqn_action, "continuous_counterpart": ppo_action["vector"]},
        reward=transition.reward_dqn_train,
        next_observation=next_observation,
        projected_action=projected_action,
        terminated=transition.terminated,
        truncated=transition.truncated,
        policy_id=DQN_POLICY_ID,
        environment_id=environment_id,
        agent_id=DQN_AGENT_ID,
        agent_role=DQN_AGENT_ROLE,
        team_id=team_id,
        joint_action_id=joint_action_id,
        local_reward=transition.reward_dqn_local,
        global_reward=transition.reward_global,
        ppo_local_reward=transition.reward_ppo_local,
        dqn_local_reward=transition.reward_dqn_local,
        info=info,
    )
    return ppo_row, dqn_row


def build_joint_trace_batch(
    transitions: Iterable[JointTransition],
    *,
    environment_id: str,
    team_id: str,
    recorded_at: datetime | None = None,
) -> list[EpisodeTransition]:
    """Create two role rows per transition for batched `EpisodeStore.write_many`."""

    rows: list[EpisodeTransition] = []
    base_recorded_at = _as_utc(recorded_at or datetime.now(timezone.utc))
    for index, transition in enumerate(transitions):
        transition_time = base_recorded_at + timedelta(microseconds=index * 2)
        rows.extend(
            build_joint_trace_rows(
                transition,
                recorded_at=transition_time,
                environment_id=environment_id,
                team_id=team_id,
            )
        )
    return rows


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_reward_blend_config(torch_cfg: Mapping[str, Any]) -> JointRewardBlendConfig:
    """Build the Phase 1 role-specific reward blend contract from config."""

    return JointRewardBlendConfig(
        ppo_global_weight=float(torch_cfg.get("ppo_global_reward_weight", 0.35)),
        ppo_local_weight=float(torch_cfg.get("ppo_local_reward_weight", 0.65)),
        dqn_local_weight=float(torch_cfg.get("dqn_local_reward_weight", 0.90)),
        dqn_global_weight=float(torch_cfg.get("dqn_global_reward_weight", 0.10)),
        dqn_local_health_offset=float(torch_cfg.get("dqn_local_health_offset", 0.10)),
        dqn_local_health_scale=float(torch_cfg.get("dqn_local_health_scale", 0.25)),
    )


def _dqn_architecture_from_config(config: Mapping[str, Any]) -> str:
    """Read the optional internal DQN architecture while preserving flat legacy defaults."""

    section = config.get("torch_joint_training")
    if not isinstance(section, Mapping):
        section = config.get("torch_training")
    value = section.get("dqn_architecture", FLAT_DQN_ARCHITECTURE) if isinstance(section, Mapping) else FLAT_DQN_ARCHITECTURE
    architecture = str(value)
    if architecture not in SUPPORTED_DQN_ARCHITECTURES:
        raise ValueError(
            f"unsupported dqn_architecture: {architecture!r}; "
            f"expected one of {sorted(SUPPORTED_DQN_ARCHITECTURES)}."
        )
    return architecture


def build_environment_config(config: Mapping[str, Any], *, max_steps: int, seed: int) -> EnvironmentConfig:
    shared_cfg = _section(config, "shared_global_parameters")
    reward_physics = config.get("reward_physics", {})
    if reward_physics is None:
        reward_physics = {}
    if not isinstance(reward_physics, Mapping):
        raise KeyError("config section 'reward_physics' must be an object when provided.")
    return EnvironmentConfig(
        action_mode="joint",
        max_steps=max_steps,
        random_seed=seed,
        decision_interval=float(shared_cfg.get("decision_interval_seconds", 300.0)),
        speed_cost_coefficient=float(reward_physics.get("speed_cost_coefficient", 1.25)),
        speed_risk_coefficient=float(reward_physics.get("speed_risk_coefficient", 0.75)),
        speed_cost_penalty_weight=float(reward_physics.get("speed_cost_penalty_weight", 0.30)),
        unjustified_speed_penalty_weight=float(reward_physics.get("speed_unjustified_penalty", 0.20)),
        planned_replenishment_credit_weight=float(reward_physics.get("planned_replenishment_credit", 0.18)),
        planned_replenishment_cost_weight=float(reward_physics.get("planned_replenishment_cost_weight", 0.12)),
        excess_inventory_penalty_weight=float(reward_physics.get("excess_inventory_penalty", 0.15)),
        safety_stock_gap_credit_weight=float(reward_physics.get("safety_stock_gap_credit", 0.16)),
        capacity_opportunity_cost=float(reward_physics.get("capacity_opportunity_cost", 0.10)),
        low_speed_lateness_weight=float(reward_physics.get("low_speed_lateness_weight", 0.12)),
        flow_capacity_neglect_weight=float(reward_physics.get("flow_capacity_neglect_weight", 0.10)),
        flow_enablement_weight=float(reward_physics.get("flow_enablement_weight", 0.08)),
        dispatch_progress_weight=float(reward_physics.get("dispatch_progress_weight", 0.05)),
        infeasible_dispatch_penalty_weight=float(reward_physics.get("infeasible_dispatch_penalty_weight", 0.04)),
        max_extra_dispatch_budget=int(shared_cfg.get("max_extra_dispatch_budget", 4)),
        feasibility_gated_macro_budget=bool(shared_cfg.get("feasibility_gated_macro_budget", True)),
        global_transport_cost_penalty_weight=float(reward_physics.get("global_transport_cost_penalty_weight", 0.18)),
        global_backlog_reduction_weight=float(reward_physics.get("global_backlog_reduction_weight", 0.10)),
        global_lateness_reduction_weight=float(reward_physics.get("global_lateness_reduction_weight", 0.12)),
    )


def validate_active_mdp_contract(config: Mapping[str, Any]) -> None:
    """Ensure training configuration declares the physical-reality MDP contract."""

    version = config.get("mdp_contract_version")
    if version != MDP_CONTRACT_VERSION:
        raise ValueError(
            f"Config MDP contract mismatch: expected {MDP_CONTRACT_VERSION!r}, got {version!r}."
        )
    shared_cfg = config.get("shared_global_parameters", {})
    if isinstance(shared_cfg, Mapping) and "observation_dim" in shared_cfg:
        configured_dim = int(shared_cfg["observation_dim"])
        if configured_dim != OBSERVATION_DIM:
            raise ValueError(
                f"Config observation dimension mismatch: expected {OBSERVATION_DIM}, got {configured_dim}."
            )


def validate_checkpoint_mdp_contract(checkpoint: Mapping[str, Any]) -> None:
    """Reject stale checkpoints trained under older physical/economic rules."""

    config = checkpoint.get("config")
    version = config.get("mdp_contract_version") if isinstance(config, Mapping) else None
    if version != MDP_CONTRACT_VERSION:
        raise ValueError(MDP_CONTRACT_MISMATCH_MESSAGE)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and Torch RNG state for exact trainer resume."""

    cuda_states: list[Tensor] = []
    if torch.cuda.is_available():
        cuda_states = [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
    return {
        "version": RNG_STATE_VERSION,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda_rng_states": cuda_states,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG state captured by `capture_rng_state`."""

    if state.get("version") != RNG_STATE_VERSION:
        raise ValueError(f"unsupported RNG state version: {state.get('version')!r}.")
    python_state = state.get("python_random_state")
    numpy_state = state.get("numpy_random_state")
    torch_state = state.get("torch_rng_state")
    if python_state is None:
        raise ValueError("RNG state is missing python_random_state.")
    if numpy_state is None:
        raise ValueError("RNG state is missing numpy_random_state.")
    if not isinstance(torch_state, Tensor):
        raise TypeError("RNG state field 'torch_rng_state' must be a tensor.")
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state.detach().cpu())
    cuda_states = state.get("torch_cuda_rng_states", [])
    if torch.cuda.is_available() and isinstance(cuda_states, list) and cuda_states:
        torch.cuda.set_rng_state_all([cuda_state.detach().cpu() for cuda_state in cuda_states])


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")
    return device


def exploration_epsilon(
    step: int,
    *,
    total_timesteps: int,
    initial: float,
    final: float,
    fraction: float,
) -> float:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive.")
    if fraction <= 0.0:
        return final
    progress = min(max(step / (total_timesteps * fraction), 0.0), 1.0)
    return float(initial + ((final - initial) * progress))


def _reward_components(info: Mapping[str, Any], *, fallback_total: float) -> dict[str, float]:
    raw = info.get("reward_components", {})
    components = raw if isinstance(raw, Mapping) else {}
    return {
        "global": float(components.get("global", fallback_total)),
        "ppo_local": float(components.get("ppo_local", 0.0)),
        "dqn_local": float(components.get("dqn_local", 0.0)),
    }


def _observation_tensor(observation: np.ndarray, *, device: torch.device) -> Tensor:
    tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape != (1, OBSERVATION_DIM):
        raise ValueError(f"expected observation shape (1, {OBSERVATION_DIM}), got {tuple(tensor.shape)}.")
    return tensor


def _trace_info(transition: JointTransition, *, joint_action_id: UUID) -> dict[str, Any]:
    info = dict(transition.info)
    reward_components = info.get("reward_components")
    if not isinstance(reward_components, Mapping):
        reward_components = {}
    info["reward_components"] = {
        **dict(reward_components),
        "global": transition.reward_global,
        "ppo_local": transition.reward_ppo_local,
        "dqn_local": transition.reward_dqn_local,
        "ppo_train": transition.reward_ppo_train,
        "dqn_train": transition.reward_dqn_train,
        "total": transition.reward_total,
    }
    info["joint_action_id"] = str(joint_action_id)
    info["projected"] = transition.projected
    info["blocked"] = transition.blocked
    return info


def _projected_action(info: Mapping[str, Any]) -> Mapping[str, Any] | None:
    action = info.get("projected_action", info.get("action"))
    if isinstance(action, Mapping):
        return dict(action)
    return None


def _tensor_payload(value: Tensor, *, expected_dim: int) -> dict[str, list[float]]:
    tensor = _one_dimensional_tensor(value, expected_dim=expected_dim)
    return {"vector": [float(item) for item in tensor.cpu().tolist()]}


def _one_dimensional_tensor(value: Tensor, *, expected_dim: int) -> Tensor:
    if value.ndim == 1:
        tensor = value.detach().to(dtype=torch.float32)
    elif value.ndim == 2 and value.shape[0] == 1:
        tensor = value.detach().to(dtype=torch.float32).squeeze(0)
    else:
        raise ValueError(f"expected tensor shape ({expected_dim},) or (1, {expected_dim}), got {tuple(value.shape)}.")
    if tensor.shape[0] != expected_dim:
        raise ValueError(f"expected tensor dimension {expected_dim}, got {tensor.shape[0]}.")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("tensor contains non-finite values.")
    return tensor


def _discrete_action(value: Tensor) -> int:
    if value.ndim == 0:
        tensor = value.detach()
    elif value.ndim == 1 and value.shape[0] == 1:
        tensor = value.detach().squeeze(0)
    else:
        raise ValueError(f"expected DQN action shape () or (1,), got {tuple(value.shape)}.")
    if tensor.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
        raise ValueError(f"DQN action must use an integer dtype, got {tensor.dtype}.")
    action = int(tensor.item())
    if not 0 <= action < DISCRETE_ACTION_COUNT:
        raise ValueError(f"DQN action must be in [0, {DISCRETE_ACTION_COUNT - 1}], got {action}.")
    return action


def _stable_uuid(prefix: str, value: str) -> UUID:
    if not value:
        raise ValueError(f"{prefix} id is required for joint trace persistence.")
    try:
        return UUID(value)
    except ValueError:
        return uuid5(NAMESPACE_URL, f"5pl:{prefix}:{value}")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _section(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"config section {key!r} is missing or not an object.")
    return value


def _ensure_finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"{name} is non-finite.")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _next_interval(current: int, interval: int) -> int:
    if interval <= 0:
        raise ValueError("interval must be positive.")
    return ((current // interval) + 1) * interval


def _should_save_periodic_checkpoint(
    *,
    global_step: int,
    next_checkpoint_step: int,
    total_timesteps: int,
) -> bool:
    return int(global_step) >= int(next_checkpoint_step) and int(global_step) < int(total_timesteps)


def _should_write_final_training_outputs(*, start_step: int, global_step: int) -> bool:
    return int(global_step) > int(start_step)


def _save_final_artifacts_if_training_advanced(
    *,
    start_step: int,
    state: TrainingState,
    save: Callable[[], None],
) -> bool:
    if not _should_write_final_training_outputs(start_step=start_step, global_step=state.global_step):
        return False
    save()
    return True


def _validate_snapshot(row: Mapping[str, Any]) -> None:
    for key in ("mean_global_reward", "mean_ppo_local_reward", "mean_dqn_local_reward"):
        value = float(row[key])
        if not np.isfinite(value):
            raise FloatingPointError(f"metric {key} is non-finite.")


if __name__ == "__main__":
    asyncio.run(main())
