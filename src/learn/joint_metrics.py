"""Metrics aggregation and JSONL reporting for synchronized joint MARL training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.act.action_projector import CONTINUOUS_ACTION_DIM
from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT, DiscreteActionMapper
from src.learn.joint_buffers import JointTransition


@dataclass(frozen=True, slots=True)
class JointMetricsSnapshot:
    """Serializable metrics snapshot for console and JSONL reporting."""

    recorded_at: str
    global_step: int
    episode_count: int
    mean_total_reward: float
    mean_global_reward: float
    mean_ppo_local_reward: float
    mean_dqn_local_reward: float
    ppo_action_mean: list[float]
    ppo_action_std: list[float]
    dqn_action_entropy: float
    top_dqn_action_distribution: dict[str, float]
    action_failed_noop_by_id: dict[str, int]
    action_no_current_work_by_id: dict[str, int]
    action_no_unassigned_by_id: dict[str, int]
    action_dispatch_attempts_by_id: dict[str, int]
    action_dispatch_successes_by_id: dict[str, int]
    action_dispatch_success_ratio_by_id: dict[str, float]
    dispatch_success_per_attempt_by_stage_action: dict[str, float]
    top_action_concentration: float
    mean_dqn_local_reward_by_action_family: dict[str, float]
    lateness_pressure_by_stage_action: dict[str, float]
    useful_dispatch_opportunity_missed_by_action: dict[str, int]
    primary_fleet_useful_dispatch_by_scenario: dict[str, int]
    hold_under_lateness_pressure_by_scenario: dict[str, int]
    secondary_fleet_lateness_exposure: float
    mean_useful_dispatch_lateness_urgency_credit_by_action_family: dict[str, float]
    mean_primary_fleet_timing_justification_credit_by_action_family: dict[str, float]
    dispatch_distribution: dict[str, float]
    route_distribution: dict[str, float]
    mode_distribution: dict[str, float]
    reorder_distribution: dict[str, float]
    blocked_action_rate: float
    projected_action_rate: float
    service_level: float
    transport_cost: float
    safety_potential: float

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "recorded_at": self.recorded_at,
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "mean_total_reward": self.mean_total_reward,
            "mean_global_reward": self.mean_global_reward,
            "mean_ppo_local_reward": self.mean_ppo_local_reward,
            "mean_dqn_local_reward": self.mean_dqn_local_reward,
            "ppo_action_mean": self.ppo_action_mean,
            "ppo_action_std": self.ppo_action_std,
            "dqn_action_entropy": self.dqn_action_entropy,
            "top_dqn_action_distribution": self.top_dqn_action_distribution,
            "action_failed_noop_by_id": self.action_failed_noop_by_id,
            "action_no_current_work_by_id": self.action_no_current_work_by_id,
            "action_no_unassigned_by_id": self.action_no_unassigned_by_id,
            "action_dispatch_attempts_by_id": self.action_dispatch_attempts_by_id,
            "action_dispatch_successes_by_id": self.action_dispatch_successes_by_id,
            "action_dispatch_success_ratio_by_id": self.action_dispatch_success_ratio_by_id,
            "dispatch_success_per_attempt_by_stage_action": self.dispatch_success_per_attempt_by_stage_action,
            "top_action_concentration": self.top_action_concentration,
            "mean_dqn_local_reward_by_action_family": self.mean_dqn_local_reward_by_action_family,
            "lateness_pressure_by_stage_action": self.lateness_pressure_by_stage_action,
            "useful_dispatch_opportunity_missed_by_action": self.useful_dispatch_opportunity_missed_by_action,
            "primary_fleet_useful_dispatch_by_scenario": self.primary_fleet_useful_dispatch_by_scenario,
            "hold_under_lateness_pressure_by_scenario": self.hold_under_lateness_pressure_by_scenario,
            "secondary_fleet_lateness_exposure": self.secondary_fleet_lateness_exposure,
            "mean_useful_dispatch_lateness_urgency_credit_by_action_family": (
                self.mean_useful_dispatch_lateness_urgency_credit_by_action_family
            ),
            "mean_primary_fleet_timing_justification_credit_by_action_family": (
                self.mean_primary_fleet_timing_justification_credit_by_action_family
            ),
            "dispatch_distribution": self.dispatch_distribution,
            "route_distribution": self.route_distribution,
            "mode_distribution": self.mode_distribution,
            "reorder_distribution": self.reorder_distribution,
            "blocked_action_rate": self.blocked_action_rate,
            "projected_action_rate": self.projected_action_rate,
            "service_level": self.service_level,
            "transport_cost": self.transport_cost,
            "safety_potential": self.safety_potential,
        }


class JointMetricsAggregator:
    """Accumulate behavioral diagnostics from the shared joint transition stream."""

    def __init__(self, *, action_dim: int = CONTINUOUS_ACTION_DIM, action_count: int = DISCRETE_ACTION_COUNT) -> None:
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if action_count <= 0:
            raise ValueError("action_count must be positive.")

        self.action_dim = action_dim
        self.action_count = action_count
        self._mapper = DiscreteActionMapper()
        self.reset()

    def reset(self) -> None:
        self.total_environment_steps = 0
        self.episode_count = 0
        self._total_reward_sum = 0.0
        self._global_reward_sum = 0.0
        self._ppo_local_reward_sum = 0.0
        self._dqn_local_reward_sum = 0.0
        self._projected_steps = 0
        self._blocked_steps = 0
        self._ppo_action_sum = [0.0 for _ in range(self.action_dim)]
        self._ppo_action_square_sum = [0.0 for _ in range(self.action_dim)]
        self._dqn_action_counts: Counter[int] = Counter()
        self._action_failed_noop_counts: Counter[int] = Counter()
        self._action_no_current_work_counts: Counter[int] = Counter()
        self._action_no_unassigned_counts: Counter[int] = Counter()
        self._action_dispatch_attempt_counts: Counter[int] = Counter()
        self._action_dispatch_success_counts: Counter[int] = Counter()
        self._stage_action_dispatch_attempt_counts: Counter[str] = Counter()
        self._stage_action_dispatch_success_counts: Counter[str] = Counter()
        self._dqn_local_reward_sum_by_action_family: Counter[str] = Counter()
        self._dqn_local_reward_count_by_action_family: Counter[str] = Counter()
        self._lateness_pressure_sum_by_stage_action: Counter[str] = Counter()
        self._lateness_pressure_count_by_stage_action: Counter[str] = Counter()
        self._useful_dispatch_opportunity_missed_counts: Counter[int] = Counter()
        self._primary_fleet_useful_dispatch_counts_by_scenario: Counter[str] = Counter()
        self._hold_under_lateness_pressure_counts_by_scenario: Counter[str] = Counter()
        self._secondary_fleet_lateness_exposure_sum = 0.0
        self._useful_dispatch_urgency_credit_sum_by_action_family: Counter[str] = Counter()
        self._useful_dispatch_urgency_credit_count_by_action_family: Counter[str] = Counter()
        self._primary_fleet_timing_credit_sum_by_action_family: Counter[str] = Counter()
        self._primary_fleet_timing_credit_count_by_action_family: Counter[str] = Counter()
        self._dispatch_counts: Counter[str] = Counter()
        self._route_counts: Counter[str] = Counter()
        self._mode_counts: Counter[str] = Counter()
        self._reorder_counts: Counter[str] = Counter()
        self._service_level_sum = 0.0
        self._transport_cost_sum = 0.0
        self._safety_potential_sum = 0.0
        self._operational_metric_count = 0

    def record_transition(self, transition: JointTransition) -> None:
        ppo_action = _action_vector(transition.ppo_action, expected_dim=self.action_dim)
        dqn_action = _discrete_action(transition.dqn_action, action_count=self.action_count)
        rewards = {
            "reward_total": transition.reward_total,
            "reward_global": transition.reward_global,
            "reward_ppo_local": transition.reward_ppo_local,
            "reward_dqn_local": transition.reward_dqn_local,
        }
        for name, value in rewards.items():
            _validate_finite(name, value)

        self.total_environment_steps += 1
        self.episode_count += int(transition.done)
        self._total_reward_sum += float(transition.reward_total)
        self._global_reward_sum += float(transition.reward_global)
        self._ppo_local_reward_sum += float(transition.reward_ppo_local)
        self._dqn_local_reward_sum += float(transition.reward_dqn_local)
        self._projected_steps += int(transition.projected)
        self._blocked_steps += int(transition.blocked)

        for index, value in enumerate(ppo_action):
            self._ppo_action_sum[index] += value
            self._ppo_action_square_sum[index] += value * value

        self._dqn_action_counts[dqn_action] += 1
        decoded = self._mapper.map(dqn_action).as_dict()
        self._dispatch_counts[decoded["dispatch"]] += 1
        self._route_counts[decoded["route"]] += 1
        self._mode_counts[decoded["mode"]] += 1
        self._reorder_counts[decoded["reorder"]] += 1
        self._record_action_quality(
            dqn_action=dqn_action,
            decoded=decoded,
            reward_dqn_local=float(transition.reward_dqn_local),
            info=transition.info,
        )

        operational_metrics = _extract_operational_metrics(transition.info)
        if operational_metrics is not None:
            service_level, transport_cost, safety_potential = operational_metrics
            self._service_level_sum += service_level
            self._transport_cost_sum += transport_cost
            self._safety_potential_sum += safety_potential
            self._operational_metric_count += 1

    def snapshot(self, *, global_step: int | None = None) -> JointMetricsSnapshot:
        denominator = max(self.total_environment_steps, 1)
        metric_denominator = max(self._operational_metric_count, 1)
        ppo_mean = [value / denominator for value in self._ppo_action_sum]
        ppo_std = []
        for index, mean in enumerate(ppo_mean):
            second_moment = self._ppo_action_square_sum[index] / denominator
            variance = max(0.0, second_moment - (mean * mean))
            ppo_std.append(math.sqrt(variance))

        return JointMetricsSnapshot(
            recorded_at=datetime.now(UTC).isoformat(),
            global_step=self.total_environment_steps if global_step is None else int(global_step),
            episode_count=self.episode_count,
            mean_total_reward=self._total_reward_sum / denominator,
            mean_global_reward=self._global_reward_sum / denominator,
            mean_ppo_local_reward=self._ppo_local_reward_sum / denominator,
            mean_dqn_local_reward=self._dqn_local_reward_sum / denominator,
            ppo_action_mean=ppo_mean,
            ppo_action_std=ppo_std,
            dqn_action_entropy=_entropy(self._dqn_action_counts),
            top_dqn_action_distribution=_top_distribution(self._dqn_action_counts, denominator=denominator),
            action_failed_noop_by_id=_count_distribution_by_id(self._action_failed_noop_counts),
            action_no_current_work_by_id=_count_distribution_by_id(self._action_no_current_work_counts),
            action_no_unassigned_by_id=_count_distribution_by_id(self._action_no_unassigned_counts),
            action_dispatch_attempts_by_id=_count_distribution_by_id(self._action_dispatch_attempt_counts),
            action_dispatch_successes_by_id=_count_distribution_by_id(self._action_dispatch_success_counts),
            action_dispatch_success_ratio_by_id=_ratio_distribution(
                self._action_dispatch_success_counts,
                self._action_dispatch_attempt_counts,
            ),
            dispatch_success_per_attempt_by_stage_action=_ratio_distribution(
                self._stage_action_dispatch_success_counts,
                self._stage_action_dispatch_attempt_counts,
            ),
            top_action_concentration=_top_action_concentration(self._dqn_action_counts, denominator=denominator),
            mean_dqn_local_reward_by_action_family=_mean_by_key(
                self._dqn_local_reward_sum_by_action_family,
                self._dqn_local_reward_count_by_action_family,
            ),
            lateness_pressure_by_stage_action=_mean_by_key(
                self._lateness_pressure_sum_by_stage_action,
                self._lateness_pressure_count_by_stage_action,
            ),
            useful_dispatch_opportunity_missed_by_action=_count_distribution_by_id(
                self._useful_dispatch_opportunity_missed_counts
            ),
            primary_fleet_useful_dispatch_by_scenario={
                key: int(value)
                for key, value in sorted(self._primary_fleet_useful_dispatch_counts_by_scenario.items())
            },
            hold_under_lateness_pressure_by_scenario={
                key: int(value)
                for key, value in sorted(self._hold_under_lateness_pressure_counts_by_scenario.items())
            },
            secondary_fleet_lateness_exposure=float(self._secondary_fleet_lateness_exposure_sum),
            mean_useful_dispatch_lateness_urgency_credit_by_action_family=_mean_by_key(
                self._useful_dispatch_urgency_credit_sum_by_action_family,
                self._useful_dispatch_urgency_credit_count_by_action_family,
            ),
            mean_primary_fleet_timing_justification_credit_by_action_family=_mean_by_key(
                self._primary_fleet_timing_credit_sum_by_action_family,
                self._primary_fleet_timing_credit_count_by_action_family,
            ),
            dispatch_distribution=_distribution(self._dispatch_counts, denominator=denominator),
            route_distribution=_distribution(self._route_counts, denominator=denominator),
            mode_distribution=_distribution(self._mode_counts, denominator=denominator),
            reorder_distribution=_distribution(self._reorder_counts, denominator=denominator),
            blocked_action_rate=self._blocked_steps / denominator,
            projected_action_rate=self._projected_steps / denominator,
            service_level=self._service_level_sum / metric_denominator,
            transport_cost=self._transport_cost_sum / metric_denominator,
            safety_potential=self._safety_potential_sum / metric_denominator,
        )

    def append_jsonl(self, path: str | Path, *, global_step: int | None = None) -> JointMetricsSnapshot:
        snapshot = self.snapshot(global_step=global_step)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot.as_json_dict(), separators=(",", ":"), sort_keys=True))
            handle.write("\n")
        return snapshot

    def format_progress(self, *, global_step: int | None = None) -> str:
        snapshot = self.snapshot(global_step=global_step)
        ppo_std_average = sum(snapshot.ppo_action_std) / max(len(snapshot.ppo_action_std), 1)
        return (
            f"step={snapshot.global_step:,} "
            f"episodes={snapshot.episode_count:,} "
            f"global={snapshot.mean_global_reward:.6f} "
            f"ppo_local={snapshot.mean_ppo_local_reward:.6f} "
            f"dqn_local={snapshot.mean_dqn_local_reward:.6f} "
            f"ppo_std={ppo_std_average:.6f} "
            f"dqn_entropy={snapshot.dqn_action_entropy:.6f} "
            f"blocked={snapshot.blocked_action_rate:.4f} "
            f"projected={snapshot.projected_action_rate:.4f}"
        )

    def _record_action_quality(
        self,
        *,
        dqn_action: int,
        decoded: dict[str, str],
        reward_dqn_local: float,
        info: dict[str, Any],
    ) -> None:
        family_key = _action_family_key(decoded)
        self._dqn_local_reward_sum_by_action_family[family_key] += reward_dqn_local
        self._dqn_local_reward_count_by_action_family[family_key] += 1

        reward_components = info.get("reward_components")
        if not isinstance(reward_components, dict):
            reward_components = {}
        stage = str(info.get("curriculum_stage") or "unknown")
        stage_action_key = f"{stage}:{dqn_action}"
        lateness_pressure = _first_optional_float(
            reward_components.get("true_lateness_pressure"),
            reward_components.get("useful_dispatch_timing_pressure"),
            reward_components.get("actionable_lateness_risk"),
        )
        if lateness_pressure is not None and lateness_pressure > 0.0:
            self._lateness_pressure_sum_by_stage_action[stage_action_key] += lateness_pressure
            self._lateness_pressure_count_by_stage_action[stage_action_key] += 1

        useful_dispatch_opportunity_missed = _optional_float(
            reward_components.get("useful_dispatch_opportunity_missed")
        )
        hold_under_lateness_pressure = _optional_float(reward_components.get("hold_under_lateness_pressure"))
        if (
            (useful_dispatch_opportunity_missed is not None and useful_dispatch_opportunity_missed > 0.0)
            or (hold_under_lateness_pressure is not None and hold_under_lateness_pressure > 0.0)
        ):
            self._useful_dispatch_opportunity_missed_counts[dqn_action] += 1
        if hold_under_lateness_pressure is not None and hold_under_lateness_pressure > 0.0:
            self._hold_under_lateness_pressure_counts_by_scenario[stage] += 1

        secondary_fleet_lateness_exposure = _optional_float(
            reward_components.get("secondary_fleet_lateness_exposure")
        )
        if secondary_fleet_lateness_exposure is not None and secondary_fleet_lateness_exposure > 0.0:
            self._secondary_fleet_lateness_exposure_sum += secondary_fleet_lateness_exposure

        useful_dispatch_urgency_credit = _optional_float(
            reward_components.get("useful_dispatch_lateness_urgency_credit")
        )
        if useful_dispatch_urgency_credit is not None:
            self._useful_dispatch_urgency_credit_sum_by_action_family[family_key] += useful_dispatch_urgency_credit
            self._useful_dispatch_urgency_credit_count_by_action_family[family_key] += 1

        primary_fleet_timing_credit = _optional_float(
            reward_components.get("primary_fleet_timing_justification_credit")
        )
        if primary_fleet_timing_credit is not None:
            self._primary_fleet_timing_credit_sum_by_action_family[family_key] += primary_fleet_timing_credit
            self._primary_fleet_timing_credit_count_by_action_family[family_key] += 1

        current_dispatch_work_count = _first_optional_float(
            reward_components.get("current_dispatch_work_count"),
            info.get("dispatch_success_count"),
        )
        current_dispatch_work_signal = _first_optional_float(
            reward_components.get("current_dispatch_work_signal"),
            info.get("current_dispatch_work_signal"),
        )
        dispatch_success_count = _first_optional_float(
            info.get("dispatch_success_count"),
            reward_components.get("dispatch_success_count"),
            current_dispatch_work_count,
        )
        dispatched_orders = _first_optional_float(
            info.get("dqn_dispatched_orders"),
            reward_components.get("dqn_dispatched_orders"),
        )

        if decoded["dispatch"] != "dispatch":
            return

        self._action_dispatch_attempt_counts[dqn_action] += 1
        self._stage_action_dispatch_attempt_counts[stage_action_key] += 1

        dispatch_succeeded = any(
            value is not None and value > 0.0
            for value in (current_dispatch_work_count, dispatch_success_count)
        )
        if dispatch_succeeded:
            self._action_dispatch_success_counts[dqn_action] += 1
            self._stage_action_dispatch_success_counts[stage_action_key] += 1
        if (
            decoded["mode"] == "primary_fleet"
            and current_dispatch_work_signal is not None
            and current_dispatch_work_signal > 0.0
        ):
            self._primary_fleet_useful_dispatch_counts_by_scenario[stage] += 1

        explicit_no_current = _optional_float(info.get("dispatch_no_current_work"))
        no_current_work = (
            (explicit_no_current is not None and explicit_no_current > 0.0)
            or (current_dispatch_work_signal is not None and current_dispatch_work_signal <= 0.0)
        )
        no_unassigned = _first_optional_float(
            info.get("dispatch_no_unassigned_orders"),
            reward_components.get("dispatch_no_unassigned_orders"),
        )
        has_no_unassigned_exposure = no_unassigned is not None and no_unassigned > 0.0
        failed_noop = (
            no_current_work
            or has_no_unassigned_exposure
            or (dispatched_orders is not None and dispatched_orders <= 0.0 and not dispatch_succeeded)
        )

        if failed_noop:
            self._action_failed_noop_counts[dqn_action] += 1
        if no_current_work:
            self._action_no_current_work_counts[dqn_action] += 1
        if has_no_unassigned_exposure:
            self._action_no_unassigned_counts[dqn_action] += 1


def _action_vector(action: Tensor, *, expected_dim: int) -> list[float]:
    if action.ndim == 1:
        tensor = action.detach().to(dtype=torch.float32)
    elif action.ndim == 2 and action.shape[0] == 1:
        tensor = action.detach().to(dtype=torch.float32).squeeze(0)
    else:
        raise ValueError(f"expected PPO action shape ({expected_dim},) or (1, {expected_dim}), got {tuple(action.shape)}.")
    if tensor.shape[0] != expected_dim:
        raise ValueError(f"expected PPO action dimension {expected_dim}, got {tensor.shape[0]}.")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("PPO action contains non-finite values.")
    return [float(value) for value in tensor.cpu().tolist()]


def _discrete_action(action: Tensor, *, action_count: int) -> int:
    if action.ndim == 0:
        tensor = action.detach()
    elif action.ndim == 1 and action.shape[0] == 1:
        tensor = action.detach().squeeze(0)
    else:
        raise ValueError(f"expected DQN action shape () or (1,), got {tuple(action.shape)}.")
    if tensor.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
        raise ValueError(f"DQN action must use an integer dtype, got {tensor.dtype}.")
    value = int(tensor.item())
    if not 0 <= value < action_count:
        raise ValueError(f"DQN action must be in [0, {action_count - 1}], got {value}.")
    return value


def _extract_operational_metrics(info: dict[str, Any]) -> tuple[float, float, float] | None:
    candidates: list[dict[str, Any]] = []
    snapshot = info.get("snapshot")
    if isinstance(snapshot, dict):
        candidates.append(snapshot)
    reward_components = info.get("reward_components")
    if isinstance(reward_components, dict):
        candidates.append(reward_components)
    candidates.append(info)

    for candidate in candidates:
        service_level = _optional_float(candidate.get("service_level"))
        safety_potential = _optional_float(
            candidate.get("safety_potential", candidate.get("network_safety_potential"))
        )
        transport_cost = _optional_float(
            candidate.get("transport_cost", candidate.get("total_transport_cost", candidate.get("transport_cost_delta")))
        )
        if service_level is not None and safety_potential is not None and transport_cost is not None:
            return service_level, transport_cost, safety_potential
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _first_optional_float(*values: Any) -> float | None:
    for value in values:
        number = _optional_float(value)
        if number is not None:
            return number
    return None


def _validate_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}.")


def _entropy(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy


def _distribution(counts: Counter[str], *, denominator: int) -> dict[str, float]:
    if denominator <= 0:
        return {}
    return {key: value / denominator for key, value in sorted(counts.items())}


def _count_distribution_by_id(counts: Counter[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counts.items())}


def _ratio_distribution(numerators: Counter[Any], denominators: Counter[Any]) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for key, denominator in sorted(denominators.items(), key=lambda item: str(item[0])):
        if denominator <= 0:
            continue
        ratios[str(key)] = float(numerators.get(key, 0)) / float(denominator)
    return ratios


def _mean_by_key(sums: Counter[str], counts: Counter[str]) -> dict[str, float]:
    means: dict[str, float] = {}
    for key, count in sorted(counts.items()):
        if count <= 0:
            continue
        means[key] = float(sums.get(key, 0.0)) / float(count)
    return means


def _action_family_key(decoded: dict[str, str]) -> str:
    return f"{decoded['dispatch']}:{decoded['route']}:{decoded['mode']}:{decoded['reorder']}"


def _top_action_concentration(counts: Counter[int], *, denominator: int) -> float:
    if denominator <= 0 or not counts:
        return 0.0
    return max(counts.values()) / denominator


def _top_distribution(counts: Counter[int], *, denominator: int, top_k: int = 8) -> dict[str, float]:
    if denominator <= 0:
        return {}
    return {str(action): count / denominator for action, count in counts.most_common(top_k)}
