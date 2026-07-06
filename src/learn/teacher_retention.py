"""Teacher-retention helpers for DQN behavior anchoring."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any, Final

import torch
from torch import Tensor
import torch.nn.functional as F

from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.act.observation_builder import OBSERVATION_DIM
from src.think.joint_policies import CHECKPOINT_VERSION, JointPolicyBundle


TEACHER_RETENTION_CONFIG_KEY: Final[str] = "teacher_retention"
TEACHER_RETENTION_STATE_BANK_KEY: Final[str] = "teacher_retention_state_bank"
TEACHER_RETENTION_STATE_BANK_VERSION: Final[str] = "teacher_retention_state_bank_v1"
TEACHER_MDP_CONTRACT_VERSION: Final[str] = "physical_reality_v5_route_candidate_visibility"
BAD_POCKET_ACTION_IDS: Final[tuple[int, ...]] = (32, 33, 41, 45, 46)
GATE_CRITICAL_TEACHER_SCENARIOS: Final[tuple[str, ...]] = (
    "baseline",
    "lead_time",
    "route_disruption",
    "premium_sla",
    "high_holding",
    "vehicle_scarcity",
    "mixed_stress",
    "demand_spike",
)
TEACHER_SCENARIO_ALIASES: Final[dict[str, str]] = {
    "normal": "baseline",
    "normal_operation": "baseline",
    "normal_v4": "baseline",
    "baseline": "baseline",
    "baseline_normal": "baseline",
    "lead_time": "lead_time",
    "lead_time_delay": "lead_time",
    "lead_time_volatility": "lead_time",
    "route": "route_disruption",
    "route_disruption": "route_disruption",
    "route_disruption_congestion": "route_disruption",
    "premium": "premium_sla",
    "premium_sla": "premium_sla",
    "premium_sla_pressure": "premium_sla",
    "high_holding": "high_holding",
    "high_holding_cost": "high_holding",
    "vehicle": "vehicle_scarcity",
    "vehicle_scarcity": "vehicle_scarcity",
    "vehicle_scarcity_capacity_shock": "vehicle_scarcity",
    "mixed": "mixed_stress",
    "mixed_stress": "mixed_stress",
    "demand_spike": "demand_spike",
    "demand_spike_volatility": "demand_spike",
}


@dataclass(frozen=True, slots=True)
class TeacherRetentionConfig:
    enabled: bool = False
    production_teacher_checkpoint: Path | None = None
    candidate_teacher_checkpoint: Path | None = None
    loss_weight: float = 0.05
    cross_entropy_weight: float = 1.0
    kl_weight: float = 0.25
    bad_pocket_margin_weight: float = 0.25
    bad_pocket_margin: float = 0.10
    max_loss: float = 0.10
    temperature: float = 1.0
    state_bank_batch_size: int = 128
    state_bank_max_per_scenario: int = 512
    bad_pocket_action_ids: tuple[int, ...] = BAD_POCKET_ACTION_IDS
    teacher_preference_by_scenario: Mapping[str, str] = field(
        default_factory=lambda: {
            "baseline": "production",
            "lead_time": "candidate",
            "route_disruption": "candidate",
            "premium_sla": "candidate",
            "high_holding": "candidate",
            "vehicle_scarcity": "candidate",
            "mixed_stress": "candidate",
            "demand_spike": "candidate",
        }
    )

    def __post_init__(self) -> None:
        if self.loss_weight < 0.0:
            raise ValueError("teacher_retention.loss_weight must be non-negative.")
        if self.cross_entropy_weight < 0.0:
            raise ValueError("teacher_retention.cross_entropy_weight must be non-negative.")
        if self.kl_weight < 0.0:
            raise ValueError("teacher_retention.kl_weight must be non-negative.")
        if self.bad_pocket_margin_weight < 0.0:
            raise ValueError("teacher_retention.bad_pocket_margin_weight must be non-negative.")
        if self.bad_pocket_margin < 0.0:
            raise ValueError("teacher_retention.bad_pocket_margin must be non-negative.")
        if self.max_loss <= 0.0:
            raise ValueError("teacher_retention.max_loss must be positive.")
        if self.temperature <= 0.0:
            raise ValueError("teacher_retention.temperature must be positive.")
        if self.state_bank_batch_size <= 0:
            raise ValueError("teacher_retention.state_bank_batch_size must be positive.")
        if self.state_bank_max_per_scenario <= 0:
            raise ValueError("teacher_retention.state_bank_max_per_scenario must be positive.")
        for action_id in self.bad_pocket_action_ids:
            if not 0 <= int(action_id) < DISCRETE_ACTION_COUNT:
                raise ValueError(f"bad pocket action id {action_id} is outside [0, {DISCRETE_ACTION_COUNT - 1}].")
        if self.enabled and self.production_teacher_checkpoint is None and self.candidate_teacher_checkpoint is None:
            raise ValueError("teacher_retention enabled requires at least one teacher checkpoint.")


@dataclass(frozen=True, slots=True)
class LoadedTeacherPolicy:
    name: str
    checkpoint_path: Path
    policy: JointPolicyBundle


@dataclass(frozen=True, slots=True)
class TeacherRetentionBatch:
    observations: Tensor
    scenario_ids: tuple[str, ...]
    scenario_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class TeacherRetentionLossResult:
    loss: Tensor
    metrics: dict[str, Any]
    bad_pocket_margin_by_action: dict[str, float]


class TeacherRetentionStateBank:
    """Small ring store of gate-critical observations grouped by scenario."""

    def __init__(
        self,
        *,
        max_per_scenario: int,
        observation_dim: int = OBSERVATION_DIM,
        scenarios: Sequence[str] = GATE_CRITICAL_TEACHER_SCENARIOS,
    ) -> None:
        if max_per_scenario <= 0:
            raise ValueError("max_per_scenario must be positive.")
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        canonical_scenarios = tuple(dict.fromkeys(canonical_teacher_scenario(scenario) for scenario in scenarios))
        if not canonical_scenarios:
            raise ValueError("at least one scenario is required.")
        self.max_per_scenario = int(max_per_scenario)
        self.observation_dim = int(observation_dim)
        self.scenarios = canonical_scenarios
        self._banks: dict[str, list[Tensor]] = {scenario: [] for scenario in self.scenarios}
        self._positions: dict[str, int] = {scenario: 0 for scenario in self.scenarios}

    @property
    def total_size(self) -> int:
        return sum(len(items) for items in self._banks.values())

    def record(self, observation: Tensor | Sequence[float], scenario_id: str | None) -> bool:
        scenario = canonical_teacher_scenario(scenario_id)
        if scenario not in self._banks:
            return False
        tensor = torch.as_tensor(observation, dtype=torch.float32).detach().cpu().flatten()
        if tuple(tensor.shape) != (self.observation_dim,):
            raise ValueError(f"teacher retention observation_dim mismatch: expected {self.observation_dim}, got {tensor.numel()}.")
        if not torch.isfinite(tensor).all():
            raise ValueError("teacher retention observation contains non-finite values.")
        bank = self._banks[scenario]
        if len(bank) < self.max_per_scenario:
            bank.append(tensor.clone())
        else:
            position = self._positions[scenario] % self.max_per_scenario
            bank[position] = tensor.clone()
            self._positions[scenario] = (position + 1) % self.max_per_scenario
        return True

    @property
    def can_sample(self) -> bool:
        return self.total_size > 0

    def scenario_counts(self) -> dict[str, int]:
        return {scenario: len(self._banks[scenario]) for scenario in self.scenarios if self._banks[scenario]}

    def sample(self, *, batch_size: int, device: torch.device | str) -> TeacherRetentionBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        items: list[tuple[str, Tensor]] = [
            (scenario, observation)
            for scenario in self.scenarios
            for observation in self._banks[scenario]
        ]
        if not items:
            raise RuntimeError("teacher retention state bank is empty.")
        if batch_size <= len(items):
            selected = random.sample(items, k=int(batch_size))
        else:
            selected = [items[index % len(items)] for index in range(int(batch_size))]
        scenario_ids = tuple(scenario for scenario, _observation in selected)
        observations = torch.stack([observation for _scenario, observation in selected]).to(device=device, dtype=torch.float32)
        return TeacherRetentionBatch(
            observations=observations,
            scenario_ids=scenario_ids,
            scenario_counts=self.scenario_counts(),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": TEACHER_RETENTION_STATE_BANK_VERSION,
            "max_per_scenario": self.max_per_scenario,
            "observation_dim": self.observation_dim,
            "scenarios": list(self.scenarios),
            "positions": dict(self._positions),
            "banks": {
                scenario: torch.stack(items).cpu() if items else torch.empty((0, self.observation_dim), dtype=torch.float32)
                for scenario, items in self._banks.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != TEACHER_RETENTION_STATE_BANK_VERSION:
            raise ValueError(f"unsupported teacher retention state bank version: {state.get('version')!r}.")
        if int(state.get("max_per_scenario", -1)) != self.max_per_scenario:
            raise ValueError("teacher retention state bank max_per_scenario mismatch.")
        if int(state.get("observation_dim", -1)) != self.observation_dim:
            raise ValueError("teacher retention state bank observation_dim mismatch.")
        scenarios = tuple(canonical_teacher_scenario(str(item)) for item in state.get("scenarios", ()))
        if scenarios != self.scenarios:
            raise ValueError("teacher retention state bank scenarios mismatch.")
        banks = state.get("banks")
        positions = state.get("positions", {})
        if not isinstance(banks, Mapping):
            raise ValueError("teacher retention state bank payload is missing banks.")
        for scenario in self.scenarios:
            raw = banks.get(scenario)
            tensor = torch.as_tensor(raw, dtype=torch.float32).detach().cpu()
            if tensor.ndim != 2 or tensor.shape[1] != self.observation_dim:
                raise ValueError(f"teacher retention bank {scenario!r} shape mismatch.")
            if tensor.shape[0] > self.max_per_scenario:
                raise ValueError(f"teacher retention bank {scenario!r} exceeds max_per_scenario.")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"teacher retention bank {scenario!r} contains non-finite observations.")
            self._banks[scenario] = [row.clone() for row in tensor]
            self._positions[scenario] = int(positions.get(scenario, 0)) % self.max_per_scenario


class TeacherRetentionRuntime:
    """Frozen teacher policies plus scenario-based teacher selection."""

    def __init__(self, *, config: TeacherRetentionConfig, teachers: Mapping[str, LoadedTeacherPolicy]) -> None:
        if not config.enabled:
            raise ValueError("TeacherRetentionRuntime requires an enabled config.")
        if not teachers:
            raise ValueError("TeacherRetentionRuntime requires at least one teacher.")
        self.config = config
        self.teachers = dict(teachers)

    @classmethod
    def from_config(cls, config: TeacherRetentionConfig, *, device: torch.device | str) -> "TeacherRetentionRuntime | None":
        if not config.enabled:
            return None
        teachers: dict[str, LoadedTeacherPolicy] = {}
        if config.production_teacher_checkpoint is not None:
            teachers["production"] = load_teacher_policy_checkpoint(
                config.production_teacher_checkpoint,
                device=device,
                teacher_name="production",
            )
        if config.candidate_teacher_checkpoint is not None:
            teachers["candidate"] = load_teacher_policy_checkpoint(
                config.candidate_teacher_checkpoint,
                device=device,
                teacher_name="candidate",
            )
        return cls(config=config, teachers=teachers)

    def loss_for_batch(
        self,
        *,
        student_policy: JointPolicyBundle,
        batch: TeacherRetentionBatch,
    ) -> TeacherRetentionLossResult:
        if batch.observations.numel() == 0:
            zero = torch.zeros((), device=batch.observations.device, dtype=torch.float32)
            return _empty_loss_result(zero)
        student_q = student_policy.dqn_q_network(batch.observations)
        teacher_groups: dict[str, list[int]] = {}
        for index, scenario_id in enumerate(batch.scenario_ids):
            teacher_name = self._teacher_name_for_scenario(scenario_id)
            teacher_groups.setdefault(teacher_name, []).append(index)

        total = max(1, len(batch.scenario_ids))
        weighted_loss = torch.zeros((), device=batch.observations.device, dtype=torch.float32)
        aggregate_metrics: dict[str, float] = {
            "teacher_dqn_kl": 0.0,
            "teacher_dqn_ce": 0.0,
            "teacher_dqn_action_match_rate": 0.0,
            "teacher_retention_loss": 0.0,
            "gate_critical_action_drift": 0.0,
        }
        bad_margin_sums: dict[str, float] = {str(action_id): 0.0 for action_id in self.config.bad_pocket_action_ids}
        bad_margin_weights: dict[str, int] = {str(action_id): 0 for action_id in self.config.bad_pocket_action_ids}
        for teacher_name, indices in teacher_groups.items():
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=batch.observations.device)
            group_observations = batch.observations.index_select(0, index_tensor)
            group_student_q = student_q.index_select(0, index_tensor)
            teacher = self.teachers[teacher_name].policy
            with torch.no_grad():
                group_teacher_q = teacher.dqn_q_network(group_observations)
            result = compute_dqn_teacher_retention_loss(
                student_q=group_student_q,
                teacher_q=group_teacher_q,
                config=self.config,
                scenario_ids=tuple(batch.scenario_ids[index] for index in indices),
            )
            weight = float(len(indices)) / float(total)
            weighted_loss = weighted_loss + (result.loss * weight)
            for key in aggregate_metrics:
                aggregate_metrics[key] += float(result.metrics[key]) * weight
            for action_id, margin in result.bad_pocket_margin_by_action.items():
                bad_margin_sums[action_id] += float(margin) * len(indices)
                bad_margin_weights[action_id] += len(indices)

        bad_margins = {
            action_id: bad_margin_sums[action_id] / float(bad_margin_weights[action_id])
            for action_id in bad_margin_sums
            if bad_margin_weights[action_id] > 0
        }
        aggregate_metrics["bad_pocket_margin_by_action"] = bad_margins
        aggregate_metrics["teacher_bank_scenario_counts"] = dict(batch.scenario_counts)
        aggregate_metrics["teacher_retention_loss"] = float(weighted_loss.detach().cpu().item())
        return TeacherRetentionLossResult(
            loss=weighted_loss,
            metrics=aggregate_metrics,
            bad_pocket_margin_by_action=bad_margins,
        )

    def _teacher_name_for_scenario(self, scenario_id: str) -> str:
        scenario = canonical_teacher_scenario(scenario_id)
        preferred = str(self.config.teacher_preference_by_scenario.get(scenario, "candidate"))
        if preferred in self.teachers:
            return preferred
        if "candidate" in self.teachers:
            return "candidate"
        if "production" in self.teachers:
            return "production"
        raise ValueError("no usable teacher policy is loaded.")


def teacher_retention_config_from_training_config(
    config: Mapping[str, Any],
    *,
    args: Any | None = None,
) -> TeacherRetentionConfig:
    section = config.get(TEACHER_RETENTION_CONFIG_KEY, {})
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError("teacher_retention config section must be an object.")

    cli_enabled = bool(getattr(args, "enable_teacher_retention", False)) if args is not None else False
    config_enabled = bool(section.get("enabled", False))
    enabled = bool(cli_enabled or config_enabled)
    explicit_enable = bool(section.get("explicit_enable", False) or cli_enabled)
    if config_enabled and not explicit_enable:
        raise ValueError("teacher_retention.enabled requires explicit_enable=true in config or --enable-teacher-retention.")

    production_path = _path_or_none(
        getattr(args, "teacher_production_checkpoint", None) if args is not None else None,
        section.get("production_teacher_checkpoint"),
    )
    candidate_path = _path_or_none(
        getattr(args, "teacher_candidate_checkpoint", None) if args is not None else None,
        section.get("candidate_teacher_checkpoint"),
    )
    preference = section.get("teacher_preference_by_scenario", None)
    preference_by_scenario = _teacher_preference(preference)
    bad_ids = tuple(int(item) for item in section.get("bad_pocket_action_ids", BAD_POCKET_ACTION_IDS))
    return TeacherRetentionConfig(
        enabled=enabled,
        production_teacher_checkpoint=production_path,
        candidate_teacher_checkpoint=candidate_path,
        loss_weight=float(section.get("loss_weight", 0.05)),
        cross_entropy_weight=float(section.get("cross_entropy_weight", 1.0)),
        kl_weight=float(section.get("kl_weight", 0.25)),
        bad_pocket_margin_weight=float(section.get("bad_pocket_margin_weight", 0.25)),
        bad_pocket_margin=float(section.get("bad_pocket_margin", 0.10)),
        max_loss=float(section.get("max_loss", 0.10)),
        temperature=float(section.get("temperature", 1.0)),
        state_bank_batch_size=int(section.get("state_bank_batch_size", 128)),
        state_bank_max_per_scenario=int(section.get("state_bank_max_per_scenario", 512)),
        bad_pocket_action_ids=bad_ids,
        teacher_preference_by_scenario=preference_by_scenario,
    )


def load_teacher_policy_checkpoint(
    path: Path,
    *,
    device: torch.device | str,
    teacher_name: str,
) -> LoadedTeacherPolicy:
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("teacher checkpoint must be a dictionary.")
    validate_teacher_checkpoint_contract(checkpoint)
    policy = JointPolicyBundle().to(device)
    policy.load_checkpoint_state(checkpoint)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return LoadedTeacherPolicy(name=teacher_name, checkpoint_path=checkpoint_path, policy=policy)


def validate_teacher_checkpoint_contract(checkpoint: Mapping[str, Any]) -> None:
    config = checkpoint.get("config")
    contract = config.get("mdp_contract_version") if isinstance(config, Mapping) else None
    if contract != TEACHER_MDP_CONTRACT_VERSION:
        raise ValueError(
            "Teacher checkpoint MDP contract mismatch: expected "
            f"{TEACHER_MDP_CONTRACT_VERSION!r}, got {contract!r}."
        )
    version = checkpoint.get("checkpoint_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported teacher checkpoint_version: expected {CHECKPOINT_VERSION!r}, got {version!r}.")
    if checkpoint.get("observation_dim") != OBSERVATION_DIM:
        raise ValueError(f"teacher checkpoint observation_dim mismatch: expected {OBSERVATION_DIM}.")
    if checkpoint.get("discrete_action_count") != DISCRETE_ACTION_COUNT:
        raise ValueError(f"teacher checkpoint discrete_action_count mismatch: expected {DISCRETE_ACTION_COUNT}.")


def compute_dqn_teacher_retention_loss(
    *,
    student_q: Tensor,
    teacher_q: Tensor,
    config: TeacherRetentionConfig,
    scenario_ids: Sequence[str] | None = None,
) -> TeacherRetentionLossResult:
    if not config.enabled:
        zero = torch.zeros((), dtype=student_q.dtype, device=student_q.device)
        return _empty_loss_result(zero)
    if student_q.shape != teacher_q.shape:
        raise ValueError(f"student_q and teacher_q shape mismatch: {tuple(student_q.shape)} vs {tuple(teacher_q.shape)}.")
    if student_q.ndim != 2 or student_q.shape[1] != DISCRETE_ACTION_COUNT:
        raise ValueError(f"expected Q tensors with shape (batch, {DISCRETE_ACTION_COUNT}).")
    if not torch.isfinite(student_q).all() or not torch.isfinite(teacher_q).all():
        raise ValueError("teacher retention Q tensors must be finite.")
    if student_q.shape[0] == 0:
        zero = torch.zeros((), dtype=student_q.dtype, device=student_q.device)
        return _empty_loss_result(zero)

    temperature = float(config.temperature)
    teacher_actions = teacher_q.argmax(dim=1)
    student_actions = student_q.argmax(dim=1)
    teacher_probs = F.softmax(teacher_q.detach() / temperature, dim=1)
    student_log_probs = F.log_softmax(student_q / temperature, dim=1)
    student_ce = F.cross_entropy(student_q / temperature, teacher_actions)
    with torch.no_grad():
        teacher_self_ce = F.cross_entropy(teacher_q.detach() / temperature, teacher_actions)
    ce = F.relu(student_ce - teacher_self_ce)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    margin_loss, bad_margins = _bad_pocket_margin_loss(
        student_q=student_q,
        teacher_actions=teacher_actions,
        bad_pocket_action_ids=config.bad_pocket_action_ids,
        margin=float(config.bad_pocket_margin),
    )
    raw_loss = (
        (float(config.cross_entropy_weight) * ce)
        + (float(config.kl_weight) * kl)
        + (float(config.bad_pocket_margin_weight) * margin_loss)
    )
    weighted_loss = torch.clamp(float(config.loss_weight) * raw_loss, max=float(config.max_loss))
    bad_ids = torch.as_tensor(config.bad_pocket_action_ids, dtype=torch.long, device=student_q.device)
    student_bad = (student_actions.unsqueeze(1) == bad_ids.view(1, -1)).any(dim=1)
    teacher_bad = (teacher_actions.unsqueeze(1) == bad_ids.view(1, -1)).any(dim=1)
    drift = (student_bad & ~teacher_bad).to(dtype=torch.float32).mean()
    scenario_counts = Counter(canonical_teacher_scenario(item) for item in (scenario_ids or ()))
    metrics: dict[str, Any] = {
        "teacher_dqn_kl": float(kl.detach().cpu().item()),
        "teacher_dqn_ce": float(ce.detach().cpu().item()),
        "teacher_dqn_action_match_rate": float((student_actions == teacher_actions).to(dtype=torch.float32).mean().item()),
        "teacher_retention_loss": float(weighted_loss.detach().cpu().item()),
        "gate_critical_action_drift": float(drift.detach().cpu().item()),
        "bad_pocket_margin_by_action": bad_margins,
        "teacher_bank_scenario_counts": dict(scenario_counts),
    }
    return TeacherRetentionLossResult(loss=weighted_loss, metrics=metrics, bad_pocket_margin_by_action=bad_margins)


def record_teacher_retention_metrics(timing_window: Any, metrics: Mapping[str, Any]) -> None:
    if timing_window is None:
        return
    timing_window.teacher_retention_updates_window += 1
    timing_window.teacher_dqn_kl_sum_window += float(metrics.get("teacher_dqn_kl", 0.0))
    timing_window.teacher_dqn_ce_sum_window += float(metrics.get("teacher_dqn_ce", 0.0))
    timing_window.teacher_dqn_action_match_rate_sum_window += float(metrics.get("teacher_dqn_action_match_rate", 0.0))
    timing_window.teacher_retention_loss_sum_window += float(metrics.get("teacher_retention_loss", 0.0))
    timing_window.gate_critical_action_drift_sum_window += float(metrics.get("gate_critical_action_drift", 0.0))
    margins = metrics.get("bad_pocket_margin_by_action", {})
    if isinstance(margins, Mapping):
        for action_id, value in margins.items():
            key = str(action_id)
            timing_window.bad_pocket_margin_sum_by_action_window[key] = (
                timing_window.bad_pocket_margin_sum_by_action_window.get(key, 0.0) + float(value)
            )
            timing_window.bad_pocket_margin_count_by_action_window[key] = (
                timing_window.bad_pocket_margin_count_by_action_window.get(key, 0) + 1
            )
    scenario_counts = metrics.get("teacher_bank_scenario_counts", {})
    if isinstance(scenario_counts, Mapping):
        for scenario, count in scenario_counts.items():
            key = canonical_teacher_scenario(str(scenario))
            timing_window.teacher_bank_scenario_counts[key] = int(count)


def canonical_teacher_scenario(scenario_id: str | None) -> str:
    if scenario_id is None:
        return "baseline"
    normalized = str(scenario_id).strip().lower()
    return TEACHER_SCENARIO_ALIASES.get(normalized, normalized)


def _bad_pocket_margin_loss(
    *,
    student_q: Tensor,
    teacher_actions: Tensor,
    bad_pocket_action_ids: Sequence[int],
    margin: float,
) -> tuple[Tensor, dict[str, float]]:
    if not bad_pocket_action_ids:
        return torch.zeros((), dtype=student_q.dtype, device=student_q.device), {}
    teacher_q = student_q.gather(1, teacher_actions.view(-1, 1)).squeeze(1)
    losses: list[Tensor] = []
    margin_by_action: dict[str, float] = {}
    for action_id in bad_pocket_action_ids:
        action = int(action_id)
        bad_q = student_q[:, action]
        margin_values = teacher_q - bad_q
        mask = teacher_actions != action
        if bool(mask.any().item()):
            losses.append(F.relu(float(margin) - margin_values[mask]).mean())
            margin_by_action[str(action)] = float(margin_values[mask].detach().mean().cpu().item())
        else:
            margin_by_action[str(action)] = float(margin_values.detach().mean().cpu().item())
    if not losses:
        return torch.zeros((), dtype=student_q.dtype, device=student_q.device), margin_by_action
    return torch.stack(losses).mean(), margin_by_action


def _empty_loss_result(zero: Tensor) -> TeacherRetentionLossResult:
    bad_margins = {str(action_id): 0.0 for action_id in BAD_POCKET_ACTION_IDS}
    return TeacherRetentionLossResult(
        loss=zero,
        metrics={
            "teacher_dqn_kl": 0.0,
            "teacher_dqn_ce": 0.0,
            "teacher_dqn_action_match_rate": 0.0,
            "teacher_retention_loss": 0.0,
            "gate_critical_action_drift": 0.0,
            "bad_pocket_margin_by_action": bad_margins,
            "teacher_bank_scenario_counts": {},
        },
        bad_pocket_margin_by_action=bad_margins,
    )


def _path_or_none(primary: Any, fallback: Any) -> Path | None:
    value = primary if primary not in (None, "") else fallback
    if value in (None, ""):
        return None
    return Path(str(value))


def _teacher_preference(raw: Any) -> Mapping[str, str]:
    defaults = TeacherRetentionConfig().teacher_preference_by_scenario
    if raw is None:
        return defaults
    if not isinstance(raw, Mapping):
        raise ValueError("teacher_retention.teacher_preference_by_scenario must be an object.")
    merged = dict(defaults)
    for scenario, teacher_name in raw.items():
        teacher_text = str(teacher_name)
        if teacher_text not in {"production", "candidate"}:
            raise ValueError("teacher preference must be 'production' or 'candidate'.")
        merged[canonical_teacher_scenario(str(scenario))] = teacher_text
    return merged
