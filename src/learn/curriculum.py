"""Curriculum generation for progressively harder digital-twin scenarios."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Mapping
from uuid import UUID, uuid4

from src.act.disruptions import DisruptionEvent, DisruptionKind
from src.shared.metrics import clamp
from src.shared.types import JsonDict


class CurriculumStage(StrEnum):
    BASELINE = "baseline"
    CONGESTED = "congested"
    DISRUPTED = "disrupted"
    ADVERSARIAL = "adversarial"


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    mean_reward: float
    service_level: float
    safety_potential: float
    completion_rate: float
    episodes: int


@dataclass(frozen=True, slots=True)
class ScenarioDifficulty:
    stage: CurriculumStage
    disruption_probability: float
    max_disruption_severity: float
    demand_multiplier: float
    capacity_loss_fraction: float
    metadata: JsonDict = field(default_factory=dict)


class CurriculumManager:
    """Select scenario difficulty from recent policy performance."""

    def __init__(
        self,
        *,
        service_threshold: float = 0.92,
        safety_threshold: float = 0.35,
        completion_threshold: float = 0.85,
    ) -> None:
        self.service_threshold = service_threshold
        self.safety_threshold = safety_threshold
        self.completion_threshold = completion_threshold

    def choose_difficulty(self, performance: PerformanceSummary) -> ScenarioDifficulty:
        if (
            performance.service_level >= self.service_threshold
            and performance.safety_potential <= self.safety_threshold
            and performance.completion_rate >= self.completion_threshold
        ):
            return ScenarioDifficulty(
                stage=CurriculumStage.DISRUPTED,
                disruption_probability=0.18,
                max_disruption_severity=0.75,
                demand_multiplier=1.30,
                capacity_loss_fraction=0.30,
                metadata={"reason": "agent_exceeds_thresholds"},
            )
        if performance.service_level >= self.service_threshold * 0.85:
            return ScenarioDifficulty(
                stage=CurriculumStage.CONGESTED,
                disruption_probability=0.08,
                max_disruption_severity=0.45,
                demand_multiplier=1.15,
                capacity_loss_fraction=0.15,
                metadata={"reason": "agent_ready_for_congestion"},
            )
        return ScenarioDifficulty(
            stage=CurriculumStage.BASELINE,
            disruption_probability=0.02,
            max_disruption_severity=0.20,
            demand_multiplier=1.00,
            capacity_loss_fraction=0.05,
            metadata={"reason": "agent_needs_baseline_stability"},
        )

    def escalate(self, difficulty: ScenarioDifficulty, amount: float = 0.10) -> ScenarioDifficulty:
        amount = clamp(amount, 0.0, 1.0)
        stage = CurriculumStage.ADVERSARIAL if difficulty.stage is CurriculumStage.DISRUPTED else difficulty.stage
        return ScenarioDifficulty(
            stage=stage,
            disruption_probability=clamp(difficulty.disruption_probability + amount, 0.0, 1.0),
            max_disruption_severity=clamp(difficulty.max_disruption_severity + amount, 0.0, 1.0),
            demand_multiplier=max(1.0, difficulty.demand_multiplier + amount),
            capacity_loss_fraction=clamp(difficulty.capacity_loss_fraction + amount, 0.0, 1.0),
            metadata={**difficulty.metadata, "escalated_by": amount},
        )

    def build_disruptions(
        self,
        *,
        difficulty: ScenarioDifficulty,
        target_node_ids: list[UUID],
        start_time: float,
        interval: float,
    ) -> list[DisruptionEvent]:
        if not target_node_ids:
            return []
        events: list[DisruptionEvent] = []
        for index, target_id in enumerate(target_node_ids):
            severity = min(difficulty.max_disruption_severity, 0.1 + difficulty.max_disruption_severity * (index + 1) / len(target_node_ids))
            kind = DisruptionKind.CAPACITY_LOSS if index % 2 == 0 else DisruptionKind.DEMAND_SHOCK
            events.append(
                DisruptionEvent(
                    event_id=uuid4(),
                    kind=kind,
                    starts_at=start_time + index * interval,
                    duration=interval,
                    severity=severity,
                    target_id=target_id,
                    payload={"stage": difficulty.stage.value, "demand_multiplier": difficulty.demand_multiplier},
                )
            )
        return events


ManualCurriculumStage = Literal[
    "normal_v4",
    "high_holding_cost",
    "vehicle_scarcity",
    "route_disruption",
    "premium_sla",
    "demand_spike",
    "lead_time_delay",
    "mixed_stress",
]

REAL_WORLD_CURRICULUM_STAGES: tuple[ManualCurriculumStage, ...] = (
    "normal_v4",
    "high_holding_cost",
    "vehicle_scarcity",
    "route_disruption",
    "premium_sla",
    "demand_spike",
    "lead_time_delay",
    "mixed_stress",
)

RANDOMIZATION_PROFILES: tuple[str, ...] = (
    "fixed_medium",
    "low",
    "medium_single_stress",
    "high_late_stress",
)


@dataclass(frozen=True, slots=True)
class CurriculumRange:
    """Provisional curriculum design range for one scenario override knob."""

    low: float
    medium: float
    high: float

    def __post_init__(self) -> None:
        for name, value in {"low": self.low, "medium": self.medium, "high": self.high}.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"curriculum range {name} value must be finite.")
        ascending = self.low <= self.medium <= self.high
        descending = self.low >= self.medium >= self.high
        if not (ascending or descending):
            raise ValueError("curriculum range values must be ordered low->medium->high or high->medium->low.")

    def fixed_medium_value(self) -> float:
        return float(self.medium)

    def sample(self, rng: random.Random, *, profile: str) -> float:
        if profile == "fixed_medium":
            return self.fixed_medium_value()
        if profile == "low":
            return _uniform_between(rng, self.low, self.medium)
        if profile == "medium_single_stress":
            lower = (self.low + self.medium) / 2.0
            upper = (self.medium + self.high) / 2.0
            return _uniform_between(rng, lower, upper)
        if profile == "high_late_stress":
            return _uniform_between(rng, self.medium, self.high)
        raise ValueError(f"unknown curriculum randomization profile: {profile!r}")


CURRICULUM_RANDOMIZATION_RANGES: dict[str, CurriculumRange] = {
    "holding_cost_multiplier": CurriculumRange(1.0, 1.5, 2.5),
    "excess_inventory_penalty_multiplier": CurriculumRange(1.0, 1.5, 2.5),
    "planned_replenishment_cost_multiplier": CurriculumRange(1.0, 1.3, 2.0),
    "stockout_penalty_multiplier": CurriculumRange(1.0, 1.5, 2.5),
    "inventory_shortfall_penalty_multiplier": CurriculumRange(1.0, 1.4, 2.5),
    "vehicle_availability_multiplier": CurriculumRange(1.0, 0.75, 0.50),
    "fleet_capacity_multiplier": CurriculumRange(1.0, 0.80, 0.60),
    "capacity_shock_severity": CurriculumRange(0.0, 0.25, 0.50),
    "route_disruption_probability": CurriculumRange(0.0, 0.25, 0.60),
    "congestion_multiplier": CurriculumRange(1.0, 1.75, 3.0),
    "shortest_route_risk_multiplier": CurriculumRange(1.0, 1.5, 2.5),
    "traversal_cost_multiplier": CurriculumRange(1.0, 1.3, 1.8),
    "disruption_risk_multiplier": CurriculumRange(1.0, 1.5, 2.0),
    "premium_sla_ratio": CurriculumRange(0.0, 0.40, 0.80),
    "urgent_due_window_multiplier": CurriculumRange(1.0, 0.75, 0.50),
    "premium_lateness_penalty_multiplier": CurriculumRange(1.0, 1.5, 2.5),
    "supplier_delay_probability": CurriculumRange(0.0, 0.25, 0.50),
    "lead_time_mean_multiplier": CurriculumRange(1.0, 1.5, 2.0),
    "lead_time_variance_multiplier": CurriculumRange(1.0, 2.0, 3.0),
    "rolling_demand_probability_multiplier": CurriculumRange(1.0, 1.5, 2.0),
    "rolling_demand_volume_multiplier": CurriculumRange(1.0, 1.5, 2.0),
    "order_units_multiplier": CurriculumRange(1.0, 1.3, 1.6),
    "urgent_order_probability_multiplier": CurriculumRange(1.0, 1.4, 1.8),
}

STAGE_OVERRIDE_KEYS: dict[ManualCurriculumStage, tuple[str, ...]] = {
    "normal_v4": (),
    "high_holding_cost": (
        "holding_cost_multiplier",
        "excess_inventory_penalty_multiplier",
        "planned_replenishment_cost_multiplier",
    ),
    "vehicle_scarcity": (
        "vehicle_availability_multiplier",
        "fleet_capacity_multiplier",
        "capacity_shock_severity",
    ),
    "route_disruption": (
        "route_disruption_probability",
        "congestion_multiplier",
        "shortest_route_risk_multiplier",
        "traversal_cost_multiplier",
        "disruption_risk_multiplier",
    ),
    "premium_sla": (
        "premium_sla_ratio",
        "urgent_due_window_multiplier",
        "premium_lateness_penalty_multiplier",
    ),
    "demand_spike": (
        "rolling_demand_probability_multiplier",
        "rolling_demand_volume_multiplier",
        "order_units_multiplier",
        "urgent_order_probability_multiplier",
    ),
    "lead_time_delay": (
        "supplier_delay_probability",
        "lead_time_mean_multiplier",
        "lead_time_variance_multiplier",
        "stockout_penalty_multiplier",
    ),
    "mixed_stress": (
        "holding_cost_multiplier",
        "vehicle_availability_multiplier",
        "fleet_capacity_multiplier",
        "capacity_shock_severity",
        "route_disruption_probability",
        "congestion_multiplier",
        "premium_sla_ratio",
        "supplier_delay_probability",
        "lead_time_mean_multiplier",
        "rolling_demand_probability_multiplier",
        "rolling_demand_volume_multiplier",
    ),
}


@dataclass(frozen=True, slots=True)
class CurriculumStageScheduleEntry:
    name: ManualCurriculumStage
    steps: int


@dataclass(frozen=True, slots=True)
class CurriculumRandomizationConfig:
    enabled: bool = False
    ranges_profile: str = "fixed_medium"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> CurriculumRandomizationConfig:
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ValueError("curriculum.randomization must be an object.")
        enabled = bool(payload.get("enabled", False))
        profile = str(payload.get("ranges_profile", "medium_single_stress" if enabled else "fixed_medium"))
        if profile not in RANDOMIZATION_PROFILES:
            raise ValueError(
                f"curriculum.randomization.ranges_profile must be one of {', '.join(RANDOMIZATION_PROFILES)}."
            )
        return cls(enabled=enabled, ranges_profile=profile)


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    """Manual-stage curriculum config for v4 clean-cold-start training."""

    enabled: bool = False
    stage: ManualCurriculumStage = "normal_v4"
    stage_schedule: tuple[CurriculumStageScheduleEntry, ...] = ()
    randomization: CurriculumRandomizationConfig = field(default_factory=CurriculumRandomizationConfig)
    seed: int = 0
    baseline_anchor_probability: float = 0.0
    explicit_overrides: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def disabled(cls, *, seed: int = 0) -> CurriculumConfig:
        return cls(enabled=False, seed=int(seed))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None, *, fallback_seed: int = 0) -> CurriculumConfig:
        if payload is None:
            return cls.disabled(seed=fallback_seed)
        if not isinstance(payload, Mapping):
            raise ValueError("curriculum section must be an object.")

        enabled = bool(payload.get("enabled", False))
        stage = _parse_stage(payload.get("stage", "normal_v4"))
        schedule = _parse_stage_schedule(payload.get("stage_schedule", ()))
        randomization = CurriculumRandomizationConfig.from_mapping(payload.get("randomization"))
        seed = int(payload.get("seed", fallback_seed))
        baseline_anchor_probability = float(payload.get("baseline_anchor_probability", 0.0))
        if not math.isfinite(baseline_anchor_probability) or not 0.0 <= baseline_anchor_probability <= 1.0:
            raise ValueError("curriculum.baseline_anchor_probability must be between 0 and 1.")
        explicit_overrides = _parse_explicit_overrides(payload.get("explicit_overrides"), stage=stage)
        return cls(
            enabled=enabled,
            stage=stage,
            stage_schedule=schedule,
            randomization=randomization,
            seed=seed,
            baseline_anchor_probability=baseline_anchor_probability,
            explicit_overrides=explicit_overrides,
        )


@dataclass(frozen=True, slots=True)
class CurriculumSample:
    enabled: bool
    stage: ManualCurriculumStage
    effective_stage: ManualCurriculumStage
    seed: int
    episode_index: int
    baseline_anchor_used: bool
    environment_overrides: dict[str, float]
    capability_matrix: tuple[dict[str, str], ...] = ()

    def with_capability_matrix(self, matrix: list[dict[str, str]]) -> CurriculumSample:
        return CurriculumSample(
            enabled=self.enabled,
            stage=self.stage,
            effective_stage=self.effective_stage,
            seed=self.seed,
            episode_index=self.episode_index,
            baseline_anchor_used=self.baseline_anchor_used,
            environment_overrides=dict(self.environment_overrides),
            capability_matrix=tuple(dict(item) for item in matrix),
        )

    def as_trace_info(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "stage": self.stage,
            "effective_stage": self.effective_stage,
            "seed": self.seed,
            "episode_index": self.episode_index,
            "baseline_anchor_used": self.baseline_anchor_used,
            "environment_overrides": dict(sorted(self.environment_overrides.items())),
            "capability_matrix": [dict(item) for item in self.capability_matrix],
        }

    def as_metrics_fields(self) -> dict[str, Any]:
        return {
            "curriculum_enabled": self.enabled,
            "curriculum_stage": self.stage,
            "curriculum_effective_stage": self.effective_stage,
            "curriculum_seed": self.seed,
            "curriculum_episode_index": self.episode_index,
            "baseline_anchor_used": self.baseline_anchor_used,
            "sampled_stress_overrides": dict(sorted(self.environment_overrides.items())),
        }


class CurriculumSampler:
    """Deterministically sample scenario overrides for the active curriculum stage."""

    def __init__(self, config: CurriculumConfig, *, total_timesteps: int | None = None) -> None:
        self.config = config
        if total_timesteps is not None and int(total_timesteps) <= 0:
            raise ValueError("total_timesteps must be positive when provided.")
        self.total_timesteps = int(total_timesteps) if total_timesteps is not None else None

    @property
    def stage_progression(self) -> str:
        return "scheduled" if self._uses_stage_schedule else "manual"

    def sample(self, episode_index: int, *, global_step: int | None = None) -> CurriculumSample:
        if episode_index < 0:
            raise ValueError("episode_index must not be negative.")
        if not self.config.enabled:
            return CurriculumSample(
                enabled=False,
                stage=self.config.stage,
                effective_stage="normal_v4",
                seed=self.config.seed,
                episode_index=episode_index,
                baseline_anchor_used=False,
                environment_overrides={},
            )

        stage = self._stage_for_sample(global_step=global_step)
        rng = random.Random(_stable_episode_seed(self.config.seed, stage, episode_index))
        baseline_anchor_used = (
            stage != "normal_v4"
            and self.config.baseline_anchor_probability > 0.0
            and rng.random() < self.config.baseline_anchor_probability
        )
        effective_stage: ManualCurriculumStage = "normal_v4" if baseline_anchor_used else stage
        overrides = self._overrides_for_stage(effective_stage, rng)
        return CurriculumSample(
            enabled=True,
            stage=stage,
            effective_stage=effective_stage,
            seed=self.config.seed,
            episode_index=episode_index,
            baseline_anchor_used=baseline_anchor_used,
            environment_overrides=overrides,
        )

    @property
    def _uses_stage_schedule(self) -> bool:
        return bool(
            self.config.enabled
            and self.config.stage == "normal_v4"
            and self.config.stage_schedule
        )

    def _stage_for_sample(self, *, global_step: int | None) -> ManualCurriculumStage:
        if not self._uses_stage_schedule:
            return self.config.stage
        if global_step is None:
            raise ValueError("global_step is required when curriculum.stage_schedule is active.")

        step = max(int(global_step), 0)
        schedule_total = sum(entry.steps for entry in self.config.stage_schedule)
        if schedule_total <= 0:
            return self.config.stage
        if self.total_timesteps is not None:
            progress = min(step, self.total_timesteps - 1) / max(float(self.total_timesteps), 1.0)
            schedule_position = progress * schedule_total
        else:
            schedule_position = min(float(step), float(schedule_total - 1))

        cursor = 0.0
        for entry in self.config.stage_schedule:
            cursor += float(entry.steps)
            if schedule_position < cursor:
                return entry.name
        return self.config.stage_schedule[-1].name

    def _overrides_for_stage(self, stage: ManualCurriculumStage, rng: random.Random) -> dict[str, float]:
        keys = STAGE_OVERRIDE_KEYS[stage]
        if not keys:
            return {}
        if not self.config.randomization.enabled:
            values = {key: CURRICULUM_RANDOMIZATION_RANGES[key].fixed_medium_value() for key in keys}
            values.update(self.config.explicit_overrides)
            return values
        profile = self.config.randomization.ranges_profile
        values = {
            key: CURRICULUM_RANDOMIZATION_RANGES[key].sample(rng, profile=profile)
            for key in keys
        }
        values.update(self.config.explicit_overrides)
        if stage == "mixed_stress":
            values = _cap_mixed_stress_severity(values)
        return values


def curriculum_config_from_training_config(
    training_config: Mapping[str, Any],
    *,
    fallback_seed: int = 0,
) -> CurriculumConfig:
    """Parse optional training config curriculum section with disabled default behavior."""

    return CurriculumConfig.from_mapping(training_config.get("curriculum"), fallback_seed=fallback_seed)


def _parse_stage(value: Any) -> ManualCurriculumStage:
    stage = str(value)
    if stage not in REAL_WORLD_CURRICULUM_STAGES:
        raise ValueError(
            f"unknown curriculum stage {stage!r}; expected one of {', '.join(REAL_WORLD_CURRICULUM_STAGES)}."
        )
    return stage  # type: ignore[return-value]


def _parse_stage_schedule(value: Any) -> tuple[CurriculumStageScheduleEntry, ...]:
    if value in (None, ()):
        return ()
    if not isinstance(value, list):
        raise ValueError("curriculum.stage_schedule must be a list when provided.")
    entries: list[CurriculumStageScheduleEntry] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"curriculum.stage_schedule[{index}] must be an object.")
        name = _parse_stage(item.get("name", item.get("stage")))
        steps = int(item.get("steps", 0))
        if steps <= 0:
            raise ValueError(f"curriculum.stage_schedule[{index}].steps must be a positive integer.")
        entries.append(CurriculumStageScheduleEntry(name=name, steps=steps))
    return tuple(entries)


def _parse_explicit_overrides(
    value: Any,
    *,
    stage: ManualCurriculumStage,
) -> dict[str, float]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("curriculum.explicit_overrides must be an object when provided.")
    allowed_keys = set(STAGE_OVERRIDE_KEYS[stage])
    parsed: dict[str, float] = {}
    for key, raw_value in value.items():
        key = str(key)
        if key not in allowed_keys:
            raise ValueError(
                f"curriculum.explicit_overrides contains unsupported key {key!r} for stage {stage!r}."
            )
        override_value = float(raw_value)
        if not math.isfinite(override_value):
            raise ValueError(f"curriculum.explicit_overrides[{key!r}] must be finite.")
        parsed[key] = override_value
    return parsed


def _stable_episode_seed(seed: int, stage: str, episode_index: int) -> int:
    payload = f"{int(seed)}:{stage}:{int(episode_index)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _uniform_between(rng: random.Random, left: float, right: float) -> float:
    low = min(float(left), float(right))
    high = max(float(left), float(right))
    return rng.uniform(low, high)


def _cap_mixed_stress_severity(values: Mapping[str, float]) -> dict[str, float]:
    """Keep mixed stress moderate for first curriculum infrastructure package."""

    capped = dict(values)
    limits = {
        "holding_cost_multiplier": 1.75,
        "vehicle_availability_multiplier": 0.70,
        "fleet_capacity_multiplier": 0.70,
        "capacity_shock_severity": 0.35,
        "route_disruption_probability": 0.35,
        "congestion_multiplier": 2.0,
        "premium_sla_ratio": 0.50,
        "supplier_delay_probability": 0.35,
        "lead_time_mean_multiplier": 1.6,
        "rolling_demand_probability_multiplier": 1.6,
        "rolling_demand_volume_multiplier": 1.6,
    }
    for key, limit in limits.items():
        if key not in capped:
            continue
        if key in {"vehicle_availability_multiplier", "fleet_capacity_multiplier"}:
            capped[key] = max(float(capped[key]), limit)
        else:
            capped[key] = min(float(capped[key]), limit)
    return capped
