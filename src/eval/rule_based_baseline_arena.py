"""Benchmark runner for deterministic rule-based logistics baselines.

This adapter evaluates the pure policies in :mod:`src.eval.rule_based_baselines`
inside the existing 5PL scenario arena. It does not load model checkpoints and
only writes to a fresh caller-provided benchmark output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from src.act.action_projector import CONTINUOUS_ACTION_DIM, ActionProjector
from src.act.discrete_action_mapper import DISCRETE_ACTION_COUNT
from src.eval.real_world_scenario_arena import (
    ScenarioConfig,
    build_scenario_environment,
    load_scenario_configs,
)
from src.eval.rule_based_baselines import (
    OrderView,
    RuleBasedBaseline,
    RuleBasedDecision,
    RuleBasedDecisionContext,
    build_rule_based_baselines,
)
from src.eval.scenario_metrics import EpisodeMetricAccumulator, EpisodeMetrics, summarize_episodes
from src.learn.train_joint_torch import MDP_CONTRACT_VERSION


DEFAULT_BASE_TRAINING_CONFIG: dict[str, Any] = {
    "mdp_contract_version": MDP_CONTRACT_VERSION,
    "shared_global_parameters": {
        "max_steps": 288,
        "decision_interval_seconds": 300.0,
        "max_extra_dispatch_budget": 4,
        "feasibility_gated_macro_budget": True,
        "observation_dim": 73,
    },
    "reward_physics": {},
}

PROTECTED_OUTPUT_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("models", "registry"),
    ("models", "production"),
    ("models", "baselines"),
    ("models", "checkpoints"),
    ("models", "eval"),
    ("db",),
)


EnvFactory = Callable[..., tuple[Any, Any]]
ContinuousActionProvider = Callable[[RuleBasedDecisionContext | None], Sequence[float] | np.ndarray]


class ContinuousBaselineMode(StrEnum):
    NEUTRAL = "neutral_continuous"
    HEURISTIC = "heuristic_continuous"
    PPO_ASSISTED = "ppo_assisted_continuous"
    ORACLE_DIAGNOSTIC = "oracle_diagnostic_continuous"


def neutral_continuous_action() -> np.ndarray:
    """Return the existing fixed normalized PPO vector for rule baselines.

    Source-truth caveat: normalized zeros are not a physical no-op. With the
    default :class:`ActionProjector`, this projects to midpoint physical
    controls: reorder_fraction=0.5, dispatch_intensity=0.5,
    speed_multiplier=1.0, safety_stock_multiplier=1.5, and
    capacity_buffer_fraction=0.25.
    """

    return np.zeros((CONTINUOUS_ACTION_DIM,), dtype=np.float32)


def continuous_mode_metadata(mode: ContinuousBaselineMode | str) -> dict[str, Any]:
    mode = _coerce_continuous_mode(mode)
    metadata = {
        ContinuousBaselineMode.NEUTRAL: {
            "deployable": True,
            "source": "fixed_zero_normalized",
            "description": "Backwards-compatible fixed normalized midpoint continuous controls.",
        },
        ContinuousBaselineMode.HEURISTIC: {
            "deployable": True,
            "source": "heuristic_current_state_only",
            "description": "Deterministic continuous controls from current rule context signals only.",
        },
        ContinuousBaselineMode.PPO_ASSISTED: {
            "deployable": True,
            "source": "injected_ppo_provider_read_only",
            "description": "Injected learned-continuous provider paired with rule tactical actions.",
        },
        ContinuousBaselineMode.ORACLE_DIAGNOSTIC: {
            "deployable": False,
            "source": "oracle_hindsight_non_deployable",
            "description": "Non-deployable diagnostic mode requiring explicit opt-in.",
        },
    }[mode]
    return {"mode": mode.value, **metadata}


def build_continuous_action(
    *,
    mode: ContinuousBaselineMode | str = ContinuousBaselineMode.NEUTRAL,
    context: RuleBasedDecisionContext | None = None,
    provider_input: Any | None = None,
    continuous_provider: ContinuousActionProvider | None = None,
    allow_oracle_diagnostic: bool = False,
    action_projector: ActionProjector | None = None,
) -> np.ndarray:
    mode = _coerce_continuous_mode(mode)
    if mode is ContinuousBaselineMode.NEUTRAL:
        return neutral_continuous_action()
    if mode is ContinuousBaselineMode.HEURISTIC:
        if context is None:
            raise ValueError("heuristic_continuous requires a rule decision context.")
        return _heuristic_continuous_action(context, action_projector=action_projector)
    if mode is ContinuousBaselineMode.PPO_ASSISTED:
        if continuous_provider is None:
            raise ValueError("ppo_assisted_continuous requires an injected continuous provider.")
        return _validate_continuous_action(
            continuous_provider(provider_input if provider_input is not None else context),
            source=mode.value,
        )
    if mode is ContinuousBaselineMode.ORACLE_DIAGNOSTIC:
        if not allow_oracle_diagnostic:
            raise ValueError("oracle_diagnostic_continuous is non-deployable and requires explicit allow.")
        if context is None:
            return neutral_continuous_action()
        return _heuristic_continuous_action(context, action_projector=action_projector)
    raise ValueError(f"unsupported continuous baseline mode: {mode}")


def validate_rule_action_id(action_id: int) -> int:
    action_id = int(action_id)
    if not 0 <= action_id < DISCRETE_ACTION_COUNT:
        raise ValueError(f"rule action_id must be in [0, {DISCRETE_ACTION_COUNT - 1}], got {action_id}.")
    return action_id


def build_joint_action_from_rule_decision(
    decision: RuleBasedDecision,
    *,
    continuous_mode: ContinuousBaselineMode | str = ContinuousBaselineMode.NEUTRAL,
    continuous_context: RuleBasedDecisionContext | None = None,
    continuous_provider_input: Any | None = None,
    continuous_provider: ContinuousActionProvider | None = None,
    allow_oracle_diagnostic: bool = False,
) -> dict[str, Any]:
    return {
        "continuous": build_continuous_action(
            mode=continuous_mode,
            context=continuous_context,
            provider_input=continuous_provider_input,
            continuous_provider=continuous_provider,
            allow_oracle_diagnostic=allow_oracle_diagnostic,
        ),
        "discrete": validate_rule_action_id(decision.action_id),
    }


def normalized_action_from_physical_targets(
    *,
    reorder_fraction: float,
    dispatch_intensity: float,
    speed_multiplier: float,
    safety_stock_multiplier: float,
    capacity_buffer_fraction: float,
    action_projector: ActionProjector | None = None,
) -> np.ndarray:
    projector = action_projector or ActionProjector()
    speed_delta = max(projector.max_speed_delta_fraction, 1.0e-9)
    safety_range = max(projector.max_safety_stock_multiplier - 1.0, 1.0e-9)
    capacity_max = max(projector.max_capacity_buffer_fraction, 1.0e-9)
    vector = np.array(
        [
            (_bounded_value(reorder_fraction, default=0.0) * 2.0) - 1.0,
            (_bounded_value(dispatch_intensity, default=0.0) * 2.0) - 1.0,
            (clamp_float(speed_multiplier, 1.0 - projector.max_speed_delta_fraction, 1.0 + projector.max_speed_delta_fraction) - 1.0)
            / speed_delta,
            (
                (
                    clamp_float(safety_stock_multiplier, 1.0, projector.max_safety_stock_multiplier)
                    - 1.0
                )
                / safety_range
                * 2.0
            )
            - 1.0,
            (
                clamp_float(capacity_buffer_fraction, 0.0, projector.max_capacity_buffer_fraction)
                / capacity_max
                * 2.0
            )
            - 1.0,
        ],
        dtype=np.float32,
    )
    return _validate_continuous_action(vector, source="physical_targets")


def build_rule_context_from_environment(env: Any) -> RuleBasedDecisionContext:
    simulation = getattr(env, "simulation", None)
    current_time = _current_time(simulation)
    orders = tuple(_order_view(order) for order in _orders(simulation) if _is_dispatchable_order(order))
    orders = tuple(order for order in orders if order is not None)
    min_units = min((max(0.0, order.total_units) for order in orders), default=0.0)
    primary_available, secondary_available = _vehicle_availability(simulation, min_units=min_units)
    snapshot = _snapshot(env)
    return RuleBasedDecisionContext(
        dispatchable_orders=orders,
        primary_vehicle_available=primary_available,
        secondary_vehicle_available=secondary_available,
        route_disruption_pressure=_route_disruption_pressure(env, snapshot),
        congestion_pressure=_congestion_pressure(env, snapshot),
        stockout_risk=_bounded_metric(env, "_stockout_risk", snapshot, default=0.0),
        holding_cost_pressure=_bounded_metric(env, "_holding_cost_pressure", snapshot, default=0.0),
        safety_stock_gap=_bounded_metric(env, "_safety_stock_target_gap", snapshot, default=0.0),
        inventory_coverage=_bounded_metric(env, "_inventory_coverage", snapshot, default=1.0),
    )


def run_rule_baseline_episode(
    scenario: ScenarioConfig,
    baseline: RuleBasedBaseline,
    *,
    seed: int,
    episode_index: int,
    base_training_config: Mapping[str, Any] | None = None,
    max_steps: int | None = None,
    env_factory: EnvFactory | None = None,
    continuous_mode: ContinuousBaselineMode | str = ContinuousBaselineMode.NEUTRAL,
    continuous_provider: ContinuousActionProvider | None = None,
    allow_oracle_diagnostic: bool = False,
) -> EpisodeMetrics:
    factory = env_factory or _default_env_factory
    env, _capabilities = factory(
        base_training_config=base_training_config or DEFAULT_BASE_TRAINING_CONFIG,
        scenario=scenario,
        seed=seed,
        max_steps=max_steps,
    )
    try:
        observation, _reset_info = env.reset(seed=seed)
        accumulator = EpisodeMetricAccumulator(
            scenario_id=scenario.scenario_id,
            seed=seed,
            episode_index=episode_index,
        )
        terminated = False
        truncated = False
        while not (terminated or truncated):
            context = build_rule_context_from_environment(env)
            decision = baseline.select(context)
            action = build_joint_action_from_rule_decision(
                decision,
                continuous_mode=continuous_mode,
                continuous_context=context,
                continuous_provider_input=observation,
                continuous_provider=continuous_provider,
                allow_oracle_diagnostic=allow_oracle_diagnostic,
            )
            observation, reward, terminated, truncated, info = env.step(action)
            accumulator.observe(reward=float(reward), info=info)
        return accumulator.finish()
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def run_rule_based_benchmark(
    *,
    scenario_dir: Path,
    output_dir: Path,
    episodes: int = 20,
    seed: int = 42,
    baseline_names: Sequence[str] | None = None,
    max_steps: int | None = None,
    base_training_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scenarios = load_scenario_configs(scenario_dir)
    baselines = build_rule_based_baselines()
    selected_names = list(baseline_names) if baseline_names else sorted(baselines)
    unknown = sorted(set(selected_names) - set(baselines))
    if unknown:
        raise ValueError(f"unknown rule baseline names: {', '.join(unknown)}")

    episode_rows: list[tuple[str, EpisodeMetrics]] = []
    baseline_summaries: list[dict[str, Any]] = []
    for baseline_name in selected_names:
        baseline = baselines[baseline_name]
        for scenario in scenarios:
            metrics: list[EpisodeMetrics] = []
            for episode_index in range(int(episodes)):
                episode_seed = _episode_seed(seed, scenario, episode_index)
                metric = run_rule_baseline_episode(
                    scenario,
                    baseline,
                    seed=episode_seed,
                    episode_index=episode_index,
                    base_training_config=base_training_config,
                    max_steps=max_steps,
                )
                metrics.append(metric)
                episode_rows.append((baseline_name, metric))
            summary = summarize_episodes(scenario.scenario_id, metrics)
            summary["baseline_name"] = baseline_name
            summary["episodes"] = int(episodes)
            summary["description"] = baseline.description
            summary["secondary_fleet_rate"] = _rate_from_distribution(summary, "fleet_distribution", "secondary_fleet")
            summary["reorder_none_rate"] = _rate_from_distribution(summary, "reorder_mode_distribution", "none")
            baseline_summaries.append(summary)

    report = write_rule_benchmark_outputs(
        output_dir=output_dir,
        benchmark_metadata={
            "benchmark": "rule_based_baselines",
            "scenario_dir": str(scenario_dir),
            "episodes_per_scenario": int(episodes),
            "seed": int(seed),
            "max_steps": max_steps,
            "baselines": selected_names,
            "contract": MDP_CONTRACT_VERSION,
            "obs_dim": 73,
            "continuous_action_dim": CONTINUOUS_ACTION_DIM,
            "discrete_action_count": DISCRETE_ACTION_COUNT,
        },
        baseline_summaries=baseline_summaries,
        episode_metrics=episode_rows,
        comparison_to_hierarchical=_compare_to_hierarchical_equal_budget(baseline_summaries),
    )
    return report


def ensure_fresh_benchmark_output_dir(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    _reject_protected_output_dir(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"benchmark output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)


def write_rule_benchmark_outputs(
    *,
    output_dir: Path,
    benchmark_metadata: Mapping[str, Any],
    baseline_summaries: Sequence[Mapping[str, Any]],
    episode_metrics: Sequence[tuple[str, EpisodeMetrics]],
    comparison_to_hierarchical: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    ensure_fresh_benchmark_output_dir(output_dir)

    decision = _benchmark_decision(baseline_summaries)
    report = {
        "benchmark_metadata": dict(benchmark_metadata),
        "decision": decision,
        "baseline_summaries": [_jsonable(summary) for summary in baseline_summaries],
        "comparison_to_hierarchical_v1": [_jsonable(row) for row in comparison_to_hierarchical],
    }
    (output_dir / "rule_based_baseline_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_summary_csv(output_dir / "rule_based_baseline_summary.csv", baseline_summaries)
    _write_episode_jsonl(output_dir / "rule_based_baseline_episode_metrics.jsonl", episode_metrics)
    _write_markdown_report(output_dir / "rule_based_baseline_report.md", report)
    return report


def _default_env_factory(**kwargs: Any) -> tuple[Any, Any]:
    return build_scenario_environment(
        kwargs["base_training_config"],
        kwargs["scenario"],
        seed=int(kwargs["seed"]),
        max_steps=kwargs.get("max_steps"),
    )


def _coerce_continuous_mode(mode: ContinuousBaselineMode | str) -> ContinuousBaselineMode:
    if isinstance(mode, ContinuousBaselineMode):
        return mode
    try:
        return ContinuousBaselineMode(str(mode))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ContinuousBaselineMode)
        raise ValueError(f"continuous baseline mode must be one of: {allowed}") from exc


def _validate_continuous_action(action: Sequence[float] | np.ndarray, *, source: str) -> np.ndarray:
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    if vector.shape != (CONTINUOUS_ACTION_DIM,):
        raise ValueError(
            f"{source} continuous action must have shape ({CONTINUOUS_ACTION_DIM},), got {vector.shape}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{source} continuous action contains non-finite values.")
    if np.any(vector < -1.0) or np.any(vector > 1.0):
        raise ValueError(f"{source} continuous action must stay in normalized [-1, 1] bounds.")
    return vector.astype(np.float32, copy=False)


def _heuristic_continuous_action(
    context: Any,
    *,
    action_projector: ActionProjector | None = None,
) -> np.ndarray:
    projector = action_projector or ActionProjector()
    orders = _context_dispatchable_orders(context)
    stockout_risk = _context_metric(context, "stockout_risk", default=0.0)
    safety_stock_gap = _context_metric(context, "safety_stock_gap", default=0.0)
    inventory_coverage = _context_metric(context, "inventory_coverage", default=1.0)
    lead_time_pressure = _bounded_value(
        max(
            _context_metric(context, "lead_time_volatility_pressure", default=0.0),
            _context_metric(context, "supplier_delay_pressure", default=0.0),
            _context_metric(context, "scenario_lead_time_volatility_pressure", default=0.0),
        ),
        default=0.0,
    )
    inventory_pressure = _bounded_value(
        max(
            stockout_risk,
            safety_stock_gap,
            1.0 - inventory_coverage,
            lead_time_pressure,
        ),
        default=0.0,
    )
    holding_pressure = _context_metric(context, "holding_cost_pressure", default=0.0)
    route_pressure = _bounded_value(
        max(
            _context_metric(context, "route_disruption_pressure", default=0.0),
            _context_metric(context, "congestion_pressure", default=0.0),
        ),
        default=0.0,
    )
    order_count = len(orders)
    order_pressure = _bounded_value(order_count / 5.0, default=0.0)
    urgent_count = sum(1 for order in orders if getattr(order, "urgent", False) or getattr(order, "premium", False))
    urgent_ratio = _bounded_value(urgent_count / max(order_count, 1), default=0.0)
    flow_pressure = _bounded_value(
        max(
            order_pressure,
            urgent_ratio,
            _context_metric(context, "pending_work_pressure", default=0.0),
            _context_metric(context, "lateness_risk", default=0.0),
            _context_metric(context, "premium_sla_pressure", default=0.0),
            _context_metric(context, "useful_dispatch_opportunity", default=0.0),
        ),
        default=0.0,
    )

    primary_vehicle_available = _context_bool(context, "primary_vehicle_available")
    secondary_vehicle_available = _context_bool(context, "secondary_vehicle_available")
    explicit_feasible_dispatch_ratio = getattr(context, "feasible_dispatch_ratio", None)
    if explicit_feasible_dispatch_ratio is not None:
        feasible_dispatch_ratio = _bounded_value(explicit_feasible_dispatch_ratio, default=0.0)
    elif primary_vehicle_available:
        feasible_dispatch_ratio = 1.0
    elif secondary_vehicle_available:
        feasible_dispatch_ratio = 0.70
    else:
        feasible_dispatch_ratio = 0.0

    if primary_vehicle_available and secondary_vehicle_available:
        vehicle_scarcity_pressure = 0.0
    elif primary_vehicle_available or secondary_vehicle_available:
        vehicle_scarcity_pressure = 0.65
    else:
        vehicle_scarcity_pressure = 1.0
    scarcity_pressure = _bounded_value(
        max(
            vehicle_scarcity_pressure,
            _context_metric(context, "vehicle_availability_pressure", default=0.0),
            _context_metric(context, "capacity_shock_pressure", default=0.0),
        ),
        default=0.0,
    )

    reorder_fraction = _bounded_value(
        inventory_pressure * (1.0 - (0.60 * holding_pressure)),
        default=0.0,
    )
    dispatch_intensity = _bounded_value(flow_pressure * feasible_dispatch_ratio, default=0.0)
    speed_multiplier = clamp_float(
        1.0 + (0.25 * max(route_pressure, urgent_ratio)),
        1.0 - projector.max_speed_delta_fraction,
        1.0 + projector.max_speed_delta_fraction,
    )
    safety_stock_multiplier = clamp_float(
        1.0
        + (
            inventory_pressure
            * (1.0 - (0.50 * holding_pressure))
            * (projector.max_safety_stock_multiplier - 1.0)
        ),
        1.0,
        projector.max_safety_stock_multiplier,
    )
    capacity_buffer_fraction = projector.max_capacity_buffer_fraction * _bounded_value(
        max(scarcity_pressure, route_pressure, flow_pressure * 0.50),
        default=0.0,
    )

    return normalized_action_from_physical_targets(
        reorder_fraction=reorder_fraction,
        dispatch_intensity=dispatch_intensity,
        speed_multiplier=speed_multiplier,
        safety_stock_multiplier=safety_stock_multiplier,
        capacity_buffer_fraction=capacity_buffer_fraction,
        action_projector=projector,
    )


def _context_dispatchable_orders(context: Any) -> tuple[Any, ...]:
    raw_orders = getattr(context, "dispatchable_orders", ())
    if raw_orders is None:
        return ()
    if isinstance(raw_orders, Sequence) and not isinstance(raw_orders, (str, bytes)):
        return tuple(raw_orders)
    return ()


def _context_bool(context: Any, name: str) -> bool:
    return bool(getattr(context, name, False))


def _context_metric(context: Any, name: str, *, default: float) -> float:
    value = getattr(context, name, default)
    if value is None:
        return default
    return _bounded_value(value, default=default)


def clamp_float(value: Any, minimum: float, maximum: float) -> float:
    return min(float(maximum), max(float(minimum), _finite_float(value, default=float(minimum))))


def _current_time(simulation: Any) -> float:
    sim_env = getattr(simulation, "env", None)
    return _finite_float(getattr(sim_env, "now", 0.0), default=0.0)


def _orders(simulation: Any) -> list[Any]:
    raw = getattr(simulation, "orders", {})
    if isinstance(raw, Mapping):
        return list(raw.values())
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return list(raw)
    return []


def _vehicles(simulation: Any) -> list[Any]:
    raw = getattr(simulation, "vehicles", {})
    if isinstance(raw, Mapping):
        return list(raw.values())
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return list(raw)
    return []


def _is_dispatchable_order(order: Any) -> bool:
    status = getattr(order, "status", None)
    status_value = getattr(status, "value", status)
    return (
        getattr(order, "assigned_vehicle_id", None) is None
        and getattr(order, "delivery_time", None) is None
        and str(status_value) != "delayed"
    )


def _order_view(order: Any) -> OrderView | None:
    order_id = getattr(order, "order_id", None)
    if order_id is None:
        return None
    metadata = getattr(order, "metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    due_time = _finite_float(getattr(order, "due_time", getattr(order, "time_window_end", 0.0)), default=0.0)
    created_at = _finite_float(getattr(order, "created_at", getattr(order, "release_time", 0.0)), default=0.0)
    total_units = _finite_float(getattr(order, "total_units", getattr(order, "demand_units", 0.0)), default=0.0)
    return OrderView(
        order_id=str(order_id),
        created_at=created_at,
        due_time=due_time,
        total_units=total_units,
        premium=bool(metadata.get("premium", False)),
        urgent=bool(metadata.get("urgent", False)),
    )


def _vehicle_availability(simulation: Any, *, min_units: float) -> tuple[bool, bool]:
    primary_available = False
    secondary_available = False
    for vehicle in _vehicles(simulation):
        if not bool(getattr(vehicle, "active", True)):
            continue
        tier = getattr(vehicle, "tier", "")
        tier_value = str(getattr(tier, "value", tier))
        remaining_units = _vehicle_remaining_units(vehicle)
        if min_units > 0.0 and remaining_units + 1e-9 < min_units:
            continue
        if tier_value == "primary":
            primary_available = True
        elif tier_value == "secondary":
            secondary_available = True
    return primary_available, secondary_available


def _vehicle_remaining_units(vehicle: Any) -> float:
    if hasattr(vehicle, "remaining_units"):
        return _finite_float(getattr(vehicle, "remaining_units"), default=0.0)
    capacity = _finite_float(getattr(vehicle, "capacity_units", 0.0), default=0.0)
    load = _finite_float(getattr(vehicle, "current_load_units", getattr(vehicle, "load_units", 0.0)), default=0.0)
    return max(0.0, capacity - load)


def _snapshot(env: Any) -> Mapping[str, Any]:
    method = getattr(env, "_snapshot", None)
    if callable(method):
        try:
            value = method()
            if isinstance(value, Mapping):
                return value
        except Exception:
            return {}
    return {}


def _route_disruption_pressure(env: Any, snapshot: Mapping[str, Any]) -> float:
    stress = getattr(env, "scenario_stress_state", {})
    stress = stress if isinstance(stress, Mapping) else {}
    values = [
        stress.get("route_disruption_probability"),
        stress.get("route_disruption_pressure"),
        snapshot.get("route_disruption_pressure"),
        snapshot.get("disruption_score"),
    ]
    return max((_bounded_value(value, default=0.0) for value in values), default=0.0)


def _congestion_pressure(env: Any, snapshot: Mapping[str, Any]) -> float:
    stress = getattr(env, "scenario_stress_state", {})
    stress = stress if isinstance(stress, Mapping) else {}
    multiplier_pressure = (_finite_float(stress.get("congestion_multiplier", 1.0), default=1.0) - 1.0) / 3.0
    values = [
        multiplier_pressure,
        stress.get("congestion_pressure"),
        snapshot.get("congestion_pressure"),
    ]
    return max((_bounded_value(value, default=0.0) for value in values), default=0.0)


def _bounded_metric(env: Any, method_name: str, snapshot: Mapping[str, Any], *, default: float) -> float:
    method = getattr(env, method_name, None)
    if callable(method):
        try:
            return _bounded_value(method(snapshot), default=default)
        except Exception:
            return default
    return default


def _bounded_value(value: Any, *, default: float) -> float:
    return min(1.0, max(0.0, _finite_float(value, default=default)))


def _finite_float(value: Any, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _episode_seed(seed: int, scenario: ScenarioConfig, episode_index: int) -> int:
    if episode_index < len(scenario.seeds):
        return int(scenario.seeds[episode_index])
    return int(seed) + episode_index


def _rate_from_distribution(summary: Mapping[str, Any], distribution_key: str, item_key: str) -> float | None:
    steps = _finite_float(summary.get("steps"), default=0.0)
    distribution = summary.get(distribution_key, {})
    if not isinstance(distribution, Mapping) or steps <= 0.0:
        return None
    return _finite_float(distribution.get(item_key, 0.0), default=0.0) / steps


def _benchmark_decision(summaries: Sequence[Mapping[str, Any]]) -> str:
    if not summaries:
        return "RULE_BASED_BASELINE_BENCHMARK_BLOCKED"
    hard_blocker_total = 0
    for summary in summaries:
        blockers = summary.get("hard_blockers", {})
        if isinstance(blockers, Mapping):
            hard_blocker_total += sum(int(value) for value in blockers.values() if isinstance(value, int | float))
    if hard_blocker_total > 0:
        return "RULE_BASED_BASELINE_BENCHMARK_NEEDS_INVESTIGATION"
    return "RULE_BASED_BASELINE_BENCHMARK_READY"


def _compare_to_hierarchical_equal_budget(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    reference_path = Path("models/eval/equal_budget_hierarchical_v1_20ep_seed42_20260611/scenario_summary.json")
    if not reference_path.exists():
        return []
    try:
        reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    reference_rows = reference_payload.get("scenarios", reference_payload)
    if not isinstance(reference_rows, list):
        return []
    by_scenario = {
        str(row.get("scenario_id")): row
        for row in reference_rows
        if isinstance(row, Mapping) and row.get("scenario_id") is not None
    }
    comparison: list[dict[str, Any]] = []
    for summary in summaries:
        scenario_id = str(summary.get("scenario_id"))
        reference = by_scenario.get(scenario_id)
        if not reference:
            continue
        comparison.append(
            {
                "baseline_name": summary.get("baseline_name"),
                "scenario_id": scenario_id,
                "service_delta_vs_hierarchical": _delta(summary, reference, "service_level"),
                "lateness_delta_vs_hierarchical": _delta(summary, reference, "mean_lateness"),
                "dispatch_success_delta_vs_hierarchical": _delta(summary, reference, "dispatch_success_per_attempt"),
            }
        )
    return comparison


def _delta(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> float | None:
    left_value = left.get(key)
    right_value = right.get(key)
    if left_value is None or right_value is None:
        return None
    return _finite_float(left_value, default=0.0) - _finite_float(right_value, default=0.0)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_summary_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "baseline_name",
        "scenario_id",
        "episodes",
        "steps",
        "service_level",
        "mean_lateness",
        "dispatch_rate",
        "dispatch_success_per_attempt",
        "no_vehicle_available",
        "route_failures",
        "action_24_percentage",
        "action_25_percentage",
        "secondary_fleet_rate",
        "reorder_none_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fields})


def _write_episode_jsonl(path: Path, episode_metrics: Sequence[tuple[str, EpisodeMetrics]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for baseline_name, metric in episode_metrics:
            row = {"baseline_name": baseline_name, **_jsonable(metric)}
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_markdown_report(path: Path, report: Mapping[str, Any]) -> None:
    metadata = report.get("benchmark_metadata", {})
    summaries = report.get("baseline_summaries", [])
    lines = [
        "# Rule-Based Baseline Benchmark",
        "",
        f"- Decision: {report.get('decision')}",
        f"- Scenario dir: {metadata.get('scenario_dir') if isinstance(metadata, Mapping) else ''}",
        f"- Episodes per scenario: {metadata.get('episodes_per_scenario') if isinstance(metadata, Mapping) else ''}",
        f"- Seed: {metadata.get('seed') if isinstance(metadata, Mapping) else ''}",
        "",
        "## Scenario Summary",
        "",
        "| Baseline | Scenario | Service | Lateness | Dispatch success | Action 24 % | Secondary fleet rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if isinstance(summaries, Sequence):
        for summary in summaries:
            if not isinstance(summary, Mapping):
                continue
            lines.append(
                "| {baseline} | {scenario} | {service} | {lateness} | {success} | {a24} | {secondary} |".format(
                    baseline=summary.get("baseline_name", ""),
                    scenario=summary.get("scenario_id", ""),
                    service=_fmt(summary.get("service_level")),
                    lateness=_fmt(summary.get("mean_lateness")),
                    success=_fmt(summary.get("dispatch_success_per_attempt")),
                    a24=_fmt(summary.get("action_24_percentage")),
                    secondary=_fmt(summary.get("secondary_fleet_rate")),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _reject_protected_output_dir(output_dir: Path) -> None:
    candidate = Path(output_dir).resolve(strict=False)
    repo_root = Path(__file__).resolve().parents[2]
    for prefix in PROTECTED_OUTPUT_PREFIXES:
        protected_root = repo_root.joinpath(*prefix).resolve(strict=False)
        if candidate == protected_root or protected_root in candidate.parents:
            raise ValueError(f"refusing to write benchmark output under protected path: {output_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic rule-based baseline benchmarks.")
    parser.add_argument("--scenario-dir", default="configs/eval_scenarios")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--baselines", nargs="*", default=None)
    args = parser.parse_args(argv)

    report = run_rule_based_benchmark(
        scenario_dir=Path(args.scenario_dir),
        output_dir=Path(args.output_dir),
        episodes=args.episodes,
        seed=args.seed,
        baseline_names=args.baselines,
        max_steps=args.max_steps,
    )
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
