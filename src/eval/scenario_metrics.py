"""Metrics and pass/fail evaluation for real-world scenario runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Any, Mapping


JsonDict = dict[str, Any]
EPSILON = 1e-9
STALE_ASSIGNED_ORDER_SECONDS = 28_800.0
ROUTE_LABELS = ("shortest", "low_congestion", "high_resilience")
ROUTE_SCORE_COMPONENTS: dict[str, str] = {
    "shortest": "shortest_route_candidate_score",
    "low_congestion": "low_congestion_route_candidate_score",
    "high_resilience": "high_resilience_route_candidate_score",
}
REORDER_MODES = ("none", "conservative", "aggressive", "emergency")
DIAGNOSTIC_REWARD_COMPONENTS: tuple[str, ...] = (
    "dqn_local",
    "hold_under_lateness_pressure",
    "hold_under_lateness_pressure_penalty",
    "hold_timing_degradation_pressure",
    "useful_dispatch_lateness_urgency_credit",
    "primary_fleet_timing_justification_credit",
    "secondary_fleet_lateness_exposure",
    "no_current_or_unassigned_dispatch_penalty",
    "dispatch_feasibility_credit",
    "dispatch_progress_credit",
    "route_resilience_credit",
    "route_resilience_adaptation_credit",
    "low_congestion_no_useful_work_credit_blocked",
    "high_resilience_no_useful_work_credit_blocked",
    "route_candidate_alignment_credit",
    "route_candidate_mismatch_penalty",
    "selected_route_candidate_score_gap",
    "route_candidate_useful_work_factor",
    "candidate_alignment_blocked_no_useful_work",
    "candidate_alignment_blocked_brittle_secondary",
    "candidate_alignment_blocked_route_failure",
    "candidate_alignment_blocked_customer_revisit",
    "candidate_alignment_blocked_severe_pressure",
    "candidate_alignment_blocked_impossible_dispatch",
    "candidate_alignment_blocked_service_lateness",
)

HARD_BLOCKER_TELEMETRY_FIELDS: tuple[tuple[str, str], ...] = (
    ("no_current_work_dqn_delivery_credit", "no-current-work dqn_delivery_credit"),
    ("hold_delivery_credit_leak", "hold route/delivery credit leak"),
    ("action8_route_or_delivery_credit_leak", "action8 route/delivery credit leak"),
    ("unsafe_24_25_candidate_credit", "unsafe 24/25 candidate credit"),
    ("no_work_positive_dqn_local", "no-work positive dqn_local"),
    ("emergency_zero_useful_positive_credit", "emergency zero-useful positive credit"),
)

GLOBAL_FATAL_HARD_BLOCKER_FIELDS: tuple[tuple[str, str], ...] = (
    ("fake_dispatch_credit", "fake dispatch credit"),
    ("customer_revisited", "customer_revisited"),
    ("route_failure_positive_dispatch_credit", "route failure with positive dispatch credit"),
    ("nan_inf_detected", "NaN/Inf"),
    ("dqn_local_negative_positive_train_rows", "dqn_local_negative_positive_train_rows"),
    *HARD_BLOCKER_TELEMETRY_FIELDS,
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _is_active_assigned_order(order: Mapping[str, Any]) -> bool:
    return (
        order.get("status") in {"assigned", "in_transit"}
        and bool(order.get("assigned_vehicle_id"))
    )


def _is_stuck_assigned_order(order: Mapping[str, Any], *, final_time: float | None) -> bool:
    if not _is_active_assigned_order(order):
        return False
    if bool(order.get("is_late")):
        return True
    due_time = _finite_float(order.get("due_time"))
    if due_time is not None and final_time is not None:
        return due_time <= final_time
    pickup_time = _finite_float(order.get("pickup_time"))
    if pickup_time is not None and final_time is not None:
        return final_time - pickup_time > STALE_ASSIGNED_ORDER_SECONDS
    return False


def _assigned_order_examples(
    snapshot: Mapping[str, Any],
    *,
    scenario_id: str,
    seed: int,
    episode_index: int,
    stuck_only: bool,
    limit: int = 5,
) -> list[JsonDict]:
    orders = snapshot.get("orders", {})
    if not isinstance(orders, Mapping):
        return []
    vehicles = snapshot.get("vehicles", {})
    if not isinstance(vehicles, Mapping):
        vehicles = {}
    final_time = _finite_float(snapshot.get("time"))
    examples: list[JsonDict] = []
    for order_id, order in orders.items():
        if len(examples) >= limit:
            break
        if not isinstance(order, Mapping):
            continue
        assigned_vehicle_id = order.get("assigned_vehicle_id")
        if not _is_active_assigned_order(order):
            continue
        if stuck_only and not _is_stuck_assigned_order(order, final_time=final_time):
            continue
        assigned_vehicle_key = str(assigned_vehicle_id)
        vehicle = vehicles.get(assigned_vehicle_key, {})
        if not isinstance(vehicle, Mapping):
            vehicle = {}
        due_time = _finite_float(order.get("due_time"))
        seconds_to_due = None
        if due_time is not None and final_time is not None:
            seconds_to_due = due_time - final_time
        examples.append(
            {
                "scenario_id": scenario_id,
                "seed": seed,
                "episode_index": episode_index,
                "order_id": str(order_id),
                "status": order.get("status"),
                "assigned_vehicle_id": assigned_vehicle_key,
                "pickup_time": order.get("pickup_time"),
                "delivery_time": order.get("delivery_time"),
                "due_time": order.get("due_time"),
                "is_late": order.get("is_late"),
                "final_snapshot_time": final_time,
                "seconds_to_due": seconds_to_due,
                "vehicle_tier": vehicle.get("tier"),
                "vehicle_asset_kind": vehicle.get("asset_kind"),
                "vehicle_load_units": vehicle.get("load_units"),
                "vehicle_utilization": vehicle.get("utilization"),
                "vehicle_current_node_id": vehicle.get("current_node_id"),
            }
        )
    return examples


def _active_assigned_order_examples(
    snapshot: Mapping[str, Any],
    *,
    scenario_id: str,
    seed: int,
    episode_index: int,
    limit: int = 5,
) -> list[JsonDict]:
    return _assigned_order_examples(
        snapshot,
        scenario_id=scenario_id,
        seed=seed,
        episode_index=episode_index,
        stuck_only=False,
        limit=limit,
    )


def _stuck_assigned_order_examples(
    snapshot: Mapping[str, Any],
    *,
    scenario_id: str,
    seed: int,
    episode_index: int,
    limit: int = 5,
) -> list[JsonDict]:
    return _assigned_order_examples(
        snapshot,
        scenario_id=scenario_id,
        seed=seed,
        episode_index=episode_index,
        stuck_only=True,
        limit=limit,
    )


def _component(info: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    raw = info.get("reward_components", {})
    components = raw if isinstance(raw, Mapping) else {}
    return _as_float(components.get(key, default), default)


def _projected_action(info: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = info.get("projected_action", {})
    return raw if isinstance(raw, Mapping) else {}


def _counter_dict(counter: Mapping[Any, Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.items()}


def _mean_map(sum_by_key: Mapping[str, float], count_by_key: Mapping[str, int]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, total in sum_by_key.items():
        count = int(count_by_key.get(key, 0))
        if count > 0:
            result[str(key)] = float(total) / count
    return result


def _nested_float_dict(nested: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        str(outer_key): {str(inner_key): float(value) for inner_key, value in inner.items()}
        for outer_key, inner in nested.items()
    }


def _nested_int_dict(nested: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        str(outer_key): {str(inner_key): int(value) for inner_key, value in inner.items()}
        for outer_key, inner in nested.items()
    }


def _nested_mean_map(
    sum_by_outer_key: Mapping[str, Mapping[str, float]],
    count_by_outer_key: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for outer_key, sums in sum_by_outer_key.items():
        means: dict[str, float] = {}
        counts = count_by_outer_key.get(outer_key, {})
        for inner_key, total in sums.items():
            count = int(counts.get(inner_key, 0))
            if count > 0:
                means[str(inner_key)] = float(total) / count
        if means:
            result[str(outer_key)] = means
    return result


def _ratio_map(success_by_key: Mapping[str, int], attempt_by_key: Mapping[str, int]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, attempts in attempt_by_key.items():
        attempts_int = int(attempts)
        if attempts_int > 0:
            result[str(key)] = int(success_by_key.get(key, 0)) / attempts_int
    return result


def _summary_numeric(summary: Mapping[str, Any], name: str) -> float | None:
    value = summary.get(name)
    return _as_float(value) if value is not None else None


@dataclass(slots=True)
class EpisodeMetrics:
    scenario_id: str
    seed: int
    episode_index: int
    steps: int = 0
    total_reward: float = 0.0
    total_delivered: int | None = None
    delivered_delta: float | None = None
    final_step_delivered_delta: float | None = None
    episode_delivered_delta_sum: float | None = None
    mean_step_delivered_delta: float | None = None
    service_level: float | None = None
    true_lateness_pressure: float | None = None
    actionable_lateness_risk: float | None = None
    stockout_risk: float | None = None
    inventory_shortfall: float | None = None
    excess_inventory_penalty: float | None = None
    holding_inventory_proxy: float | None = None
    reorder_fraction: float | None = None
    safety_stock_multiplier: float | None = None
    capacity_buffer_fraction: float | None = None
    capacity_pressure: float | None = None
    capacity_rejected: float | None = None
    no_vehicle_available: int = 0
    dispatch_attempts: int = 0
    dispatch_successes: int = 0
    dispatch_success_per_attempt: float | None = None
    dispatch_rate: float | None = None
    hold_rate: float | None = None
    dispatch_distribution: dict[str, int] = field(default_factory=dict)
    route_distribution: dict[str, int] = field(default_factory=dict)
    fleet_distribution: dict[str, int] = field(default_factory=dict)
    reorder_mode_distribution: dict[str, int] = field(default_factory=dict)
    action_distribution: dict[str, int] = field(default_factory=dict)
    top_action_ids: list[dict[str, float | int]] = field(default_factory=list)
    action_24_count: int = 0
    action_24_percentage: float | None = None
    action_25_count: int = 0
    action_25_percentage: float | None = None
    action_24_25_concentration: float | None = None
    dispatch_success_by_action_id: dict[str, int] = field(default_factory=dict)
    failed_noop_dispatch_by_action_id: dict[str, int] = field(default_factory=dict)
    action_attempts_by_id: dict[str, int] = field(default_factory=dict)
    action_successes_by_id: dict[str, int] = field(default_factory=dict)
    action_failed_noop_by_id: dict[str, int] = field(default_factory=dict)
    action_hold_by_id: dict[str, int] = field(default_factory=dict)
    action_dispatch_by_id: dict[str, int] = field(default_factory=dict)
    action_no_vehicle_by_id: dict[str, int] = field(default_factory=dict)
    action_already_assigned_by_id: dict[str, int] = field(default_factory=dict)
    action_no_unassigned_by_id: dict[str, int] = field(default_factory=dict)
    action_no_current_work_by_id: dict[str, int] = field(default_factory=dict)
    action_route_failure_by_id: dict[str, int] = field(default_factory=dict)
    action_dispatch_success_ratio_by_id: dict[str, float] = field(default_factory=dict)
    reward_component_sum_by_action_id: dict[str, dict[str, float]] = field(default_factory=dict)
    reward_component_count_by_action_id: dict[str, dict[str, int]] = field(default_factory=dict)
    reward_component_mean_by_action_id: dict[str, dict[str, float]] = field(default_factory=dict)
    reward_component_sum_by_action_family: dict[str, dict[str, float]] = field(default_factory=dict)
    reward_component_count_by_action_family: dict[str, dict[str, int]] = field(default_factory=dict)
    reward_component_mean_by_action_family: dict[str, dict[str, float]] = field(default_factory=dict)
    route_distribution_successful_dispatch: dict[str, int] = field(default_factory=dict)
    route_distribution_failed_dispatch: dict[str, int] = field(default_factory=dict)
    route_distribution_hold: dict[str, int] = field(default_factory=dict)
    route_distribution_by_action_id: dict[str, dict[str, int]] = field(default_factory=dict)
    route_success_count_by_route: dict[str, int] = field(default_factory=dict)
    route_failed_noop_count_by_route: dict[str, int] = field(default_factory=dict)
    route_dispatch_success_ratio_by_route: dict[str, float] = field(default_factory=dict)
    route_candidate_score_sum_by_route: dict[str, float] = field(default_factory=dict)
    route_candidate_score_count_by_route: dict[str, int] = field(default_factory=dict)
    route_candidate_score_mean_by_route: dict[str, float] = field(default_factory=dict)
    selected_route_candidate_score_sum_by_route: dict[str, float] = field(default_factory=dict)
    selected_route_candidate_score_count_by_route: dict[str, int] = field(default_factory=dict)
    selected_route_candidate_score_mean_by_route: dict[str, float] = field(default_factory=dict)
    selected_route_rank_counts: dict[str, int] = field(default_factory=dict)
    selected_route_best_count: int = 0
    selected_route_near_best_count: int = 0
    selected_route_margin_to_best_sum: float = 0.0
    selected_route_margin_to_best_mean: float | None = None
    shortest_near_best_but_not_selected_count: int = 0
    high_resilience_selected_when_shortest_near_best_count: int = 0
    high_resilience_selected_when_not_best_count: int = 0
    mixed_route_overconservative_candidate_count: int = 0
    dqn_local_negative_positive_train_rows: int = 0
    already_assigned_context_rows: int = 0
    active_assigned_in_transit_orders: int | None = None
    active_assigned_in_transit_order_examples: list[JsonDict] = field(default_factory=list)
    stuck_assigned_in_transit_orders: int | None = None
    stuck_assigned_in_transit_order_examples: list[JsonDict] = field(default_factory=list)
    route_failures: int = 0
    customer_revisited: int = 0
    fake_dispatch_credit: int = 0
    route_failure_positive_dispatch_credit: int = 0
    mixed_success_route_failure_steps: int = 0
    no_current_work_dqn_delivery_credit: int = 0
    hold_delivery_credit_leak: int = 0
    action8_route_or_delivery_credit_leak: int = 0
    unsafe_24_25_candidate_credit: int = 0
    no_work_positive_dqn_local: int = 0
    emergency_zero_useful_positive_credit: int = 0
    premium_sla_fleet_credit_blocked_no_current_work: int = 0
    premium_primary_adaptation_credit_blocked_no_current_work: int = 0
    premium_fleet_credit_allowed: int = 0
    premium_pressure_steps: int = 0
    premium_hold_steps: int = 0
    premium_dispatch_steps: int = 0
    premium_hold_under_useful_dispatch_opportunity: int = 0
    premium_dispatch_under_useful_dispatch_opportunity: int = 0
    premium_no_vehicle_steps: int = 0
    premium_already_assigned_steps: int = 0
    premium_no_unassigned_steps: int = 0
    premium_reorder_none_steps: int = 0
    premium_reorder_conservative_steps: int = 0
    premium_reorder_aggressive_steps: int = 0
    premium_reorder_emergency_steps: int = 0
    premium_useful_reorder_opportunity_steps: int = 0
    premium_reorder_missed_opportunity_steps: int = 0
    premium_service_pressure_steps: int = 0
    premium_late_or_at_risk_backlog_steps: int = 0
    hold_primary_under_premium_pressure: int = 0
    dispatch_primary_under_premium_pressure: int = 0
    transport_cost_delta: float | None = None
    emergency_aggressive_reorder_usage: float | None = None
    nan_inf_detected: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "episode_index": self.episode_index,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "total_delivered": self.total_delivered,
            "delivered_delta": self.delivered_delta,
            "final_step_delivered_delta": self.final_step_delivered_delta,
            "episode_delivered_delta_sum": self.episode_delivered_delta_sum,
            "mean_step_delivered_delta": self.mean_step_delivered_delta,
            "service_level": self.service_level,
            "true_lateness_pressure": self.true_lateness_pressure,
            "actionable_lateness_risk": self.actionable_lateness_risk,
            "stockout_risk": self.stockout_risk,
            "inventory_shortfall": self.inventory_shortfall,
            "excess_inventory_penalty": self.excess_inventory_penalty,
            "holding_inventory_proxy": self.holding_inventory_proxy,
            "reorder_fraction": self.reorder_fraction,
            "safety_stock_multiplier": self.safety_stock_multiplier,
            "capacity_buffer_fraction": self.capacity_buffer_fraction,
            "capacity_pressure": self.capacity_pressure,
            "capacity_rejected": self.capacity_rejected,
            "no_vehicle_available": self.no_vehicle_available,
            "dispatch_attempts": self.dispatch_attempts,
            "dispatch_successes": self.dispatch_successes,
            "dispatch_success_per_attempt": self.dispatch_success_per_attempt,
            "dispatch_rate": self.dispatch_rate,
            "hold_rate": self.hold_rate,
            "dispatch_distribution": dict(self.dispatch_distribution),
            "route_distribution": dict(self.route_distribution),
            "fleet_distribution": dict(self.fleet_distribution),
            "reorder_mode_distribution": dict(self.reorder_mode_distribution),
            "action_distribution": dict(self.action_distribution),
            "top_action_ids": list(self.top_action_ids),
            "action_24_count": self.action_24_count,
            "action_24_percentage": self.action_24_percentage,
            "action_25_count": self.action_25_count,
            "action_25_percentage": self.action_25_percentage,
            "action_24_25_concentration": self.action_24_25_concentration,
            "dispatch_success_by_action_id": dict(self.dispatch_success_by_action_id),
            "failed_noop_dispatch_by_action_id": dict(self.failed_noop_dispatch_by_action_id),
            "action_attempts_by_id": dict(self.action_attempts_by_id),
            "action_successes_by_id": dict(self.action_successes_by_id),
            "action_failed_noop_by_id": dict(self.action_failed_noop_by_id),
            "action_hold_by_id": dict(self.action_hold_by_id),
            "action_dispatch_by_id": dict(self.action_dispatch_by_id),
            "action_no_vehicle_by_id": dict(self.action_no_vehicle_by_id),
            "action_already_assigned_by_id": dict(self.action_already_assigned_by_id),
            "action_no_unassigned_by_id": dict(self.action_no_unassigned_by_id),
            "action_no_current_work_by_id": dict(self.action_no_current_work_by_id),
            "action_route_failure_by_id": dict(self.action_route_failure_by_id),
            "action_dispatch_success_ratio_by_id": dict(self.action_dispatch_success_ratio_by_id),
            "reward_component_sum_by_action_id": {
                str(action_id): dict(components)
                for action_id, components in self.reward_component_sum_by_action_id.items()
            },
            "reward_component_count_by_action_id": {
                str(action_id): dict(components)
                for action_id, components in self.reward_component_count_by_action_id.items()
            },
            "reward_component_mean_by_action_id": {
                str(action_id): dict(components)
                for action_id, components in self.reward_component_mean_by_action_id.items()
            },
            "reward_component_sum_by_action_family": {
                str(family): dict(components)
                for family, components in self.reward_component_sum_by_action_family.items()
            },
            "reward_component_count_by_action_family": {
                str(family): dict(components)
                for family, components in self.reward_component_count_by_action_family.items()
            },
            "reward_component_mean_by_action_family": {
                str(family): dict(components)
                for family, components in self.reward_component_mean_by_action_family.items()
            },
            "route_distribution_successful_dispatch": dict(self.route_distribution_successful_dispatch),
            "route_distribution_failed_dispatch": dict(self.route_distribution_failed_dispatch),
            "route_distribution_hold": dict(self.route_distribution_hold),
            "route_distribution_by_action_id": {
                str(action_id): dict(routes) for action_id, routes in self.route_distribution_by_action_id.items()
            },
            "route_success_count_by_route": dict(self.route_success_count_by_route),
            "route_failed_noop_count_by_route": dict(self.route_failed_noop_count_by_route),
            "route_dispatch_success_ratio_by_route": dict(self.route_dispatch_success_ratio_by_route),
            "route_candidate_score_sum_by_route": dict(self.route_candidate_score_sum_by_route),
            "route_candidate_score_count_by_route": dict(self.route_candidate_score_count_by_route),
            "route_candidate_score_mean_by_route": dict(self.route_candidate_score_mean_by_route),
            "selected_route_candidate_score_sum_by_route": dict(self.selected_route_candidate_score_sum_by_route),
            "selected_route_candidate_score_count_by_route": dict(self.selected_route_candidate_score_count_by_route),
            "selected_route_candidate_score_mean_by_route": dict(self.selected_route_candidate_score_mean_by_route),
            "selected_route_rank_counts": dict(self.selected_route_rank_counts),
            "selected_route_best_count": self.selected_route_best_count,
            "selected_route_near_best_count": self.selected_route_near_best_count,
            "selected_route_margin_to_best_sum": self.selected_route_margin_to_best_sum,
            "selected_route_margin_to_best_mean": self.selected_route_margin_to_best_mean,
            "shortest_near_best_but_not_selected_count": self.shortest_near_best_but_not_selected_count,
            "high_resilience_selected_when_shortest_near_best_count": (
                self.high_resilience_selected_when_shortest_near_best_count
            ),
            "high_resilience_selected_when_not_best_count": self.high_resilience_selected_when_not_best_count,
            "mixed_route_overconservative_candidate_count": self.mixed_route_overconservative_candidate_count,
            "dqn_local_negative_positive_train_rows": self.dqn_local_negative_positive_train_rows,
            "already_assigned_context_rows": self.already_assigned_context_rows,
            "active_assigned_in_transit_orders": self.active_assigned_in_transit_orders,
            "active_assigned_in_transit_order_examples": [
                dict(item) for item in self.active_assigned_in_transit_order_examples
            ],
            "stuck_assigned_in_transit_orders": self.stuck_assigned_in_transit_orders,
            "stuck_assigned_in_transit_order_examples": [
                dict(item) for item in self.stuck_assigned_in_transit_order_examples
            ],
            "route_failures": self.route_failures,
            "customer_revisited": self.customer_revisited,
            "fake_dispatch_credit": self.fake_dispatch_credit,
            "route_failure_positive_dispatch_credit": self.route_failure_positive_dispatch_credit,
            "mixed_success_route_failure_steps": self.mixed_success_route_failure_steps,
            "no_current_work_dqn_delivery_credit": self.no_current_work_dqn_delivery_credit,
            "hold_delivery_credit_leak": self.hold_delivery_credit_leak,
            "action8_route_or_delivery_credit_leak": self.action8_route_or_delivery_credit_leak,
            "unsafe_24_25_candidate_credit": self.unsafe_24_25_candidate_credit,
            "no_work_positive_dqn_local": self.no_work_positive_dqn_local,
            "emergency_zero_useful_positive_credit": self.emergency_zero_useful_positive_credit,
            "premium_sla_fleet_credit_blocked_no_current_work": (
                self.premium_sla_fleet_credit_blocked_no_current_work
            ),
            "premium_primary_adaptation_credit_blocked_no_current_work": (
                self.premium_primary_adaptation_credit_blocked_no_current_work
            ),
            "premium_fleet_credit_allowed": self.premium_fleet_credit_allowed,
            "premium_pressure_steps": self.premium_pressure_steps,
            "premium_hold_steps": self.premium_hold_steps,
            "premium_dispatch_steps": self.premium_dispatch_steps,
            "premium_hold_under_useful_dispatch_opportunity": (
                self.premium_hold_under_useful_dispatch_opportunity
            ),
            "premium_dispatch_under_useful_dispatch_opportunity": (
                self.premium_dispatch_under_useful_dispatch_opportunity
            ),
            "premium_no_vehicle_steps": self.premium_no_vehicle_steps,
            "premium_already_assigned_steps": self.premium_already_assigned_steps,
            "premium_no_unassigned_steps": self.premium_no_unassigned_steps,
            "premium_reorder_none_steps": self.premium_reorder_none_steps,
            "premium_reorder_conservative_steps": self.premium_reorder_conservative_steps,
            "premium_reorder_aggressive_steps": self.premium_reorder_aggressive_steps,
            "premium_reorder_emergency_steps": self.premium_reorder_emergency_steps,
            "premium_useful_reorder_opportunity_steps": self.premium_useful_reorder_opportunity_steps,
            "premium_reorder_missed_opportunity_steps": self.premium_reorder_missed_opportunity_steps,
            "premium_service_pressure_steps": self.premium_service_pressure_steps,
            "premium_late_or_at_risk_backlog_steps": self.premium_late_or_at_risk_backlog_steps,
            "hold_primary_under_premium_pressure": self.hold_primary_under_premium_pressure,
            "dispatch_primary_under_premium_pressure": self.dispatch_primary_under_premium_pressure,
            "transport_cost_delta": self.transport_cost_delta,
            "emergency_aggressive_reorder_usage": self.emergency_aggressive_reorder_usage,
            "nan_inf_detected": self.nan_inf_detected,
        }


class EpisodeMetricAccumulator:
    """Collect per-step info dictionaries into one episode metric row."""

    def __init__(self, *, scenario_id: str, seed: int, episode_index: int) -> None:
        self.scenario_id = scenario_id
        self.seed = seed
        self.episode_index = episode_index
        self.steps = 0
        self.total_reward = 0.0
        self.dispatch_steps = 0
        self.hold_steps = 0
        self.raw_action_counts: Counter[int] = Counter()
        self.action_dispatch_success_counts: Counter[int] = Counter()
        self.action_failed_noop_dispatch_counts: Counter[int] = Counter()
        self.action_hold_counts: Counter[int] = Counter()
        self.action_dispatch_counts: Counter[int] = Counter()
        self.action_no_vehicle_counts: Counter[int] = Counter()
        self.action_already_assigned_counts: Counter[int] = Counter()
        self.action_no_unassigned_counts: Counter[int] = Counter()
        self.action_no_current_work_counts: Counter[int] = Counter()
        self.action_route_failure_counts: Counter[int] = Counter()
        self.reward_component_sum_by_action_id: dict[str, Counter[str]] = {}
        self.reward_component_count_by_action_id: dict[str, Counter[str]] = {}
        self.reward_component_sum_by_action_family: dict[str, Counter[str]] = {}
        self.reward_component_count_by_action_family: dict[str, Counter[str]] = {}
        self.route_counts: Counter[str] = Counter()
        self.route_success_counts: Counter[str] = Counter()
        self.route_failed_noop_counts: Counter[str] = Counter()
        self.route_hold_counts: Counter[str] = Counter()
        self.route_by_action_counts: dict[int, Counter[str]] = {}
        self.route_candidate_score_sums: Counter[str] = Counter()
        self.route_candidate_score_counts: Counter[str] = Counter()
        self.selected_route_candidate_score_sums: Counter[str] = Counter()
        self.selected_route_candidate_score_counts: Counter[str] = Counter()
        self.selected_route_rank_counts: Counter[str] = Counter()
        self.selected_route_best_count = 0
        self.selected_route_near_best_count = 0
        self.selected_route_margin_to_best_sum = 0.0
        self.selected_route_margin_to_best_count = 0
        self.shortest_near_best_but_not_selected_count = 0
        self.high_resilience_selected_when_shortest_near_best_count = 0
        self.high_resilience_selected_when_not_best_count = 0
        self.mixed_route_overconservative_candidate_count = 0
        self.fleet_counts: Counter[str] = Counter()
        self.reorder_mode_counts: Counter[str] = Counter()
        self.last_values: dict[str, float] = {}
        self.final_step_delivered_delta: float | None = None
        self.episode_delivered_delta_sum = 0.0
        self.delivered_delta_observed = False
        self.dispatch_attempts = 0
        self.dispatch_successes = 0
        self.no_vehicle_rows = 0
        self.route_failures = 0
        self.customer_revisited = 0
        self.fake_dispatch_credit = 0
        self.route_failure_positive_credit = 0
        self.mixed_success_route_failure_steps = 0
        self.no_current_work_dqn_delivery_credit = 0
        self.hold_delivery_credit_leak = 0
        self.action8_route_or_delivery_credit_leak = 0
        self.unsafe_24_25_candidate_credit = 0
        self.no_work_positive_dqn_local = 0
        self.emergency_zero_useful_positive_credit = 0
        self.premium_sla_fleet_credit_blocked_no_current_work = 0
        self.premium_primary_adaptation_credit_blocked_no_current_work = 0
        self.premium_fleet_credit_allowed = 0
        self.premium_pressure_steps = 0
        self.premium_hold_steps = 0
        self.premium_dispatch_steps = 0
        self.premium_hold_under_useful_dispatch_opportunity = 0
        self.premium_dispatch_under_useful_dispatch_opportunity = 0
        self.premium_no_vehicle_steps = 0
        self.premium_already_assigned_steps = 0
        self.premium_no_unassigned_steps = 0
        self.premium_reorder_counts: Counter[str] = Counter()
        self.premium_useful_reorder_opportunity_steps = 0
        self.premium_reorder_missed_opportunity_steps = 0
        self.premium_service_pressure_steps = 0
        self.premium_late_or_at_risk_backlog_steps = 0
        self.hold_primary_under_premium_pressure = 0
        self.dispatch_primary_under_premium_pressure = 0
        self.dqn_local_negative_positive_train_rows = 0
        self.already_assigned_context_rows = 0
        self.nan_inf_detected = False
        self.last_snapshot: Mapping[str, Any] = {}

    def observe(self, *, reward: float, info: Mapping[str, Any]) -> None:
        self.steps += 1
        self.total_reward += reward
        if not math.isfinite(reward):
            self.nan_inf_detected = True

        projected = _projected_action(info)
        discrete = projected.get("discrete", {}) if isinstance(projected.get("discrete"), Mapping) else {}
        continuous = projected.get("continuous", {}) if isinstance(projected.get("continuous"), Mapping) else {}
        raw_action_id: int | None = None
        dispatch = discrete.get("dispatch")
        route_label = str(discrete["route"]) if discrete.get("route") is not None else None
        fleet_label = str(discrete["mode"]) if discrete.get("mode") is not None else None
        reorder_label = str(discrete["reorder"]) if discrete.get("reorder") is not None else None
        if dispatch == "dispatch":
            self.dispatch_steps += 1
        elif dispatch == "hold":
            self.hold_steps += 1
        if route_label is not None:
            self.route_counts[route_label] += 1
        if fleet_label is not None:
            self.fleet_counts[fleet_label] += 1
        if reorder_label is not None:
            self.reorder_mode_counts[reorder_label] += 1

        raw_action = info.get("raw_action", {})
        if isinstance(raw_action, Mapping) and raw_action.get("discrete") is not None:
            try:
                raw_action_id = int(raw_action["discrete"])
                self.raw_action_counts[raw_action_id] += 1
                if dispatch == "dispatch":
                    self.action_dispatch_counts[raw_action_id] += 1
                elif dispatch == "hold":
                    self.action_hold_counts[raw_action_id] += 1
                if route_label is not None:
                    self.route_by_action_counts.setdefault(raw_action_id, Counter())[route_label] += 1
            except (TypeError, ValueError):
                pass

        for source_key, target_key in {
            "reorder_fraction": "reorder_fraction",
            "safety_stock_multiplier": "safety_stock_multiplier",
            "capacity_buffer_fraction": "capacity_buffer_fraction",
        }.items():
            if source_key in continuous:
                self.last_values[target_key] = _as_float(continuous[source_key])

        components = info.get("reward_components", {})
        if isinstance(components, Mapping):
            for key in (
                "delivered_delta",
                "service_level",
                "true_lateness_pressure",
                "actionable_lateness_risk",
                "stockout_risk",
                "inventory_shortfall",
                "excess_inventory_penalty",
                "inventory_coverage",
                "capacity_pressure",
                "capacity_rejected",
                "transport_cost_delta",
            ):
                if key in components:
                    self.last_values[key] = _as_float(components[key])
                    if key == "delivered_delta":
                        delivered_delta = _as_float(components[key])
                        self.final_step_delivered_delta = delivered_delta
                        self.episode_delivered_delta_sum += delivered_delta
                        self.delivered_delta_observed = True
            for value in components.values():
                if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                    self.nan_inf_detected = True
            candidate_scores: dict[str, float] = {}
            for route, component_key in ROUTE_SCORE_COMPONENTS.items():
                if component_key in components:
                    score = _as_float(components[component_key])
                    candidate_scores[route] = score
                    self.route_candidate_score_sums[route] += score
                    self.route_candidate_score_counts[route] += 1
            selected_score: float | None = None
            if "selected_route_candidate_score" in components:
                selected_score = _as_float(components["selected_route_candidate_score"])
            elif route_label is not None and route_label in candidate_scores:
                selected_score = candidate_scores[route_label]
            if route_label is not None and selected_score is not None:
                self.selected_route_candidate_score_sums[route_label] += selected_score
                self.selected_route_candidate_score_counts[route_label] += 1
            if candidate_scores and route_label in candidate_scores:
                best_score = (
                    _as_float(components["best_route_candidate_score"])
                    if "best_route_candidate_score" in components
                    else max(candidate_scores.values())
                )
                selected_score = selected_score if selected_score is not None else candidate_scores[route_label]
                margin_to_best = (
                    max(_as_float(components["selected_route_candidate_score_gap"]), 0.0)
                    if "selected_route_candidate_score_gap" in components
                    else max(best_score - selected_score, 0.0)
                )
                selected_rank = 1 + sum(
                    1 for score in candidate_scores.values() if score > selected_score + EPSILON
                )
                self.selected_route_rank_counts[str(selected_rank)] += 1
                self.selected_route_margin_to_best_sum += margin_to_best
                self.selected_route_margin_to_best_count += 1
                selected_near_best = (
                    _as_float(components["selected_route_candidate_near_best"]) > EPSILON
                    if "selected_route_candidate_near_best" in components
                    else margin_to_best <= 0.30
                )
                if selected_rank == 1:
                    self.selected_route_best_count += 1
                if selected_near_best:
                    self.selected_route_near_best_count += 1
                shortest_score = candidate_scores.get("shortest")
                shortest_near_best = (
                    shortest_score is not None
                    and shortest_score > EPSILON
                    and max(best_score - shortest_score, 0.0) <= 0.30
                )
                if shortest_near_best and route_label != "shortest":
                    self.shortest_near_best_but_not_selected_count += 1
                if route_label == "high_resilience" and shortest_near_best:
                    self.high_resilience_selected_when_shortest_near_best_count += 1
                    if selected_rank > 1:
                        self.mixed_route_overconservative_candidate_count += 1
                if route_label == "high_resilience" and selected_rank > 1:
                    self.high_resilience_selected_when_not_best_count += 1

            action_family_key = (
                f"{dispatch}:{route_label}:{fleet_label}:{reorder_label}"
                if dispatch is not None
                and route_label is not None
                and fleet_label is not None
                and reorder_label is not None
                else None
            )
            for component_key in DIAGNOSTIC_REWARD_COMPONENTS:
                if component_key not in components:
                    continue
                component_value = _finite_float(components[component_key])
                if component_value is None:
                    continue
                if raw_action_id is not None:
                    action_key = str(raw_action_id)
                    self.reward_component_sum_by_action_id.setdefault(action_key, Counter())[component_key] += (
                        component_value
                    )
                    self.reward_component_count_by_action_id.setdefault(action_key, Counter())[component_key] += 1
                if action_family_key is not None:
                    self.reward_component_sum_by_action_family.setdefault(
                        action_family_key,
                        Counter(),
                    )[component_key] += component_value
                    self.reward_component_count_by_action_family.setdefault(
                        action_family_key,
                        Counter(),
                    )[component_key] += 1

        dqn_delivery_credit = _component(info, "dqn_delivery_credit")
        route_or_delivery_credit = any(
            _component(info, key) > EPSILON
            for key in (
                "dqn_delivery_credit",
                "route_candidate_alignment_credit",
                "route_adaptation_reward_or_credit",
                "low_congestion_adaptation_credit",
                "high_resilience_adaptation_credit",
            )
        )
        route_credit = any(
            _component(info, key) > EPSILON
            for key in (
                "route_candidate_alignment_credit",
                "route_adaptation_reward_or_credit",
                "low_congestion_adaptation_credit",
                "high_resilience_adaptation_credit",
            )
        )
        current_dispatch_work_signal = _component(info, "current_dispatch_work_signal")
        current_dispatch_work_count = _component(info, "current_dispatch_work_count")
        if current_dispatch_work_signal <= EPSILON and dqn_delivery_credit > EPSILON:
            self.no_current_work_dqn_delivery_credit += 1
        if dispatch == "hold" and route_or_delivery_credit:
            self.hold_delivery_credit_leak += 1
        if raw_action_id == 8 and route_or_delivery_credit:
            self.action8_route_or_delivery_credit_leak += 1
        if (
            raw_action_id in {24, 25}
            and _component(info, "action24_25_candidate_credit_allowed") <= EPSILON
            and route_credit
        ):
            self.unsafe_24_25_candidate_credit += 1
        if (
            current_dispatch_work_count <= EPSILON
            and current_dispatch_work_signal <= EPSILON
            and _component(info, "dqn_local") > EPSILON
        ):
            self.no_work_positive_dqn_local += 1
        if (
            _component(info, "emergency_useful_factor") <= EPSILON
            and _component(info, "emergency_procurement_justification_credit") > EPSILON
        ):
            self.emergency_zero_useful_positive_credit += 1
        if _component(info, "premium_sla_fleet_credit_blocked_no_current_work") > EPSILON:
            self.premium_sla_fleet_credit_blocked_no_current_work += 1
        if _component(info, "premium_primary_adaptation_credit_blocked_no_current_work") > EPSILON:
            self.premium_primary_adaptation_credit_blocked_no_current_work += 1
        if _component(info, "premium_fleet_credit_allowed") > EPSILON:
            self.premium_fleet_credit_allowed += 1

        dispatch_attempted = _component(info, "dispatch_attempted")
        dispatch_success_count = _component(info, "dispatch_success_count")
        dispatch_no_vehicle = _component(info, "dispatch_no_vehicle_available")
        no_vehicle_rate = _component(info, "no_vehicle_available_rate")
        dispatch_no_unassigned = _component(info, "dispatch_no_unassigned_orders")
        dispatch_already_assigned = _component(info, "dispatch_already_assigned_count")
        already_assigned_rate = _component(info, "already_assigned_rate")
        route_failed_this_step = (
            _component(info, "dispatch_route_failure") > EPSILON
            or _component(info, "route_feasibility_failed_count") > EPSILON
        )
        no_current_dispatch_work = (
            current_dispatch_work_count <= EPSILON
            and current_dispatch_work_signal <= EPSILON
        )

        if dispatch_attempted > 0:
            self.dispatch_attempts += 1
        if dispatch_success_count > 0:
            self.dispatch_successes += 1
            if raw_action_id is not None:
                self.action_dispatch_success_counts[raw_action_id] += 1
        if dispatch == "dispatch" and dispatch_success_count <= EPSILON:
            if raw_action_id is not None:
                self.action_failed_noop_dispatch_counts[raw_action_id] += 1
        if raw_action_id is not None:
            if dispatch_no_vehicle > EPSILON or no_vehicle_rate > EPSILON:
                self.action_no_vehicle_counts[raw_action_id] += 1
            if dispatch_already_assigned > EPSILON or already_assigned_rate > EPSILON:
                self.action_already_assigned_counts[raw_action_id] += 1
            if dispatch_no_unassigned > EPSILON:
                self.action_no_unassigned_counts[raw_action_id] += 1
            if route_failed_this_step:
                self.action_route_failure_counts[raw_action_id] += 1
            if dispatch == "dispatch" and dispatch_success_count <= EPSILON and no_current_dispatch_work:
                self.action_no_current_work_counts[raw_action_id] += 1
        if route_label is not None:
            if dispatch == "dispatch" and dispatch_success_count > EPSILON:
                self.route_success_counts[route_label] += 1
            elif dispatch == "dispatch":
                self.route_failed_noop_counts[route_label] += 1
            elif dispatch == "hold":
                self.route_hold_counts[route_label] += 1
        if dispatch_no_vehicle > 0 or no_vehicle_rate > 0:
            self.no_vehicle_rows += 1
        if route_failed_this_step:
            self.route_failures += 1
        dispatch_progress_credit = _component(info, "dispatch_progress_credit")
        if dispatch_progress_credit > 0 and dispatch_success_count <= 0:
            self.fake_dispatch_credit += 1
        if dispatch_progress_credit > 0 and route_failed_this_step and dispatch_success_count <= 0:
            self.route_failure_positive_credit += 1
        if dispatch_progress_credit > 0 and route_failed_this_step and dispatch_success_count > 0:
            self.mixed_success_route_failure_steps += 1
        if _component(info, "dqn_local") < 0 and _component(info, "dqn_train") > 0:
            self.dqn_local_negative_positive_train_rows += 1
        if dispatch_already_assigned > 0 or already_assigned_rate > 0:
            self.already_assigned_context_rows += 1

        premium_pressure = _component(info, "premium_sla_pressure") > EPSILON
        pending_work_pressure = _component(info, "pending_work_pressure")
        useful_dispatch_opportunity = (
            pending_work_pressure > EPSILON
            or current_dispatch_work_count > EPSILON
            or current_dispatch_work_signal > EPSILON
        )
        useful_reorder_opportunity = (
            _component(info, "emergency_gate") > EPSILON
            or _component(info, "emergency_useful_factor") > EPSILON
            or _component(info, "inventory_shortfall") > EPSILON
            or _component(info, "backlog_age_pressure") > EPSILON
        )
        late_or_at_risk_backlog = any(
            _component(info, key) > EPSILON
            for key in (
                "lateness_risk",
                "raw_lateness_risk",
                "actionable_lateness_risk",
                "backlog_age_pressure",
            )
        )
        if premium_pressure:
            self.premium_pressure_steps += 1
            if dispatch == "hold":
                self.premium_hold_steps += 1
            elif dispatch == "dispatch":
                self.premium_dispatch_steps += 1
            if dispatch == "hold" and useful_dispatch_opportunity:
                self.premium_hold_under_useful_dispatch_opportunity += 1
            if dispatch == "dispatch" and useful_dispatch_opportunity:
                self.premium_dispatch_under_useful_dispatch_opportunity += 1
            if dispatch_no_vehicle > EPSILON or no_vehicle_rate > EPSILON:
                self.premium_no_vehicle_steps += 1
            if dispatch_already_assigned > EPSILON or already_assigned_rate > EPSILON:
                self.premium_already_assigned_steps += 1
            if dispatch_no_unassigned > EPSILON:
                self.premium_no_unassigned_steps += 1
            if reorder_label in REORDER_MODES:
                self.premium_reorder_counts[reorder_label] += 1
            if useful_reorder_opportunity:
                self.premium_useful_reorder_opportunity_steps += 1
                if reorder_label in {"none", "conservative"}:
                    self.premium_reorder_missed_opportunity_steps += 1
            if useful_dispatch_opportunity or late_or_at_risk_backlog:
                self.premium_service_pressure_steps += 1
            if late_or_at_risk_backlog:
                self.premium_late_or_at_risk_backlog_steps += 1
            if fleet_label == "primary_fleet" and dispatch == "hold":
                self.hold_primary_under_premium_pressure += 1
            if fleet_label == "primary_fleet" and dispatch == "dispatch":
                self.dispatch_primary_under_premium_pressure += 1

        snapshot = info.get("snapshot", {})
        if isinstance(snapshot, Mapping):
            self.last_snapshot = snapshot
            if "customer_revisited" in str(snapshot):
                self.customer_revisited += 1

    def finish(self) -> EpisodeMetrics:
        total_delivered = None
        active_orders = None
        active_order_examples: list[JsonDict] = []
        stuck_orders = None
        stuck_order_examples: list[JsonDict] = []
        if isinstance(self.last_snapshot, Mapping):
            orders = self.last_snapshot.get("orders", {})
            final_time = _finite_float(self.last_snapshot.get("time"))
            if isinstance(orders, Mapping):
                total_delivered = sum(
                    1 for order in orders.values() if isinstance(order, Mapping) and order.get("status") == "delivered"
                )
                active_orders = sum(
                    1
                    for order in orders.values()
                    if isinstance(order, Mapping)
                    and _is_active_assigned_order(order)
                )
                stuck_orders = sum(
                    1
                    for order in orders.values()
                    if isinstance(order, Mapping)
                    and _is_stuck_assigned_order(order, final_time=final_time)
                )
                active_order_examples = _active_assigned_order_examples(
                    self.last_snapshot,
                    scenario_id=self.scenario_id,
                    seed=self.seed,
                    episode_index=self.episode_index,
                )
                stuck_order_examples = _stuck_assigned_order_examples(
                    self.last_snapshot,
                    scenario_id=self.scenario_id,
                    seed=self.seed,
                    episode_index=self.episode_index,
                )

        dispatch_success = (
            self.dispatch_successes / self.dispatch_attempts if self.dispatch_attempts > 0 else None
        )
        action_24_count = self.raw_action_counts[24]
        action_25_count = self.raw_action_counts[25]
        action_24_25 = (
            (action_24_count + action_25_count) / self.steps if self.steps else None
        )
        top_action_ids = [
            {
                "action_id": int(action_id),
                "count": int(count),
                "percentage": count / self.steps if self.steps else 0.0,
            }
            for action_id, count in self.raw_action_counts.most_common(10)
        ]
        emergency_aggressive = (
            (self.reorder_mode_counts["emergency"] + self.reorder_mode_counts["aggressive"]) / self.steps
            if self.steps
            else None
        )
        mean_step_delivered_delta = (
            self.episode_delivered_delta_sum / self.steps
            if self.steps and self.delivered_delta_observed
            else None
        )
        action_dispatch_by_id = _counter_dict(self.action_dispatch_counts)
        action_successes_by_id = _counter_dict(self.action_dispatch_success_counts)
        action_failed_noop_by_id = _counter_dict(self.action_failed_noop_dispatch_counts)
        route_success_by_route = _counter_dict(self.route_success_counts)
        route_failed_by_route = _counter_dict(self.route_failed_noop_counts)
        route_candidate_score_sums = {
            route: float(total) for route, total in self.route_candidate_score_sums.items()
        }
        route_candidate_score_counts = _counter_dict(self.route_candidate_score_counts)
        selected_route_candidate_score_sums = {
            route: float(total) for route, total in self.selected_route_candidate_score_sums.items()
        }
        selected_route_candidate_score_counts = _counter_dict(self.selected_route_candidate_score_counts)
        route_candidate_score_means = _mean_map(
            route_candidate_score_sums,
            route_candidate_score_counts,
        )
        selected_route_candidate_score_means = _mean_map(
            selected_route_candidate_score_sums,
            selected_route_candidate_score_counts,
        )
        route_dispatch_attempts_by_route = {
            route: int(route_success_by_route.get(route, 0)) + int(route_failed_by_route.get(route, 0))
            for route in set(route_success_by_route) | set(route_failed_by_route)
        }
        selected_route_margin_to_best_mean = (
            self.selected_route_margin_to_best_sum / self.selected_route_margin_to_best_count
            if self.selected_route_margin_to_best_count
            else None
        )
        reward_component_sum_by_action_id = _nested_float_dict(self.reward_component_sum_by_action_id)
        reward_component_count_by_action_id = _nested_int_dict(self.reward_component_count_by_action_id)
        reward_component_sum_by_action_family = _nested_float_dict(self.reward_component_sum_by_action_family)
        reward_component_count_by_action_family = _nested_int_dict(self.reward_component_count_by_action_family)
        return EpisodeMetrics(
            scenario_id=self.scenario_id,
            seed=self.seed,
            episode_index=self.episode_index,
            steps=self.steps,
            total_reward=self.total_reward,
            total_delivered=total_delivered,
            delivered_delta=mean_step_delivered_delta,
            final_step_delivered_delta=self.final_step_delivered_delta,
            episode_delivered_delta_sum=self.episode_delivered_delta_sum,
            mean_step_delivered_delta=mean_step_delivered_delta,
            service_level=self.last_values.get("service_level"),
            true_lateness_pressure=self.last_values.get("true_lateness_pressure"),
            actionable_lateness_risk=self.last_values.get("actionable_lateness_risk"),
            stockout_risk=self.last_values.get("stockout_risk"),
            inventory_shortfall=self.last_values.get("inventory_shortfall"),
            excess_inventory_penalty=self.last_values.get("excess_inventory_penalty"),
            holding_inventory_proxy=self.last_values.get("inventory_coverage"),
            reorder_fraction=self.last_values.get("reorder_fraction"),
            safety_stock_multiplier=self.last_values.get("safety_stock_multiplier"),
            capacity_buffer_fraction=self.last_values.get("capacity_buffer_fraction"),
            capacity_pressure=self.last_values.get("capacity_pressure"),
            capacity_rejected=self.last_values.get("capacity_rejected"),
            no_vehicle_available=self.no_vehicle_rows,
            dispatch_attempts=self.dispatch_attempts,
            dispatch_successes=self.dispatch_successes,
            dispatch_success_per_attempt=dispatch_success,
            dispatch_rate=self.dispatch_steps / self.steps if self.steps else None,
            hold_rate=self.hold_steps / self.steps if self.steps else None,
            dispatch_distribution={"dispatch": self.dispatch_steps, "hold": self.hold_steps},
            route_distribution=dict(self.route_counts),
            fleet_distribution=dict(self.fleet_counts),
            reorder_mode_distribution=dict(self.reorder_mode_counts),
            action_distribution={str(action_id): int(count) for action_id, count in self.raw_action_counts.items()},
            top_action_ids=top_action_ids,
            action_24_count=action_24_count,
            action_24_percentage=action_24_count / self.steps if self.steps else None,
            action_25_count=action_25_count,
            action_25_percentage=action_25_count / self.steps if self.steps else None,
            action_24_25_concentration=action_24_25,
            dispatch_success_by_action_id={
                str(action_id): int(count) for action_id, count in self.action_dispatch_success_counts.items()
            },
            failed_noop_dispatch_by_action_id={
                str(action_id): int(count) for action_id, count in self.action_failed_noop_dispatch_counts.items()
            },
            action_attempts_by_id={str(action_id): int(count) for action_id, count in self.raw_action_counts.items()},
            action_successes_by_id=action_successes_by_id,
            action_failed_noop_by_id=action_failed_noop_by_id,
            action_hold_by_id=_counter_dict(self.action_hold_counts),
            action_dispatch_by_id=action_dispatch_by_id,
            action_no_vehicle_by_id=_counter_dict(self.action_no_vehicle_counts),
            action_already_assigned_by_id=_counter_dict(self.action_already_assigned_counts),
            action_no_unassigned_by_id=_counter_dict(self.action_no_unassigned_counts),
            action_no_current_work_by_id=_counter_dict(self.action_no_current_work_counts),
            action_route_failure_by_id=_counter_dict(self.action_route_failure_counts),
            action_dispatch_success_ratio_by_id=_ratio_map(action_successes_by_id, action_dispatch_by_id),
            reward_component_sum_by_action_id=reward_component_sum_by_action_id,
            reward_component_count_by_action_id=reward_component_count_by_action_id,
            reward_component_mean_by_action_id=_nested_mean_map(
                reward_component_sum_by_action_id,
                reward_component_count_by_action_id,
            ),
            reward_component_sum_by_action_family=reward_component_sum_by_action_family,
            reward_component_count_by_action_family=reward_component_count_by_action_family,
            reward_component_mean_by_action_family=_nested_mean_map(
                reward_component_sum_by_action_family,
                reward_component_count_by_action_family,
            ),
            route_distribution_successful_dispatch=route_success_by_route,
            route_distribution_failed_dispatch=route_failed_by_route,
            route_distribution_hold=_counter_dict(self.route_hold_counts),
            route_distribution_by_action_id={
                str(action_id): _counter_dict(route_counter)
                for action_id, route_counter in self.route_by_action_counts.items()
            },
            route_success_count_by_route=route_success_by_route,
            route_failed_noop_count_by_route=route_failed_by_route,
            route_dispatch_success_ratio_by_route=_ratio_map(route_success_by_route, route_dispatch_attempts_by_route),
            route_candidate_score_sum_by_route=route_candidate_score_sums,
            route_candidate_score_count_by_route=route_candidate_score_counts,
            route_candidate_score_mean_by_route=route_candidate_score_means,
            selected_route_candidate_score_sum_by_route=selected_route_candidate_score_sums,
            selected_route_candidate_score_count_by_route=selected_route_candidate_score_counts,
            selected_route_candidate_score_mean_by_route=selected_route_candidate_score_means,
            selected_route_rank_counts=_counter_dict(self.selected_route_rank_counts),
            selected_route_best_count=self.selected_route_best_count,
            selected_route_near_best_count=self.selected_route_near_best_count,
            selected_route_margin_to_best_sum=self.selected_route_margin_to_best_sum,
            selected_route_margin_to_best_mean=selected_route_margin_to_best_mean,
            shortest_near_best_but_not_selected_count=self.shortest_near_best_but_not_selected_count,
            high_resilience_selected_when_shortest_near_best_count=(
                self.high_resilience_selected_when_shortest_near_best_count
            ),
            high_resilience_selected_when_not_best_count=self.high_resilience_selected_when_not_best_count,
            mixed_route_overconservative_candidate_count=self.mixed_route_overconservative_candidate_count,
            dqn_local_negative_positive_train_rows=self.dqn_local_negative_positive_train_rows,
            already_assigned_context_rows=self.already_assigned_context_rows,
            active_assigned_in_transit_orders=active_orders,
            active_assigned_in_transit_order_examples=active_order_examples,
            stuck_assigned_in_transit_orders=stuck_orders,
            stuck_assigned_in_transit_order_examples=stuck_order_examples,
            route_failures=self.route_failures,
            customer_revisited=self.customer_revisited,
            fake_dispatch_credit=self.fake_dispatch_credit,
            route_failure_positive_dispatch_credit=self.route_failure_positive_credit,
            mixed_success_route_failure_steps=self.mixed_success_route_failure_steps,
            no_current_work_dqn_delivery_credit=self.no_current_work_dqn_delivery_credit,
            hold_delivery_credit_leak=self.hold_delivery_credit_leak,
            action8_route_or_delivery_credit_leak=self.action8_route_or_delivery_credit_leak,
            unsafe_24_25_candidate_credit=self.unsafe_24_25_candidate_credit,
            no_work_positive_dqn_local=self.no_work_positive_dqn_local,
            emergency_zero_useful_positive_credit=self.emergency_zero_useful_positive_credit,
            premium_sla_fleet_credit_blocked_no_current_work=(
                self.premium_sla_fleet_credit_blocked_no_current_work
            ),
            premium_primary_adaptation_credit_blocked_no_current_work=(
                self.premium_primary_adaptation_credit_blocked_no_current_work
            ),
            premium_fleet_credit_allowed=self.premium_fleet_credit_allowed,
            premium_pressure_steps=self.premium_pressure_steps,
            premium_hold_steps=self.premium_hold_steps,
            premium_dispatch_steps=self.premium_dispatch_steps,
            premium_hold_under_useful_dispatch_opportunity=(
                self.premium_hold_under_useful_dispatch_opportunity
            ),
            premium_dispatch_under_useful_dispatch_opportunity=(
                self.premium_dispatch_under_useful_dispatch_opportunity
            ),
            premium_no_vehicle_steps=self.premium_no_vehicle_steps,
            premium_already_assigned_steps=self.premium_already_assigned_steps,
            premium_no_unassigned_steps=self.premium_no_unassigned_steps,
            premium_reorder_none_steps=self.premium_reorder_counts["none"],
            premium_reorder_conservative_steps=self.premium_reorder_counts["conservative"],
            premium_reorder_aggressive_steps=self.premium_reorder_counts["aggressive"],
            premium_reorder_emergency_steps=self.premium_reorder_counts["emergency"],
            premium_useful_reorder_opportunity_steps=self.premium_useful_reorder_opportunity_steps,
            premium_reorder_missed_opportunity_steps=self.premium_reorder_missed_opportunity_steps,
            premium_service_pressure_steps=self.premium_service_pressure_steps,
            premium_late_or_at_risk_backlog_steps=self.premium_late_or_at_risk_backlog_steps,
            hold_primary_under_premium_pressure=self.hold_primary_under_premium_pressure,
            dispatch_primary_under_premium_pressure=self.dispatch_primary_under_premium_pressure,
            transport_cost_delta=self.last_values.get("transport_cost_delta"),
            emergency_aggressive_reorder_usage=emergency_aggressive,
            nan_inf_detected=self.nan_inf_detected,
        )


def summarize_episodes(scenario_id: str, episodes: list[EpisodeMetrics]) -> JsonDict:
    if not episodes:
        return {"scenario_id": scenario_id, "episodes": 0}

    def mean(name: str) -> float | None:
        values = [getattr(item, name) for item in episodes if getattr(item, name) is not None]
        return sum(values) / len(values) if values else None

    def mean_delivery() -> float | None:
        values: list[float] = []
        for item in episodes:
            value = item.mean_step_delivered_delta
            if value is None:
                value = item.delivered_delta
            if value is not None:
                values.append(value)
        return sum(values) / len(values) if values else None

    def sum_counter(name: str) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for item in episodes:
            raw = getattr(item, name)
            if isinstance(raw, Mapping):
                for key, value in raw.items():
                    counter[str(key)] += int(value)
        return dict(counter)

    def sum_nested_counter(name: str) -> dict[str, dict[str, int]]:
        nested: dict[str, Counter[str]] = {}
        for item in episodes:
            raw = getattr(item, name)
            if isinstance(raw, Mapping):
                for outer_key, inner_raw in raw.items():
                    if isinstance(inner_raw, Mapping):
                        target = nested.setdefault(str(outer_key), Counter())
                        for inner_key, value in inner_raw.items():
                            target[str(inner_key)] += int(value)
        return {key: dict(counter) for key, counter in nested.items()}

    def sum_nested_float_map(name: str) -> dict[str, dict[str, float]]:
        nested: dict[str, Counter[str]] = {}
        for item in episodes:
            raw = getattr(item, name)
            if isinstance(raw, Mapping):
                for outer_key, inner_raw in raw.items():
                    if isinstance(inner_raw, Mapping):
                        target = nested.setdefault(str(outer_key), Counter())
                        for inner_key, value in inner_raw.items():
                            target[str(inner_key)] += _as_float(value)
        return {key: {inner_key: float(value) for inner_key, value in counter.items()} for key, counter in nested.items()}

    def sum_nested_int_map(name: str) -> dict[str, dict[str, int]]:
        nested: dict[str, Counter[str]] = {}
        for item in episodes:
            raw = getattr(item, name)
            if isinstance(raw, Mapping):
                for outer_key, inner_raw in raw.items():
                    if isinstance(inner_raw, Mapping):
                        target = nested.setdefault(str(outer_key), Counter())
                        for inner_key, value in inner_raw.items():
                            target[str(inner_key)] += int(value)
        return {key: {inner_key: int(value) for inner_key, value in counter.items()} for key, counter in nested.items()}

    def sum_float_map(name: str) -> dict[str, float]:
        totals: Counter[str] = Counter()
        for item in episodes:
            raw = getattr(item, name)
            if isinstance(raw, Mapping):
                for key, value in raw.items():
                    totals[str(key)] += _as_float(value)
        return {key: float(value) for key, value in totals.items()}

    def sum_int(name: str) -> int:
        return sum(int(getattr(item, name)) for item in episodes)

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator > 0 else None

    total_steps = sum(item.steps for item in episodes)
    action_distribution = sum_counter("action_distribution")
    dispatch_success_by_action_id = sum_counter("dispatch_success_by_action_id")
    failed_noop_dispatch_by_action_id = sum_counter("failed_noop_dispatch_by_action_id")
    action_attempts_by_id = sum_counter("action_attempts_by_id")
    action_successes_by_id = sum_counter("action_successes_by_id")
    action_failed_noop_by_id = sum_counter("action_failed_noop_by_id")
    action_hold_by_id = sum_counter("action_hold_by_id")
    action_dispatch_by_id = sum_counter("action_dispatch_by_id")
    action_no_vehicle_by_id = sum_counter("action_no_vehicle_by_id")
    action_already_assigned_by_id = sum_counter("action_already_assigned_by_id")
    action_no_unassigned_by_id = sum_counter("action_no_unassigned_by_id")
    action_no_current_work_by_id = sum_counter("action_no_current_work_by_id")
    action_route_failure_by_id = sum_counter("action_route_failure_by_id")
    action_dispatch_success_ratio_by_id = _ratio_map(action_successes_by_id, action_dispatch_by_id)
    route_successful_dispatch = sum_counter("route_distribution_successful_dispatch")
    route_failed_dispatch = sum_counter("route_distribution_failed_dispatch")
    route_hold = sum_counter("route_distribution_hold")
    route_success_count_by_route = sum_counter("route_success_count_by_route")
    route_failed_noop_count_by_route = sum_counter("route_failed_noop_count_by_route")
    route_dispatch_attempts_by_route = {
        route: int(route_success_count_by_route.get(route, 0)) + int(route_failed_noop_count_by_route.get(route, 0))
        for route in set(route_success_count_by_route) | set(route_failed_noop_count_by_route)
    }
    route_dispatch_success_ratio_by_route = _ratio_map(
        route_success_count_by_route,
        route_dispatch_attempts_by_route,
    )
    active_order_examples: list[JsonDict] = []
    for item in episodes:
        if len(active_order_examples) >= 20:
            break
        for example in item.active_assigned_in_transit_order_examples:
            if len(active_order_examples) >= 20:
                break
            active_order_examples.append(dict(example))
    stuck_order_examples: list[JsonDict] = []
    for item in episodes:
        if len(stuck_order_examples) >= 20:
            break
        for example in item.stuck_assigned_in_transit_order_examples:
            if len(stuck_order_examples) >= 20:
                break
            stuck_order_examples.append(dict(example))
    route_candidate_score_sum_by_route = sum_float_map("route_candidate_score_sum_by_route")
    route_candidate_score_count_by_route = sum_counter("route_candidate_score_count_by_route")
    route_candidate_score_mean_by_route = _mean_map(
        route_candidate_score_sum_by_route,
        route_candidate_score_count_by_route,
    )
    reward_component_sum_by_action_id = sum_nested_float_map("reward_component_sum_by_action_id")
    reward_component_count_by_action_id = sum_nested_int_map("reward_component_count_by_action_id")
    reward_component_sum_by_action_family = sum_nested_float_map("reward_component_sum_by_action_family")
    reward_component_count_by_action_family = sum_nested_int_map("reward_component_count_by_action_family")
    selected_route_candidate_score_sum_by_route = sum_float_map("selected_route_candidate_score_sum_by_route")
    selected_route_candidate_score_count_by_route = sum_counter("selected_route_candidate_score_count_by_route")
    selected_route_candidate_score_mean_by_route = _mean_map(
        selected_route_candidate_score_sum_by_route,
        selected_route_candidate_score_count_by_route,
    )
    selected_route_rank_counts = sum_counter("selected_route_rank_counts")
    selected_route_margin_to_best_sum = sum(item.selected_route_margin_to_best_sum for item in episodes)
    selected_route_margin_count = sum(sum(item.selected_route_rank_counts.values()) for item in episodes)
    selected_route_margin_to_best_mean = (
        selected_route_margin_to_best_sum / selected_route_margin_count
        if selected_route_margin_count
        else None
    )
    top_failed_noop_actions = [
        {
            "action_id": int(action_id),
            "failed_noop": int(failed_count),
            "dispatch_attempts": int(action_dispatch_by_id.get(action_id, 0)),
            "successes": int(action_successes_by_id.get(action_id, 0)),
            "dispatch_success_ratio": action_dispatch_success_ratio_by_id.get(action_id),
        }
        for action_id, failed_count in sorted(
            action_failed_noop_by_id.items(),
            key=lambda item: (-int(item[1]), int(item[0]) if str(item[0]).isdigit() else str(item[0])),
        )[:10]
    ]
    top_low_success_dispatch_actions = [
        {
            "action_id": int(action_id),
            "dispatch_attempts": int(attempts),
            "successes": int(action_successes_by_id.get(action_id, 0)),
            "failed_noop": int(action_failed_noop_by_id.get(action_id, 0)),
            "dispatch_success_ratio": action_dispatch_success_ratio_by_id.get(action_id),
        }
        for action_id, attempts in sorted(
            action_dispatch_by_id.items(),
            key=lambda item: (
                action_dispatch_success_ratio_by_id.get(item[0], 0.0),
                -int(item[1]),
                int(item[0]) if str(item[0]).isdigit() else str(item[0]),
            ),
        )[:10]
    ]
    action31_attempts = int(action_attempts_by_id.get("31", 0))
    action31_dispatches = int(action_dispatch_by_id.get("31", action31_attempts))
    action31_successes = int(action_successes_by_id.get("31", 0))
    action31_failed_noop = int(action_failed_noop_by_id.get("31", 0))
    top_action_ids = [
        {
            "action_id": int(action_id),
            "count": int(count),
            "percentage": count / total_steps if total_steps else 0.0,
        }
        for action_id, count in Counter(
            {int(action_id): count for action_id, count in action_distribution.items()}
        ).most_common(10)
    ]
    action_24_count = int(action_distribution.get("24", 0))
    action_25_count = int(action_distribution.get("25", 0))
    premium_pressure_steps = sum_int("premium_pressure_steps")
    premium_hold_steps = sum_int("premium_hold_steps")
    premium_dispatch_steps = sum_int("premium_dispatch_steps")
    premium_hold_under_useful_dispatch = sum_int("premium_hold_under_useful_dispatch_opportunity")
    premium_dispatch_under_useful_dispatch = sum_int("premium_dispatch_under_useful_dispatch_opportunity")
    premium_useful_reorder_opportunity_steps = sum_int("premium_useful_reorder_opportunity_steps")
    premium_reorder_missed_opportunity_steps = sum_int("premium_reorder_missed_opportunity_steps")
    hold_primary_under_premium_pressure = sum_int("hold_primary_under_premium_pressure")

    return {
        "scenario_id": scenario_id,
        "episodes": len(episodes),
        "steps": total_steps,
        "total_delivered": sum(item.total_delivered or 0 for item in episodes),
        "delivered_delta": mean_delivery(),
        "final_step_delivered_delta": mean("final_step_delivered_delta"),
        "episode_delivered_delta_sum": sum(item.episode_delivered_delta_sum or 0 for item in episodes),
        "mean_step_delivered_delta": mean_delivery(),
        "service_level": mean("service_level"),
        "true_lateness_pressure": mean("true_lateness_pressure"),
        "actionable_lateness_risk": mean("actionable_lateness_risk"),
        "stockout_risk": mean("stockout_risk"),
        "inventory_shortfall": mean("inventory_shortfall"),
        "excess_inventory_penalty": mean("excess_inventory_penalty"),
        "holding_inventory_proxy": mean("holding_inventory_proxy"),
        "reorder_fraction": mean("reorder_fraction"),
        "safety_stock_multiplier": mean("safety_stock_multiplier"),
        "capacity_buffer_fraction": mean("capacity_buffer_fraction"),
        "capacity_pressure": mean("capacity_pressure"),
        "capacity_rejected": mean("capacity_rejected"),
        "no_vehicle_available": sum(item.no_vehicle_available for item in episodes),
        "dispatch_success_per_attempt": mean("dispatch_success_per_attempt"),
        "dispatch_rate": mean("dispatch_rate"),
        "hold_rate": mean("hold_rate"),
        "dispatch_distribution": sum_counter("dispatch_distribution"),
        "route_distribution": sum_counter("route_distribution"),
        "fleet_distribution": sum_counter("fleet_distribution"),
        "reorder_mode_distribution": sum_counter("reorder_mode_distribution"),
        "action_distribution": action_distribution,
        "top_action_ids": top_action_ids,
        "action_24_count": action_24_count,
        "action_24_percentage": action_24_count / total_steps if total_steps else None,
        "action_25_count": action_25_count,
        "action_25_percentage": action_25_count / total_steps if total_steps else None,
        "action_24_25_concentration": mean("action_24_25_concentration"),
        "dispatch_success_by_action_id": dispatch_success_by_action_id,
        "failed_noop_dispatch_by_action_id": failed_noop_dispatch_by_action_id,
        "action_attempts_by_id": action_attempts_by_id,
        "action_successes_by_id": action_successes_by_id,
        "action_failed_noop_by_id": action_failed_noop_by_id,
        "action_hold_by_id": action_hold_by_id,
        "action_dispatch_by_id": action_dispatch_by_id,
        "action_no_vehicle_by_id": action_no_vehicle_by_id,
        "action_already_assigned_by_id": action_already_assigned_by_id,
        "action_no_unassigned_by_id": action_no_unassigned_by_id,
        "action_no_current_work_by_id": action_no_current_work_by_id,
        "action_route_failure_by_id": action_route_failure_by_id,
        "action_dispatch_success_ratio_by_id": action_dispatch_success_ratio_by_id,
        "reward_component_sum_by_action_id": reward_component_sum_by_action_id,
        "reward_component_count_by_action_id": reward_component_count_by_action_id,
        "reward_component_mean_by_action_id": _nested_mean_map(
            reward_component_sum_by_action_id,
            reward_component_count_by_action_id,
        ),
        "reward_component_sum_by_action_family": reward_component_sum_by_action_family,
        "reward_component_count_by_action_family": reward_component_count_by_action_family,
        "reward_component_mean_by_action_family": _nested_mean_map(
            reward_component_sum_by_action_family,
            reward_component_count_by_action_family,
        ),
        "scenario_action_attempts_by_id": action_attempts_by_id,
        "scenario_action_successes_by_id": action_successes_by_id,
        "scenario_action_failed_noop_by_id": action_failed_noop_by_id,
        "scenario_action_dispatch_success_ratio_by_id": action_dispatch_success_ratio_by_id,
        "top_failed_noop_actions": top_failed_noop_actions,
        "top_failed_noop_actions_json": top_failed_noop_actions,
        "top_low_success_dispatch_actions": top_low_success_dispatch_actions,
        "action31_attempts": action31_attempts,
        "action31_successes": action31_successes,
        "action31_failed_noop": action31_failed_noop,
        "action31_dispatch_success_ratio": ratio(action31_successes, action31_dispatches),
        "action31_no_vehicle": int(action_no_vehicle_by_id.get("31", 0)),
        "action31_already_assigned": int(action_already_assigned_by_id.get("31", 0)),
        "action31_no_unassigned": int(action_no_unassigned_by_id.get("31", 0)),
        "action31_no_current_work": int(action_no_current_work_by_id.get("31", 0)),
        "action31_route_failure": int(action_route_failure_by_id.get("31", 0)),
        "route_distribution_successful_dispatch": route_successful_dispatch,
        "route_distribution_failed_dispatch": route_failed_dispatch,
        "route_distribution_hold": route_hold,
        "route_distribution_by_action_id": sum_nested_counter("route_distribution_by_action_id"),
        "route_success_count_by_route": route_success_count_by_route,
        "route_failed_noop_count_by_route": route_failed_noop_count_by_route,
        "route_dispatch_success_ratio_by_route": route_dispatch_success_ratio_by_route,
        "scenario_route_distribution_successful_dispatch": route_successful_dispatch,
        "scenario_route_distribution_failed_dispatch": route_failed_dispatch,
        "scenario_route_distribution_hold": route_hold,
        "scenario_route_dispatch_success_ratio_by_route": route_dispatch_success_ratio_by_route,
        "route_successful_dispatch_json": route_successful_dispatch,
        "route_failed_dispatch_json": route_failed_dispatch,
        "route_candidate_score_sum_by_route": route_candidate_score_sum_by_route,
        "route_candidate_score_count_by_route": route_candidate_score_count_by_route,
        "route_candidate_score_mean_by_route": route_candidate_score_mean_by_route,
        "selected_route_candidate_score_sum_by_route": selected_route_candidate_score_sum_by_route,
        "selected_route_candidate_score_count_by_route": selected_route_candidate_score_count_by_route,
        "selected_route_candidate_score_mean_by_route": selected_route_candidate_score_mean_by_route,
        "selected_route_rank_counts": selected_route_rank_counts,
        "selected_route_best_count": sum_int("selected_route_best_count"),
        "selected_route_near_best_count": sum_int("selected_route_near_best_count"),
        "selected_route_margin_to_best_sum": selected_route_margin_to_best_sum,
        "selected_route_margin_to_best_mean": selected_route_margin_to_best_mean,
        "shortest_near_best_but_not_selected_count": sum_int("shortest_near_best_but_not_selected_count"),
        "high_resilience_selected_when_shortest_near_best_count": sum_int(
            "high_resilience_selected_when_shortest_near_best_count"
        ),
        "high_resilience_selected_when_not_best_count": sum_int("high_resilience_selected_when_not_best_count"),
        "mixed_route_overconservative_candidate_count": sum_int("mixed_route_overconservative_candidate_count"),
        "dqn_local_negative_positive_train_rows": sum(
            item.dqn_local_negative_positive_train_rows for item in episodes
        ),
        "already_assigned_context_rows": sum(item.already_assigned_context_rows for item in episodes),
        "active_assigned_in_transit_orders": sum(
            item.active_assigned_in_transit_orders or 0 for item in episodes
        ),
        "active_assigned_in_transit_order_examples": active_order_examples,
        "stuck_assigned_in_transit_orders": sum(item.stuck_assigned_in_transit_orders or 0 for item in episodes),
        "stuck_assigned_in_transit_order_examples": stuck_order_examples,
        "route_failures": sum(item.route_failures for item in episodes),
        "customer_revisited": sum(item.customer_revisited for item in episodes),
        "fake_dispatch_credit": sum(item.fake_dispatch_credit for item in episodes),
        "route_failure_positive_dispatch_credit": sum(
            item.route_failure_positive_dispatch_credit for item in episodes
        ),
        "mixed_success_route_failure_steps": sum(item.mixed_success_route_failure_steps for item in episodes),
        "no_current_work_dqn_delivery_credit": sum(
            item.no_current_work_dqn_delivery_credit for item in episodes
        ),
        "hold_delivery_credit_leak": sum(item.hold_delivery_credit_leak for item in episodes),
        "action8_route_or_delivery_credit_leak": sum(
            item.action8_route_or_delivery_credit_leak for item in episodes
        ),
        "unsafe_24_25_candidate_credit": sum(item.unsafe_24_25_candidate_credit for item in episodes),
        "no_work_positive_dqn_local": sum(item.no_work_positive_dqn_local for item in episodes),
        "emergency_zero_useful_positive_credit": sum(
            item.emergency_zero_useful_positive_credit for item in episodes
        ),
        "premium_sla_fleet_credit_blocked_no_current_work": sum(
            item.premium_sla_fleet_credit_blocked_no_current_work for item in episodes
        ),
        "premium_primary_adaptation_credit_blocked_no_current_work": sum(
            item.premium_primary_adaptation_credit_blocked_no_current_work for item in episodes
        ),
        "premium_fleet_credit_allowed": sum(item.premium_fleet_credit_allowed for item in episodes),
        "premium_pressure_steps": premium_pressure_steps,
        "premium_hold_steps": premium_hold_steps,
        "premium_dispatch_steps": premium_dispatch_steps,
        "premium_hold_under_useful_dispatch_opportunity": premium_hold_under_useful_dispatch,
        "premium_dispatch_under_useful_dispatch_opportunity": premium_dispatch_under_useful_dispatch,
        "premium_no_vehicle_steps": sum_int("premium_no_vehicle_steps"),
        "premium_already_assigned_steps": sum_int("premium_already_assigned_steps"),
        "premium_no_unassigned_steps": sum_int("premium_no_unassigned_steps"),
        "premium_reorder_none_steps": sum_int("premium_reorder_none_steps"),
        "premium_reorder_conservative_steps": sum_int("premium_reorder_conservative_steps"),
        "premium_reorder_aggressive_steps": sum_int("premium_reorder_aggressive_steps"),
        "premium_reorder_emergency_steps": sum_int("premium_reorder_emergency_steps"),
        "premium_useful_reorder_opportunity_steps": premium_useful_reorder_opportunity_steps,
        "premium_reorder_missed_opportunity_steps": premium_reorder_missed_opportunity_steps,
        "premium_service_pressure_steps": sum_int("premium_service_pressure_steps"),
        "premium_late_or_at_risk_backlog_steps": sum_int("premium_late_or_at_risk_backlog_steps"),
        "hold_primary_under_premium_pressure": hold_primary_under_premium_pressure,
        "dispatch_primary_under_premium_pressure": sum_int("dispatch_primary_under_premium_pressure"),
        "premium_hold_rate": ratio(premium_hold_steps, premium_pressure_steps),
        "premium_dispatch_rate": ratio(premium_dispatch_steps, premium_pressure_steps),
        "premium_missed_useful_dispatch_rate": ratio(
            premium_hold_under_useful_dispatch,
            premium_hold_under_useful_dispatch + premium_dispatch_under_useful_dispatch,
        ),
        "premium_missed_useful_reorder_rate": ratio(
            premium_reorder_missed_opportunity_steps,
            premium_useful_reorder_opportunity_steps,
        ),
        "hold_primary_under_premium_pressure_rate": ratio(
            hold_primary_under_premium_pressure,
            premium_pressure_steps,
        ),
        "transport_cost_delta": mean("transport_cost_delta"),
        "emergency_aggressive_reorder_usage": mean("emergency_aggressive_reorder_usage"),
        "nan_inf_detected": any(item.nan_inf_detected for item in episodes),
    }


def evaluate_global_hard_blockers(summary: Mapping[str, Any]) -> list[str]:
    """Return globally fatal hard-blocker failures, independent of scenario config."""

    failures: list[str] = []
    for key, label in GLOBAL_FATAL_HARD_BLOCKER_FIELDS:
        if key == "nan_inf_detected":
            if summary.get(key):
                failures.append("NaN/Inf detected")
            continue
        value = _summary_numeric(summary, key)
        if value is not None and value > 0:
            display = int(value) if float(value).is_integer() else value
            failures.append(f"{label} present: {display}")
    return failures


def evaluate_thresholds(summary: Mapping[str, Any], thresholds: Mapping[str, Any]) -> list[str]:
    """Return human-readable threshold failures for a scenario summary."""

    failures: list[str] = evaluate_global_hard_blockers(summary)

    def numeric(name: str) -> float | None:
        return _summary_numeric(summary, name)

    def delivered_metric() -> float | None:
        value = numeric("mean_step_delivered_delta")
        return value if value is not None else numeric("delivered_delta")

    def configured_min_delivery(default: float = 0.45) -> float:
        return _as_float(
            thresholds.get(
                "min_mean_step_delivered_delta",
                thresholds.get("min_delivered_delta", default),
            ),
            default,
        )

    def operational_degradation() -> bool:
        service = numeric("service_level")
        lateness = numeric("true_lateness_pressure")
        delivered = delivered_metric()
        dispatch_success = numeric("dispatch_success_per_attempt")
        min_service = _as_float(thresholds.get("min_service_level"), 0.93)
        max_lateness = _as_float(thresholds.get("max_true_lateness_pressure"), 0.10)
        min_delivered = configured_min_delivery()
        min_dispatch = _as_float(thresholds.get("min_dispatch_success_per_attempt"), 0.70)
        return (
            (service is not None and service < min_service)
            or (lateness is not None and lateness > max_lateness)
            or (delivered is not None and delivered < min_delivered)
            or (dispatch_success is not None and dispatch_success < min_dispatch)
        )

    hard_zero_keys = {
        "no_vehicle_available": "no_vehicle_available",
        "stuck_assigned_in_transit_orders": "stuck assigned/in-transit orders",
    }
    for key, label in hard_zero_keys.items():
        if thresholds.get(f"fail_on_{key}", False) and numeric(key) and numeric(key) > 0:
            failures.append(f"{label} present: {summary.get(key)}")

    min_dispatch = thresholds.get("min_dispatch_success_per_attempt")
    if min_dispatch is not None and numeric("dispatch_success_per_attempt") is not None:
        if numeric("dispatch_success_per_attempt") < float(min_dispatch):
            failures.append(
                f"dispatch_success_per_attempt {numeric('dispatch_success_per_attempt'):.3f} < {float(min_dispatch):.3f}"
            )

    min_service = thresholds.get("min_service_level")
    if min_service is not None and numeric("service_level") is not None:
        if numeric("service_level") < float(min_service):
            failures.append(f"service_level {numeric('service_level'):.3f} < {float(min_service):.3f}")

    max_lateness = thresholds.get("max_true_lateness_pressure")
    if max_lateness is not None and numeric("true_lateness_pressure") is not None:
        if numeric("true_lateness_pressure") > float(max_lateness):
            failures.append(
                f"true_lateness_pressure {numeric('true_lateness_pressure'):.3f} > {float(max_lateness):.3f}"
            )

    min_mean_delivered = thresholds.get("min_mean_step_delivered_delta")
    legacy_min_delivered = thresholds.get("min_delivered_delta")
    min_delivered = min_mean_delivered if min_mean_delivered is not None else legacy_min_delivered
    delivered = delivered_metric()
    if min_delivered is not None and delivered is not None:
        if delivered < float(min_delivered):
            if min_mean_delivered is not None:
                failures.append(
                    f"mean_step_delivered_delta {delivered:.3f} < {float(min_delivered):.3f}"
                )
            else:
                failures.append(
                    f"mean_step_delivered_delta {delivered:.3f} < legacy min_delivered_delta {float(min_delivered):.3f}"
                )

    max_action = thresholds.get(
        "max_action_24_25_concentration_fail",
        thresholds.get("max_action_24_25_concentration"),
    )
    if max_action is not None and numeric("action_24_25_concentration") is not None:
        if numeric("action_24_25_concentration") > float(max_action):
            require_degradation = bool(
                thresholds.get("require_operational_degradation_for_action_concentration_fail", False)
            )
            if not require_degradation or operational_degradation():
                failures.append(
                    f"action_24_25_concentration {numeric('action_24_25_concentration'):.3f} > {float(max_action):.3f}"
                )

    max_masking = thresholds.get("max_dqn_local_negative_positive_train_rows")
    if max_masking is not None and numeric("dqn_local_negative_positive_train_rows") is not None:
        if numeric("dqn_local_negative_positive_train_rows") > float(max_masking):
            failures.append(
                "dqn_local_negative_positive_train_rows "
                f"{numeric('dqn_local_negative_positive_train_rows'):.1f} > {float(max_masking):.1f}"
            )

    min_capacity = thresholds.get("min_capacity_buffer_fraction")
    if min_capacity is not None and numeric("capacity_buffer_fraction") is not None:
        if numeric("capacity_buffer_fraction") < float(min_capacity):
            failures.append(
                f"capacity_buffer_fraction {numeric('capacity_buffer_fraction'):.3f} < {float(min_capacity):.3f}"
            )

    return failures


def evaluate_warnings(summary: Mapping[str, Any], thresholds: Mapping[str, Any]) -> list[str]:
    """Return non-hard-fail warnings for diagnostic-only scenario concerns."""

    warnings: list[str] = []

    def numeric(name: str) -> float | None:
        value = summary.get(name)
        return _as_float(value) if value is not None else None

    max_action_warn = thresholds.get(
        "max_action_24_25_concentration_warn",
        thresholds.get("max_action_24_25_concentration"),
    )
    if max_action_warn is not None and numeric("action_24_25_concentration") is not None:
        if numeric("action_24_25_concentration") > float(max_action_warn):
            warnings.append(
                "action_24_25_concentration "
                f"{numeric('action_24_25_concentration'):.3f} > warning {float(max_action_warn):.3f}"
            )

    if numeric("mixed_success_route_failure_steps") and numeric("mixed_success_route_failure_steps") > 0:
        warnings.append(f"mixed_success_route_failure_steps present: {summary.get('mixed_success_route_failure_steps')}")

    return warnings
