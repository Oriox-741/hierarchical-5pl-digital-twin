"""Deterministic rule-based logistics baselines for future benchmark adapters.

This module is intentionally pure: it does not instantiate environments, load
checkpoints, write reports, or run evaluation loops.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from src.act.discrete_action_mapper import (
    DispatchDecision,
    DiscreteActionMapper,
    DiscreteLogisticsAction,
    ModeDecision,
    ReorderDecision,
    RouteDecision,
)


_MAPPER = DiscreteActionMapper()


def encode_action_id(action: DiscreteLogisticsAction) -> int:
    return _MAPPER.encode(action)


def decode_action_id(action_id: int) -> DiscreteLogisticsAction:
    return _MAPPER.map(action_id)


def _action(
    dispatch: DispatchDecision,
    route: RouteDecision,
    mode: ModeDecision,
    reorder: ReorderDecision,
) -> int:
    return encode_action_id(
        DiscreteLogisticsAction(dispatch=dispatch, route=route, mode=mode, reorder=reorder)
    )


ACTION_HOLD = _action(
    DispatchDecision.HOLD,
    RouteDecision.SHORTEST,
    ModeDecision.SECONDARY_FLEET,
    ReorderDecision.NONE,
)
ACTION_DISPATCH_SHORTEST_SECONDARY_NONE = _action(
    DispatchDecision.DISPATCH,
    RouteDecision.SHORTEST,
    ModeDecision.SECONDARY_FLEET,
    ReorderDecision.NONE,
)
ACTION_DISPATCH_SHORTEST_PRIMARY_NONE = _action(
    DispatchDecision.DISPATCH,
    RouteDecision.SHORTEST,
    ModeDecision.PRIMARY_FLEET,
    ReorderDecision.NONE,
)
ACTION_DISPATCH_SHORTEST_PRIMARY_CONSERVATIVE = _action(
    DispatchDecision.DISPATCH,
    RouteDecision.SHORTEST,
    ModeDecision.PRIMARY_FLEET,
    ReorderDecision.CONSERVATIVE,
)
ACTION_DISPATCH_SHORTEST_PRIMARY_EMERGENCY = _action(
    DispatchDecision.DISPATCH,
    RouteDecision.SHORTEST,
    ModeDecision.PRIMARY_FLEET,
    ReorderDecision.EMERGENCY,
)
ACTION_DISPATCH_LOW_CONGESTION_SECONDARY_NONE = _action(
    DispatchDecision.DISPATCH,
    RouteDecision.LOW_CONGESTION,
    ModeDecision.SECONDARY_FLEET,
    ReorderDecision.NONE,
)
ACTION_DISPATCH_LOW_CONGESTION_PRIMARY_NONE = _action(
    DispatchDecision.DISPATCH,
    RouteDecision.LOW_CONGESTION,
    ModeDecision.PRIMARY_FLEET,
    ReorderDecision.NONE,
)


@dataclass(frozen=True, slots=True)
class OrderView:
    order_id: str
    created_at: float
    due_time: float
    total_units: float = 0.0
    premium: bool = False
    urgent: bool = False


@dataclass(frozen=True, slots=True)
class RuleBasedDecisionContext:
    dispatchable_orders: tuple[OrderView, ...] = ()
    primary_vehicle_available: bool = False
    secondary_vehicle_available: bool = False
    route_disruption_pressure: float = 0.0
    congestion_pressure: float = 0.0
    stockout_risk: float = 0.0
    holding_cost_pressure: float = 0.0
    safety_stock_gap: float = 0.0
    inventory_coverage: float = 1.0


@dataclass(frozen=True, slots=True)
class RuleBasedDecision:
    action_id: int
    selected_order_id: str | None = None
    baseline_name: str = ""
    reason: str = ""


Selector = Callable[[RuleBasedDecisionContext], RuleBasedDecision]


@dataclass(frozen=True, slots=True)
class RuleBasedBaseline:
    name: str
    description: str
    _selector: Selector

    def select(self, context: RuleBasedDecisionContext) -> RuleBasedDecision:
        decision = self._selector(context)
        return RuleBasedDecision(
            action_id=decision.action_id,
            selected_order_id=decision.selected_order_id,
            baseline_name=self.name,
            reason=decision.reason,
        )


def _hold(reason: str) -> RuleBasedDecision:
    return RuleBasedDecision(action_id=ACTION_HOLD, selected_order_id=None, reason=reason)


def _dispatch(action_id: int, order: OrderView, reason: str) -> RuleBasedDecision:
    return RuleBasedDecision(action_id=action_id, selected_order_id=order.order_id, reason=reason)


def _oldest_order(context: RuleBasedDecisionContext) -> OrderView | None:
    return min(context.dispatchable_orders, key=lambda order: (order.created_at, order.due_time, order.order_id), default=None)


def _earliest_due_order(context: RuleBasedDecisionContext) -> OrderView | None:
    return min(context.dispatchable_orders, key=lambda order: (order.due_time, order.created_at, order.order_id), default=None)


def _premium_first_order(context: RuleBasedDecisionContext) -> OrderView | None:
    return min(
        context.dispatchable_orders,
        key=lambda order: (
            0 if order.premium or order.urgent else 1,
            order.due_time,
            order.created_at,
            order.order_id,
        ),
        default=None,
    )


def _has_dispatchable_work(context: RuleBasedDecisionContext) -> bool:
    return bool(context.dispatchable_orders)


def _route_pressure(context: RuleBasedDecisionContext) -> float:
    return max(context.route_disruption_pressure, context.congestion_pressure)


def _moderate_inventory_pressure(context: RuleBasedDecisionContext) -> bool:
    return (
        context.stockout_risk >= 0.25
        or context.safety_stock_gap >= 0.50
        or context.inventory_coverage <= 0.50
    )


def _severe_inventory_pressure(context: RuleBasedDecisionContext) -> bool:
    return (
        context.stockout_risk >= 0.80
        or context.safety_stock_gap >= 0.80
        or context.inventory_coverage <= 0.20
    )


def _fifo_shortest_primary_none(context: RuleBasedDecisionContext) -> RuleBasedDecision:
    if not _has_dispatchable_work(context):
        return _hold("no_dispatchable_work")
    if not context.primary_vehicle_available:
        return _hold("primary_vehicle_unavailable")
    order = _oldest_order(context)
    if order is None:
        return _hold("no_dispatchable_work")
    return _dispatch(ACTION_DISPATCH_SHORTEST_PRIMARY_NONE, order, "fifo_shortest_primary_none")


def _earliest_due_shortest_primary_none(context: RuleBasedDecisionContext) -> RuleBasedDecision:
    if not _has_dispatchable_work(context):
        return _hold("no_dispatchable_work")
    if not context.primary_vehicle_available:
        return _hold("primary_vehicle_unavailable")
    order = _earliest_due_order(context)
    if order is None:
        return _hold("no_dispatchable_work")
    return _dispatch(ACTION_DISPATCH_SHORTEST_PRIMARY_NONE, order, "earliest_due_shortest_primary_none")


def _premium_first_shortest_primary_secondary_none(context: RuleBasedDecisionContext) -> RuleBasedDecision:
    if not _has_dispatchable_work(context):
        return _hold("no_dispatchable_work")
    if not (context.primary_vehicle_available or context.secondary_vehicle_available):
        return _hold("no_vehicle_available")
    order = _premium_first_order(context)
    if order is None:
        return _hold("no_dispatchable_work")
    action_id = (
        ACTION_DISPATCH_SHORTEST_PRIMARY_NONE
        if context.primary_vehicle_available
        else ACTION_DISPATCH_SHORTEST_SECONDARY_NONE
    )
    return _dispatch(action_id, order, "premium_first_shortest_fleet_fallback")


def _low_congestion_under_disruption(context: RuleBasedDecisionContext) -> RuleBasedDecision:
    if not _has_dispatchable_work(context):
        return _hold("no_dispatchable_work")
    if not (context.primary_vehicle_available or context.secondary_vehicle_available):
        return _hold("no_vehicle_available")
    order = _earliest_due_order(context)
    if order is None:
        return _hold("no_dispatchable_work")
    use_low_congestion = _route_pressure(context) >= 0.50
    if context.primary_vehicle_available:
        action_id = (
            ACTION_DISPATCH_LOW_CONGESTION_PRIMARY_NONE
            if use_low_congestion
            else ACTION_DISPATCH_SHORTEST_PRIMARY_NONE
        )
    else:
        action_id = (
            ACTION_DISPATCH_LOW_CONGESTION_SECONDARY_NONE
            if use_low_congestion
            else ACTION_DISPATCH_SHORTEST_SECONDARY_NONE
        )
    return _dispatch(action_id, order, "low_congestion_under_disruption")


def _conservative_stock_threshold_reorder(context: RuleBasedDecisionContext) -> RuleBasedDecision:
    if not _has_dispatchable_work(context):
        return _hold("no_dispatchable_work")
    if not context.primary_vehicle_available:
        return _hold("primary_vehicle_unavailable")
    order = _earliest_due_order(context)
    if order is None:
        return _hold("no_dispatchable_work")
    use_conservative = _moderate_inventory_pressure(context) and context.holding_cost_pressure < 0.80
    action_id = (
        ACTION_DISPATCH_SHORTEST_PRIMARY_CONSERVATIVE
        if use_conservative
        else ACTION_DISPATCH_SHORTEST_PRIMARY_NONE
    )
    return _dispatch(action_id, order, "conservative_stock_threshold_reorder")


def _emergency_stockout_prevention_reorder(context: RuleBasedDecisionContext) -> RuleBasedDecision:
    if not _has_dispatchable_work(context):
        return _hold("no_dispatchable_work")
    if not context.primary_vehicle_available:
        return _hold("primary_vehicle_unavailable")
    order = _earliest_due_order(context)
    if order is None:
        return _hold("no_dispatchable_work")
    action_id = (
        ACTION_DISPATCH_SHORTEST_PRIMARY_EMERGENCY
        if _severe_inventory_pressure(context)
        else ACTION_DISPATCH_SHORTEST_PRIMARY_NONE
    )
    return _dispatch(action_id, order, "emergency_stockout_prevention_reorder")


def _vehicle_scarcity_primary_first_secondary_fallback(context: RuleBasedDecisionContext) -> RuleBasedDecision:
    if not _has_dispatchable_work(context):
        return _hold("no_dispatchable_work")
    if not (context.primary_vehicle_available or context.secondary_vehicle_available):
        return _hold("no_vehicle_available")
    order = _earliest_due_order(context)
    if order is None:
        return _hold("no_dispatchable_work")
    action_id = (
        ACTION_DISPATCH_SHORTEST_PRIMARY_NONE
        if context.primary_vehicle_available
        else ACTION_DISPATCH_SHORTEST_SECONDARY_NONE
    )
    return _dispatch(action_id, order, "vehicle_scarcity_fleet_fallback")


def _high_holding_no_overstock(context: RuleBasedDecisionContext) -> RuleBasedDecision:
    if not _has_dispatchable_work(context):
        return _hold("no_dispatchable_work")
    if not context.primary_vehicle_available:
        return _hold("primary_vehicle_unavailable")
    order = _earliest_due_order(context)
    if order is None:
        return _hold("no_dispatchable_work")
    return _dispatch(ACTION_DISPATCH_SHORTEST_PRIMARY_NONE, order, "high_holding_no_overstock")


def build_rule_based_baselines() -> Mapping[str, RuleBasedBaseline]:
    baselines = (
        RuleBasedBaseline(
            name="fifo_shortest_primary_none",
            description="Oldest eligible order, shortest route, primary fleet, no reorder.",
            _selector=_fifo_shortest_primary_none,
        ),
        RuleBasedBaseline(
            name="earliest_due_shortest_primary_none",
            description="Earliest due eligible order, shortest route, primary fleet, no reorder.",
            _selector=_earliest_due_shortest_primary_none,
        ),
        RuleBasedBaseline(
            name="premium_first_shortest_primary_secondary_none",
            description="Premium or urgent work first, primary fleet with secondary fallback, no reorder.",
            _selector=_premium_first_shortest_primary_secondary_none,
        ),
        RuleBasedBaseline(
            name="low_congestion_under_disruption",
            description="Earliest due dispatch with low-congestion route under route pressure.",
            _selector=_low_congestion_under_disruption,
        ),
        RuleBasedBaseline(
            name="conservative_stock_threshold_reorder",
            description="Conservative reorder when moderate inventory pressure appears.",
            _selector=_conservative_stock_threshold_reorder,
        ),
        RuleBasedBaseline(
            name="emergency_stockout_prevention_reorder",
            description="Emergency reorder under severe stockout or safety-stock pressure.",
            _selector=_emergency_stockout_prevention_reorder,
        ),
        RuleBasedBaseline(
            name="vehicle_scarcity_primary_first_secondary_fallback",
            description="Primary fleet first, secondary fleet fallback under scarcity.",
            _selector=_vehicle_scarcity_primary_first_secondary_fallback,
        ),
        RuleBasedBaseline(
            name="high_holding_no_overstock",
            description="Avoid reorder overrides under high holding-cost pressure.",
            _selector=_high_holding_no_overstock,
        ),
    )
    return {baseline.name: baseline for baseline in baselines}
