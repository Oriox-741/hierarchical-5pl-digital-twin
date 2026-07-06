"""Hard action-constraint filter before commands reach the SimPy simulation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from src.act.action_projector import PhysicalAction
from src.act.discrete_action_mapper import DispatchDecision, DiscreteLogisticsAction, ReorderDecision
from src.shared.metrics import clamp


@dataclass(frozen=True, slots=True)
class SafetyEnvelope:
    max_safety_potential: float = 0.85
    max_capacity_utilization: float = 0.95
    max_backlog_pressure: float = 0.80
    max_speed_multiplier: float = 1.25
    min_speed_multiplier: float = 0.50


@dataclass(frozen=True, slots=True)
class SafetyProjectionResult:
    action: PhysicalAction | DiscreteLogisticsAction
    projected: bool
    blocked: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


class SafetyProjector:
    """Project or block unsafe actions using Safety Potential and hard limits."""

    def __init__(self, envelope: SafetyEnvelope | None = None) -> None:
        self.envelope = envelope or SafetyEnvelope()

    def project_continuous(
        self,
        action: PhysicalAction,
        *,
        safety_potential: float,
        capacity_utilization: float,
        backlog_pressure: float,
    ) -> SafetyProjectionResult:
        reasons: list[str] = []
        projected = False

        speed_multiplier = clamp(
            action.speed_multiplier,
            self.envelope.min_speed_multiplier,
            self.envelope.max_speed_multiplier,
        )
        if speed_multiplier != action.speed_multiplier:
            projected = True
            reasons.append("speed_multiplier_clamped")

        dispatch_intensity = action.dispatch_intensity
        if safety_potential >= self.envelope.max_safety_potential:
            dispatch_intensity = min(dispatch_intensity, 0.25)
            projected = True
            reasons.append("safety_potential_dispatch_limited")

        if capacity_utilization >= self.envelope.max_capacity_utilization:
            dispatch_intensity = 0.0
            projected = True
            reasons.append("capacity_dispatch_blocked")

        reorder_fraction = action.reorder_fraction
        if backlog_pressure >= self.envelope.max_backlog_pressure:
            reorder_fraction = max(reorder_fraction, 0.75)
            projected = True
            reasons.append("backlog_reorder_raised")

        safe_action = PhysicalAction(
            reorder_fraction=reorder_fraction,
            dispatch_intensity=dispatch_intensity,
            speed_multiplier=speed_multiplier,
            safety_stock_multiplier=action.safety_stock_multiplier,
            capacity_buffer_fraction=action.capacity_buffer_fraction,
        )
        return SafetyProjectionResult(
            action=safe_action,
            projected=projected,
            blocked=dispatch_intensity <= 0.0 and action.dispatch_intensity > 0.0,
            reasons=tuple(reasons),
        )

    def project_discrete(
        self,
        action: DiscreteLogisticsAction,
        *,
        safety_potential: float,
        capacity_utilization: float,
        backlog_pressure: float,
    ) -> SafetyProjectionResult:
        reasons: list[str] = []
        projected = False
        safe_action = action

        if safety_potential >= self.envelope.max_safety_potential and action.dispatch is DispatchDecision.DISPATCH:
            safe_action = replace(safe_action, dispatch=DispatchDecision.HOLD)
            projected = True
            reasons.append("safety_potential_dispatch_hold")

        if capacity_utilization >= self.envelope.max_capacity_utilization and safe_action.dispatch is DispatchDecision.DISPATCH:
            safe_action = replace(safe_action, dispatch=DispatchDecision.HOLD)
            projected = True
            reasons.append("capacity_dispatch_hold")

        if backlog_pressure >= self.envelope.max_backlog_pressure and safe_action.reorder is ReorderDecision.NONE:
            safe_action = replace(safe_action, reorder=ReorderDecision.AGGRESSIVE)
            projected = True
            reasons.append("backlog_reorder_forced")

        return SafetyProjectionResult(
            action=safe_action,
            projected=projected,
            blocked=action.dispatch is DispatchDecision.DISPATCH and safe_action.dispatch is DispatchDecision.HOLD,
            reasons=tuple(reasons),
        )

