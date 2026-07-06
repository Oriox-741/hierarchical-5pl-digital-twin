"""Map DQN discrete outputs into categorical logistics decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Final


class DispatchDecision(IntEnum):
    HOLD = 0
    DISPATCH = 1


class RouteDecision(IntEnum):
    SHORTEST = 0
    LOW_CONGESTION = 1
    HIGH_RESILIENCE = 2


class ModeDecision(IntEnum):
    SECONDARY_FLEET = 0
    PRIMARY_FLEET = 1


class ReorderDecision(IntEnum):
    NONE = 0
    CONSERVATIVE = 1
    AGGRESSIVE = 2
    EMERGENCY = 3


class DiscreteDecisionKind(StrEnum):
    DISPATCH = "dispatch"
    ROUTE = "route"
    MODE = "mode"
    REORDER = "reorder"


DISCRETE_ACTION_COUNT: Final[int] = (
    len(DispatchDecision) * len(RouteDecision) * len(ModeDecision) * len(ReorderDecision)
)


@dataclass(frozen=True, slots=True)
class DiscreteLogisticsAction:
    dispatch: DispatchDecision
    route: RouteDecision
    mode: ModeDecision
    reorder: ReorderDecision

    def as_dict(self) -> dict[str, str]:
        return {
            DiscreteDecisionKind.DISPATCH.value: self.dispatch.name.lower(),
            DiscreteDecisionKind.ROUTE.value: self.route.name.lower(),
            DiscreteDecisionKind.MODE.value: self.mode.name.lower(),
            DiscreteDecisionKind.REORDER.value: self.reorder.name.lower(),
        }


class DiscreteActionMapper:
    """Decode a single DQN integer into structured logistics categories."""

    @property
    def action_count(self) -> int:
        return DISCRETE_ACTION_COUNT

    def map(self, action: int) -> DiscreteLogisticsAction:
        if not 0 <= int(action) < DISCRETE_ACTION_COUNT:
            raise ValueError(f"discrete action must be in [0, {DISCRETE_ACTION_COUNT - 1}].")

        value = int(action)
        reorder_mod = len(ReorderDecision)
        mode_mod = len(ModeDecision)
        route_mod = len(RouteDecision)

        reorder = ReorderDecision(value % reorder_mod)
        value //= reorder_mod
        mode = ModeDecision(value % mode_mod)
        value //= mode_mod
        route = RouteDecision(value % route_mod)
        value //= route_mod
        dispatch = DispatchDecision(value)

        return DiscreteLogisticsAction(dispatch=dispatch, route=route, mode=mode, reorder=reorder)

    def encode(self, action: DiscreteLogisticsAction) -> int:
        value = int(action.dispatch)
        value = (value * len(RouteDecision)) + int(action.route)
        value = (value * len(ModeDecision)) + int(action.mode)
        value = (value * len(ReorderDecision)) + int(action.reorder)
        return value

