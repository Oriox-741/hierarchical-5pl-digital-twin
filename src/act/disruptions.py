"""Disruption processes for node failures, arc delays, demand shocks, and capacity loss."""

from __future__ import annotations

import random
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID, uuid4

import simpy

from src.act.inventory_dynamics import InventoryNetwork
from src.act.resources import CapacityState
from src.act.routing_physics import RoutingPhysics
from src.shared.metrics import clamp
from src.shared.types import EventKind, JsonDict


class DisruptionKind(StrEnum):
    NODE_FAILURE = "node_failure"
    ARC_DELAY = "arc_delay"
    DEMAND_SHOCK = "demand_shock"
    CAPACITY_LOSS = "capacity_loss"


@dataclass(frozen=True, slots=True)
class DisruptionEvent:
    event_id: UUID
    kind: DisruptionKind
    starts_at: float
    duration: float
    severity: float
    target_id: UUID
    payload: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.starts_at < 0.0:
            raise ValueError("starts_at must not be negative.")
        if self.duration < 0.0:
            raise ValueError("duration must not be negative.")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be within [0, 1].")


class DisruptionManager:
    """Schedules and applies disruptions as SimPy processes."""

    def __init__(
        self,
        env: simpy.Environment,
        *,
        routing_physics: RoutingPhysics,
        inventory_network: InventoryNetwork | None = None,
        capacity_states: dict[UUID, CapacityState] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.env = env
        self.routing_physics = routing_physics
        self.inventory_network = inventory_network
        self.capacity_states = capacity_states or {}
        self.rng = rng or random.Random()
        self.active_events: dict[UUID, DisruptionEvent] = {}
        self.event_log: list[JsonDict] = []

    def schedule(self, event: DisruptionEvent) -> simpy.Process:
        return self.env.process(self._run_event(event))

    def schedule_arc_delay(
        self,
        *,
        origin_node_id: UUID,
        destination_node_id: UUID,
        starts_at: float,
        duration: float,
        severity: float,
    ) -> simpy.Process:
        return self.schedule(
            DisruptionEvent(
                event_id=uuid4(),
                kind=DisruptionKind.ARC_DELAY,
                starts_at=starts_at,
                duration=duration,
                severity=severity,
                target_id=origin_node_id,
                payload={"origin_node_id": str(origin_node_id), "destination_node_id": str(destination_node_id)},
            )
        )

    def random_event(
        self,
        *,
        target_id: UUID,
        starts_at: float,
        duration_range: tuple[float, float] = (300.0, 3_600.0),
        severity_range: tuple[float, float] = (0.1, 0.8),
    ) -> DisruptionEvent:
        kind = self.rng.choice(tuple(DisruptionKind))
        return DisruptionEvent(
            event_id=uuid4(),
            kind=kind,
            starts_at=starts_at,
            duration=self.rng.uniform(*duration_range),
            severity=self.rng.uniform(*severity_range),
            target_id=target_id,
        )

    def disruption_score(self) -> float:
        if not self.active_events:
            return 0.0
        return clamp(max(event.severity for event in self.active_events.values()), 0.0, 1.0)

    def _run_event(self, event: DisruptionEvent) -> Generator[simpy.Event, None, None]:
        yield self.env.timeout(max(0.0, event.starts_at - self.env.now))
        self.active_events[event.event_id] = event
        self._log(event, "started")
        self._apply(event)
        yield self.env.timeout(event.duration)
        self._recover(event)
        self.active_events.pop(event.event_id, None)
        self._log(event, "recovered")

    def _apply(self, event: DisruptionEvent) -> None:
        match event.kind:
            case DisruptionKind.ARC_DELAY:
                origin = UUID(str(event.payload.get("origin_node_id", event.target_id)))
                destination_value = event.payload.get("destination_node_id")
                if destination_value is not None:
                    self.routing_physics.set_arc_delay_multiplier(origin, UUID(str(destination_value)), 1.0 + 4.0 * event.severity)
            case DisruptionKind.NODE_FAILURE | DisruptionKind.CAPACITY_LOSS:
                state = self.capacity_states.get(event.target_id)
                if state is not None:
                    state.reduce(event.severity)
            case DisruptionKind.DEMAND_SHOCK:
                if self.inventory_network is not None:
                    for position in self.inventory_network.positions.values():
                        if position.node_id == event.target_id:
                            position.update_forecast(position.demand_mean * (1.0 + event.severity), alpha=0.8)

    def _recover(self, event: DisruptionEvent) -> None:
        match event.kind:
            case DisruptionKind.ARC_DELAY:
                origin = UUID(str(event.payload.get("origin_node_id", event.target_id)))
                destination_value = event.payload.get("destination_node_id")
                if destination_value is not None:
                    self.routing_physics.clear_arc_delay(origin, UUID(str(destination_value)))
            case DisruptionKind.NODE_FAILURE | DisruptionKind.CAPACITY_LOSS:
                state = self.capacity_states.get(event.target_id)
                if state is not None:
                    state.restore(event.severity)
            case DisruptionKind.DEMAND_SHOCK:
                return

    def _log(self, event: DisruptionEvent, status: str) -> None:
        self.event_log.append(
            {
                "event_id": str(event.event_id),
                "kind": event.kind.value,
                "event_kind": EventKind.DISRUPTION.value,
                "status": status,
                "simulation_time": self.env.now,
                "duration": event.duration,
                "severity": event.severity,
                "target_id": str(event.target_id),
                "payload": event.payload,
            }
        )
