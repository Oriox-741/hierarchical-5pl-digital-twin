"""Shared metric helpers for operational and learning feedback."""

from __future__ import annotations

from collections.abc import Sequence


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a value to a closed interval."""
    if lower > upper:
        raise ValueError("lower must be less than or equal to upper.")
    return max(lower, min(upper, value))


def service_level(delivered_on_time: int, total_shipments: int) -> float:
    """Compute on-time service level in [0, 1]."""
    if total_shipments <= 0:
        return 0.0
    return clamp(delivered_on_time / total_shipments, 0.0, 1.0)


def capacity_utilization(used_capacity: float, total_capacity: float) -> float:
    """Compute capacity utilization in [0, 1] when capacity is known."""
    if total_capacity <= 0.0:
        return 0.0
    return clamp(used_capacity / total_capacity, 0.0, 1.0)


def backlog_ratio(backlog_units: float, demand_units: float) -> float:
    """Compute unmet-demand pressure in [0, 1]."""
    if demand_units <= 0.0:
        return 0.0
    return clamp(backlog_units / demand_units, 0.0, 1.0)


def safety_potential(
    *,
    congestion_score: float,
    disruption_score: float,
    capacity_pressure: float,
    backlog_pressure: float,
    weights: Sequence[float] = (0.25, 0.30, 0.25, 0.20),
) -> float:
    """Compute a bounded proactive risk score used by rewards and action projection."""
    signals = (congestion_score, disruption_score, capacity_pressure, backlog_pressure)
    if len(weights) != len(signals):
        raise ValueError("weights must match the number of safety signals.")

    normalizer = sum(weights)
    if normalizer <= 0.0:
        raise ValueError("weights must have positive total mass.")

    weighted = sum(clamp(signal, 0.0, 1.0) * weight for signal, weight in zip(signals, weights))
    return clamp(weighted / normalizer, 0.0, 1.0)
