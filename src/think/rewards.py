"""Weighted multi-objective rewards and Safety Potential shaping."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.metrics import backlog_ratio, capacity_utilization, clamp, safety_potential


DEFAULT_REWARD_CONFIG = Path("configs/reward_weights.json")


@dataclass(frozen=True, slots=True)
class RewardWeights:
    holding_cost: float
    shipping_cost: float
    late_delivery: float
    backlog: float
    stockout: float
    capacity_overflow: float
    service_level: float
    throughput: float
    resilience: float
    safety_potential: float
    cash_conversion_cycle: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RewardWeights:
        return cls(**{field: float(data[field]) for field in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class RewardComponents:
    service_level: float = 0.0
    throughput: float = 0.0
    resilience: float = 0.0
    holding_cost: float = 0.0
    shipping_cost: float = 0.0
    late_delivery: float = 0.0
    backlog: float = 0.0
    stockout: float = 0.0
    capacity_overflow: float = 0.0
    cash_conversion_cycle: float = 0.0
    safety_potential: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "service_level": self.service_level,
            "throughput": self.throughput,
            "resilience": self.resilience,
            "holding_cost": self.holding_cost,
            "shipping_cost": self.shipping_cost,
            "late_delivery": self.late_delivery,
            "backlog": self.backlog,
            "stockout": self.stockout,
            "capacity_overflow": self.capacity_overflow,
            "cash_conversion_cycle": self.cash_conversion_cycle,
            "safety_potential": self.safety_potential,
        }


class RewardModel:
    """Reward function aligned to Phase 6 service, cost, resilience, and risk terms."""

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self.weights = weights or load_reward_weights()

    def compute(self, components: RewardComponents) -> float:
        values = components.as_dict()
        score = 0.0
        for name, value in values.items():
            score += getattr(self.weights, name) * value
        return float(score)

    def from_snapshots(
        self,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
        *,
        projected: bool = False,
        blocked: bool = False,
    ) -> tuple[float, RewardComponents]:
        delivered_delta = _delivered_count(current) - _delivered_count(previous)
        late_delta = _late_count(current) - _late_count(previous)
        cost_delta = max(0.0, float(current.get("total_transport_cost", 0.0)) - float(previous.get("total_transport_cost", 0.0)))
        service = clamp(float(current.get("service_level", 0.0)), 0.0, 1.0)
        sp = clamp(float(current.get("network_safety_potential", 0.0)), 0.0, 1.0)
        disruption = clamp(float(current.get("disruption_score", 0.0)), 0.0, 1.0)

        inventory = current.get("inventory", {})
        backlog_pressure = _inventory_average(inventory, "backlog_pressure")
        inventory_pressure = _inventory_average(inventory, "inventory_pressure")
        stockout_pressure = _stockout_pressure(inventory)

        components = RewardComponents(
            service_level=service,
            throughput=max(0.0, delivered_delta),
            resilience=1.0 - disruption,
            holding_cost=inventory_pressure,
            shipping_cost=min(cost_delta / 100_000.0, 1.0),
            late_delivery=max(0.0, late_delta),
            backlog=backlog_pressure,
            stockout=stockout_pressure,
            capacity_overflow=1.0 if blocked else 0.0,
            cash_conversion_cycle=_cash_cycle_proxy(inventory),
            safety_potential=max(sp, predictive_safety_potential(current)),
        )
        reward = self.compute(components)
        if projected:
            reward -= 0.03
        if blocked:
            reward -= 0.15
        return reward, components


def load_reward_weights(path: str | Path = DEFAULT_REWARD_CONFIG) -> RewardWeights:
    with Path(path).open("r", encoding="utf-8") as handle:
        return RewardWeights.from_mapping(json.load(handle))


def predictive_safety_potential(snapshot: Mapping[str, Any]) -> float:
    inventory = snapshot.get("inventory", {})
    return safety_potential(
        congestion_score=_safe_float(snapshot.get("db_congestion_score", 0.0)),
        disruption_score=_safe_float(snapshot.get("disruption_score", 0.0)),
        capacity_pressure=_inventory_average(inventory, "inventory_pressure"),
        backlog_pressure=_inventory_average(inventory, "backlog_pressure"),
    )


def inventory_reward_terms(
    *,
    backlog_units: float,
    demand_units: float,
    used_capacity: float,
    total_capacity: float,
    congestion_score: float,
    disruption_score: float,
) -> dict[str, float]:
    backlog = backlog_ratio(backlog_units, max(demand_units, backlog_units, 1.0))
    capacity = capacity_utilization(used_capacity, max(total_capacity, 1.0))
    return {
        "backlog_pressure": backlog,
        "capacity_utilization": capacity,
        "safety_potential": safety_potential(
            congestion_score=congestion_score,
            disruption_score=disruption_score,
            capacity_pressure=capacity,
            backlog_pressure=backlog,
        ),
    }


def _delivered_count(snapshot: Mapping[str, Any]) -> int:
    orders = snapshot.get("orders", {})
    if not isinstance(orders, Mapping):
        return 0
    return sum(1 for order in orders.values() if isinstance(order, Mapping) and order.get("status") == "delivered")


def _late_count(snapshot: Mapping[str, Any]) -> int:
    orders = snapshot.get("orders", {})
    if not isinstance(orders, Mapping):
        return 0
    return sum(1 for order in orders.values() if isinstance(order, Mapping) and bool(order.get("is_late")))


def _inventory_average(inventory: Any, key: str) -> float:
    if not isinstance(inventory, Mapping) or not inventory:
        return 0.0
    values = [float(row.get(key, 0.0)) for row in inventory.values() if isinstance(row, Mapping)]
    if not values:
        return 0.0
    return clamp(sum(values) / len(values), 0.0, 1.0)


def _stockout_pressure(inventory: Any) -> float:
    if not isinstance(inventory, Mapping) or not inventory:
        return 0.0
    rows = [row for row in inventory.values() if isinstance(row, Mapping)]
    if not rows:
        return 0.0
    stockout_count = sum(1 for row in rows if float(row.get("on_hand_units", 0.0)) <= 0.0 and float(row.get("demand_mean", 0.0)) > 0.0)
    return clamp(stockout_count / len(rows), 0.0, 1.0)


def _cash_cycle_proxy(inventory: Any) -> float:
    if not isinstance(inventory, Mapping) or not inventory:
        return 0.0
    on_hand = 0.0
    in_transit = 0.0
    backlog = 0.0
    for row in inventory.values():
        if isinstance(row, Mapping):
            on_hand += float(row.get("on_hand_units", 0.0))
            in_transit += float(row.get("in_transit_units", 0.0))
            backlog += float(row.get("backlog_units", 0.0))
    denominator = max(on_hand + in_transit + backlog, 1.0)
    return clamp((on_hand + in_transit) / denominator, 0.0, 1.0)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

