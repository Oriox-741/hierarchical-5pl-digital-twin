"""Project normalized PPO actions into physical logistics control parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from src.shared.constants import MAX_NORMALIZED_ACTION, MIN_NORMALIZED_ACTION
from src.shared.metrics import clamp


CONTINUOUS_ACTION_DIM: Final[int] = 5


@dataclass(frozen=True, slots=True)
class PhysicalAction:
    """Physical controls derived from a normalized continuous PPO action."""

    reorder_fraction: float
    dispatch_intensity: float
    speed_multiplier: float
    safety_stock_multiplier: float
    capacity_buffer_fraction: float

    def as_dict(self) -> dict[str, float]:
        return {
            "reorder_fraction": self.reorder_fraction,
            "dispatch_intensity": self.dispatch_intensity,
            "speed_multiplier": self.speed_multiplier,
            "safety_stock_multiplier": self.safety_stock_multiplier,
            "capacity_buffer_fraction": self.capacity_buffer_fraction,
        }

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.reorder_fraction,
                self.dispatch_intensity,
                self.speed_multiplier,
                self.safety_stock_multiplier,
                self.capacity_buffer_fraction,
            ],
            dtype=np.float32,
        )


class ActionProjector:
    """Map continuous PPO `[-1, 1]` controls into bounded logistics values."""

    def __init__(
        self,
        *,
        max_speed_delta_fraction: float = 0.35,
        max_safety_stock_multiplier: float = 2.0,
        max_capacity_buffer_fraction: float = 0.50,
    ) -> None:
        if max_speed_delta_fraction < 0.0:
            raise ValueError("max_speed_delta_fraction must not be negative.")
        if max_safety_stock_multiplier < 1.0:
            raise ValueError("max_safety_stock_multiplier must be >= 1.")
        if max_capacity_buffer_fraction < 0.0:
            raise ValueError("max_capacity_buffer_fraction must not be negative.")

        self.max_speed_delta_fraction = max_speed_delta_fraction
        self.max_safety_stock_multiplier = max_safety_stock_multiplier
        self.max_capacity_buffer_fraction = max_capacity_buffer_fraction

    def project(self, action: np.ndarray) -> PhysicalAction:
        vector = np.asarray(action, dtype=np.float32).reshape(-1)
        if vector.shape[0] != CONTINUOUS_ACTION_DIM:
            raise ValueError(f"expected action dimension {CONTINUOUS_ACTION_DIM}, got {vector.shape[0]}.")

        clipped = np.clip(vector, MIN_NORMALIZED_ACTION, MAX_NORMALIZED_ACTION)
        reorder_fraction = _unit_interval(clipped[0])
        dispatch_intensity = _unit_interval(clipped[1])
        speed_multiplier = 1.0 + (float(clipped[2]) * self.max_speed_delta_fraction)
        safety_stock_multiplier = 1.0 + (_unit_interval(clipped[3]) * (self.max_safety_stock_multiplier - 1.0))
        capacity_buffer_fraction = _unit_interval(clipped[4]) * self.max_capacity_buffer_fraction

        return PhysicalAction(
            reorder_fraction=clamp(reorder_fraction, 0.0, 1.0),
            dispatch_intensity=clamp(dispatch_intensity, 0.0, 1.0),
            speed_multiplier=max(0.1, speed_multiplier),
            safety_stock_multiplier=max(1.0, safety_stock_multiplier),
            capacity_buffer_fraction=clamp(capacity_buffer_fraction, 0.0, self.max_capacity_buffer_fraction),
        )


def _unit_interval(value: float | np.float32) -> float:
    return (float(value) + 1.0) * 0.5
