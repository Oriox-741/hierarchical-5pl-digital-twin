"""Small deterministic statistics helpers for benchmark reports."""

from __future__ import annotations

import math
import random
from statistics import mean, stdev
from typing import Any, Sequence


def finite_values(values: Sequence[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            result.append(numeric)
    return result


def summary_stats(values: Sequence[Any]) -> dict[str, float | int]:
    sample = finite_values(values)
    if not sample:
        raise ValueError("summary_stats requires at least one finite value")
    ordered = sorted(sample)
    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "sd": stdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "q25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.5),
        "q75": _quantile(ordered, 0.75),
        "max": ordered[-1],
    }


def bootstrap_mean_ci(
    values: Sequence[Any],
    *,
    iterations: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    sample = finite_values(values)
    if not sample:
        raise ValueError("bootstrap_mean_ci requires at least one finite value")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    rng = random.Random(seed)
    size = len(sample)
    boot_means = []
    for _ in range(iterations):
        boot_means.append(mean(sample[rng.randrange(size)] for _ in range(size)))
    boot_means.sort()
    lower_index = max(0, min(iterations - 1, int((alpha / 2.0) * iterations)))
    upper_index = max(0, min(iterations - 1, int((1.0 - alpha / 2.0) * iterations) - 1))
    return {
        "sample_size": size,
        "mean": mean(sample),
        "lower": boot_means[lower_index],
        "upper": boot_means[upper_index],
        "iterations": int(iterations),
        "alpha": float(alpha),
    }


def exact_sign_test_two_sided(
    deltas: Sequence[Any],
    *,
    zero_epsilon: float = 1e-12,
) -> dict[str, float | int]:
    positive = 0
    negative = 0
    ties = 0
    for value in finite_values(deltas):
        if value > zero_epsilon:
            positive += 1
        elif value < -zero_epsilon:
            negative += 1
        else:
            ties += 1

    non_ties = positive + negative
    if non_ties == 0:
        p_value = 1.0
    else:
        observed = min(positive, negative)
        tail_probability = sum(math.comb(non_ties, k) for k in range(observed + 1)) / (2**non_ties)
        p_value = min(1.0, 2.0 * tail_probability)
    return {
        "positive": positive,
        "negative": negative,
        "ties": ties,
        "n": non_ties,
        "p_value": p_value,
    }


def _quantile(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        raise ValueError("ordered sample must not be empty")
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)
