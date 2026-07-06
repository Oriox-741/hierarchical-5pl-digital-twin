"""Build training and evaluation datasets from persisted episode traces."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.learn.episode_store import EpisodeTransition


@dataclass(frozen=True, slots=True)
class ReplayBatchArrays:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


class EpisodeReplayDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Torch dataset of state-action-reward transitions."""

    def __init__(self, transitions: Iterable[EpisodeTransition]) -> None:
        self.arrays = build_replay_arrays(transitions)

    def __len__(self) -> int:
        return int(self.arrays.rewards.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.arrays.observations[index]).float(),
            torch.from_numpy(self.arrays.actions[index]).float(),
            torch.tensor(self.arrays.rewards[index], dtype=torch.float32),
            torch.from_numpy(self.arrays.next_observations[index]).float(),
            torch.tensor(self.arrays.dones[index], dtype=torch.float32),
        )


def build_replay_arrays(transitions: Iterable[EpisodeTransition]) -> ReplayBatchArrays:
    rows = list(transitions)
    if not rows:
        empty = np.empty((0, 0), dtype=np.float32)
        return ReplayBatchArrays(
            observations=empty,
            actions=empty,
            rewards=np.empty((0,), dtype=np.float32),
            next_observations=empty,
            dones=np.empty((0,), dtype=np.float32),
        )

    observations = np.vstack([_vector(row.observation, key="observation") for row in rows]).astype(np.float32)
    actions = np.vstack([_vector(row.action, key="action") for row in rows]).astype(np.float32)
    next_observations = np.vstack([
        _vector(row.next_observation or row.observation, key="next_observation") for row in rows
    ]).astype(np.float32)
    rewards = np.array([row.reward for row in rows], dtype=np.float32)
    dones = np.array([row.terminated or row.truncated for row in rows], dtype=np.float32)
    return ReplayBatchArrays(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        dones=dones,
    )


def split_dataset(
    transitions: list[EpisodeTransition],
    *,
    validation_fraction: float = 0.20,
) -> tuple[EpisodeReplayDataset, EpisodeReplayDataset]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    split_at = int(len(transitions) * (1.0 - validation_fraction))
    return EpisodeReplayDataset(transitions[:split_at]), EpisodeReplayDataset(transitions[split_at:])


def _vector(mapping: Any, *, key: str) -> np.ndarray:
    if isinstance(mapping, dict):
        if "vector" in mapping:
            return np.asarray(mapping["vector"], dtype=np.float32).reshape(1, -1)
        if key in mapping:
            return np.asarray(mapping[key], dtype=np.float32).reshape(1, -1)
        numeric = [value for value in mapping.values() if isinstance(value, int | float | bool)]
        if numeric:
            return np.asarray(numeric, dtype=np.float32).reshape(1, -1)
    raise ValueError(f"Could not convert {key} mapping to a numeric vector.")

