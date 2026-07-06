"""PyTorch policy extensions for Stable-Baselines3 logistics agents."""

from __future__ import annotations

from typing import Any

import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class LogisticsFeatureExtractor(BaseFeaturesExtractor):
    """Compact MLP extractor tuned for dense normalized logistics observations."""

    def __init__(self, observation_space: Any, features_dim: int = 128) -> None:
        super().__init__(observation_space, features_dim)
        input_dim = int(observation_space.shape[0])
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim),
            nn.SiLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.network(observations.float())


class LogisticsActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy with stable initialization for PPO logistics control."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault(
            "features_extractor_class",
            LogisticsFeatureExtractor,
        )
        kwargs.setdefault(
            "features_extractor_kwargs",
            {"features_dim": 128},
        )
        kwargs.setdefault(
            "net_arch",
            {"pi": [128, 128], "vf": [128, 128]},
        )
        kwargs.setdefault("activation_fn", nn.SiLU)
        super().__init__(*args, **kwargs)
        self.apply(self._orthogonal_init)

    @staticmethod
    def _orthogonal_init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def ppo_policy_kwargs() -> dict[str, Any]:
    return {
        "features_extractor_class": LogisticsFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        "activation_fn": nn.SiLU,
    }


def dqn_policy_kwargs() -> dict[str, Any]:
    return {
        "features_extractor_class": LogisticsFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": [256, 128],
        "activation_fn": nn.SiLU,
    }

