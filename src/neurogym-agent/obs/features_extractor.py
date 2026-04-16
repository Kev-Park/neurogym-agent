from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DinoFeaturesExtractor(BaseFeaturesExtractor):
    """Consumes `{image_features: Box(dino_dim,), pos_state: Box(pos_dim,)}` from
    the env and produces a flat feature vector for SB3's policy MLP head.

    Image features are left as-is (already encoded upstream by a frozen DINO ViT
    inside each environment worker). `pos_state` is passed through a small MLP
    so it isn't drowned out by the much-higher-dimensional image features.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        pos_hidden_dim: int = 64,
    ):
        img_dim = int(observation_space["image_features"].shape[0])
        pos_dim = int(observation_space["pos_state"].shape[0])
        output_dim = img_dim + pos_hidden_dim
        super().__init__(observation_space, features_dim=output_dim)

        self.image_projection = nn.Identity()
        self.pos_projection = nn.Sequential(
            nn.Linear(pos_dim, pos_hidden_dim),
            nn.LayerNorm(pos_hidden_dim),
            nn.GELU(),
            nn.Linear(pos_hidden_dim, pos_hidden_dim),
            nn.GELU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        image_features = self.image_projection(observations["image_features"])
        pos_features = self.pos_projection(observations["pos_state"])
        return torch.cat([image_features, pos_features], dim=-1)
