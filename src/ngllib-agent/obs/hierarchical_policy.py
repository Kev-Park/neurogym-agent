from __future__ import annotations

import torch
from torch.distributions import Categorical
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


class HierarchicalDistribution:
    """Two-level action distribution for the click+rotate task.

    Level 1  action_type  ∈ {0=click, 1=rotate}
    Level 2a cell         ∈ {0..1023}   — sampled only when action_type==0
    Level 2b (d_ex, d_ey, d_ez) ∈ {0..8} each — sampled only when action_type==1

    log_prob = log P(action_type)
             + I(click)  * log P(cell)            [action_type==0]
             + I(rotate) * [log P(d_ex) + log P(d_ey) + log P(d_ez)]  [action_type==1]

    entropy  = H(action_type)
             + p(click)  * H(cell)
             + p(rotate) * [H(d_ex) + H(d_ey) + H(d_ez)]

    Cell and rotation heads are always evaluated (forward pass runs in full),
    but gradients flow only through the sub-head that matches the sampled
    action_type, so each head is trained exclusively on the steps where it
    is actually used.
    """

    def __init__(
        self,
        type_logits: torch.Tensor,  # (B, 3)
        cell_logits: torch.Tensor,  # (B, 1024)
        ex_logits: torch.Tensor,    # (B, 9)
        ey_logits: torch.Tensor,    # (B, 9)
        ez_logits: torch.Tensor,    # (B, 9)
    ):
        self.type_dist = Categorical(logits=type_logits)
        self.cell_dist = Categorical(logits=cell_logits)
        self.ex_dist   = Categorical(logits=ex_logits)
        self.ey_dist   = Categorical(logits=ey_logits)
        self.ez_dist   = Categorical(logits=ez_logits)

    def sample(self) -> torch.Tensor:
        return torch.stack([
            self.type_dist.sample(),
            self.cell_dist.sample(),
            self.ex_dist.sample(),
            self.ey_dist.sample(),
            self.ez_dist.sample(),
        ], dim=-1)

    def mode(self) -> torch.Tensor:
        return torch.stack([
            self.type_dist.probs.argmax(dim=-1),
            self.cell_dist.probs.argmax(dim=-1),
            self.ex_dist.probs.argmax(dim=-1),
            self.ey_dist.probs.argmax(dim=-1),
            self.ez_dist.probs.argmax(dim=-1),
        ], dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        typ  = actions[..., 0].long()
        cell = actions[..., 1].long()
        d_ex = actions[..., 2].long()
        d_ey = actions[..., 3].long()
        d_ez = actions[..., 4].long()

        is_click  = (typ == 0).float()
        is_rotate = (typ == 1).float()

        return (
            self.type_dist.log_prob(typ)
            + is_click  * self.cell_dist.log_prob(cell)
            + is_rotate * (
                self.ex_dist.log_prob(d_ex)
                + self.ey_dist.log_prob(d_ey)
                + self.ez_dist.log_prob(d_ez)
            )
        )

    def entropy(self) -> torch.Tensor:
        p_click  = self.type_dist.probs[..., 0]
        p_rotate = self.type_dist.probs[..., 1]
        return (
            self.type_dist.entropy()
            + p_click  * self.cell_dist.entropy()
            + p_rotate * (
                self.ex_dist.entropy()
                + self.ey_dist.entropy()
                + self.ez_dist.entropy()
            )
        )


# Logit slice boundaries — must match ActionSpec.multidiscrete_nvec() order:
# [2, num_cells, rot_bins, rot_bins, rot_bins]
_TYPE_END = 2
_CELL_END = _TYPE_END + 1024   # 1026
_EX_END   = _CELL_END + 9     # 1035
_EY_END   = _EX_END   + 9     # 1044
_EZ_END   = _EY_END   + 9     # 1053


class HierarchicalPolicy(MultiInputActorCriticPolicy):
    """Drop-in SB3 policy for MultiDiscrete([3, 1024, 9, 9, 9]).

    SB3 builds action_net as Linear(latent_dim_pi, 1054) — the sum of the nvec.
    We reinterpret those 1054 logits as five independent heads and route
    gradients through only the relevant sub-head per step.

    Everything else (feature extractor, MLP trunk, value head, optimizer,
    checkpointing) is inherited unchanged from MultiInputActorCriticPolicy.
    """

    def _get_dist(self, latent_pi: torch.Tensor) -> HierarchicalDistribution:
        logits = self.action_net(latent_pi)
        return HierarchicalDistribution(
            type_logits=logits[:, :_TYPE_END],
            cell_logits=logits[:, _TYPE_END:_CELL_END],
            ex_logits=logits[:, _CELL_END:_EX_END],
            ey_logits=logits[:, _EX_END:_EY_END],
            ez_logits=logits[:, _EY_END:_EZ_END],
        )

    def forward(
        self,
        obs: dict,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        dist = self._get_dist(latent_pi)
        actions = dist.mode() if deterministic else dist.sample()
        return actions, values, dist.log_prob(actions)

    def evaluate_actions(
        self,
        obs: dict,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        dist = self._get_dist(latent_pi)
        return values, dist.log_prob(actions), dist.entropy()

    def _predict(
        self,
        observation: dict,
        deterministic: bool = False,
    ) -> torch.Tensor:
        features = self.extract_features(observation, self.features_extractor)
        latent_pi, _ = self.mlp_extractor(features)
        dist = self._get_dist(latent_pi)
        return dist.mode() if deterministic else dist.sample()
