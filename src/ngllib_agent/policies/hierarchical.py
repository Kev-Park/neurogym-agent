"""Hierarchical policy for the Z-navigate task — RLlib new API stack.

Ports the legacy SB3 `HierarchicalPolicy`/`HierarchicalDistribution` (which had
click+rotate) to a custom `RLModule`, adding the zoom verb per agent_plan.md §10.

Action space: `MultiDiscrete([3, num_cells, R, R, R, Z])` —
`[action_type, click_cell, rot_x, rot_y, rot_z, zoom]`, verbs mutually exclusive.

The distribution gates log-prob/entropy/KL through the sub-head selected by
`action_type`, so each parameter head trains only on the steps where its verb
actually fired:

    logp    = logP(type) + I(click)*logP(cell)
                         + I(rotate)*(logP(rx)+logP(ry)+logP(rz))
                         + I(zoom)*logP(zoom)
    entropy = H(type) + p(click)*H(cell) + p(rotate)*ΣH(r*) + p(zoom)*H(zoom)
    kl      = analogous, weighted by self's p(type)

Observation: `Dict(image_features: Box(D,), pos_state: Box(8,))` — DINO features
are computed env-side (Round 8), so this module is a small MLP: the legacy
`DinoFeaturesExtractor` pos-MLP is absorbed into `setup()`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ray.rllib.core.columns import Columns
from ray.rllib.core.distribution.torch.torch_distribution import (
    TorchMultiCategorical,
)
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override


class HierarchicalMultiCategorical(TorchMultiCategorical):
    """TorchMultiCategorical with verb-gated logp/entropy/kl.

    Component order must be `[type(3), cell, rot_x, rot_y, rot_z, zoom]`.
    Use `for_nvec(nvec)` to bind the logit split sizes (RLlib instantiates
    distribution classes via `from_logits(logits)` with no extra args).
    """

    _input_lens: List[int] = []

    @classmethod
    def for_nvec(cls, nvec) -> type:
        lens = [int(n) for n in nvec]
        if len(lens) != 6 or lens[0] != 3:
            raise ValueError(f"expected nvec [3, cells, R, R, R, Z]; got {lens}")

        class _Bound(cls):
            _input_lens = lens

        _Bound.__name__ = f"{cls.__name__}_{'_'.join(map(str, lens))}"
        return _Bound

    @classmethod
    def from_logits(cls, logits: torch.Tensor, **kwargs) -> "HierarchicalMultiCategorical":
        if not cls._input_lens:
            raise ValueError("use for_nvec(nvec) to bind input_lens before from_logits")
        return super().from_logits(logits, input_lens=cls._input_lens)

    # -- gated overrides ------------------------------------------------------

    def _type_probs(self) -> torch.Tensor:
        return torch.softmax(self._cats[0].logits, dim=-1)

    @override(TorchMultiCategorical)
    def logp(self, value: torch.Tensor) -> torch.Tensor:
        parts = torch.unbind(value, dim=-1)
        typ = parts[0].long()
        lp = [cat.logp(act) for cat, act in zip(self._cats, parts)]
        is_click = (typ == 0).float()
        is_rotate = (typ == 1).float()
        is_zoom = (typ == 2).float()
        return (
            lp[0]
            + is_click * lp[1]
            + is_rotate * (lp[2] + lp[3] + lp[4])
            + is_zoom * lp[5]
        )

    @override(TorchMultiCategorical)
    def entropy(self) -> torch.Tensor:
        h = [cat.entropy() for cat in self._cats]
        p = self._type_probs()
        return (
            h[0]
            + p[..., 0] * h[1]
            + p[..., 1] * (h[2] + h[3] + h[4])
            + p[..., 2] * h[5]
        )

    @override(TorchMultiCategorical)
    def kl(self, other: "HierarchicalMultiCategorical") -> torch.Tensor:
        kls = [cat.kl(oth) for cat, oth in zip(self._cats, other._cats)]
        p = self._type_probs()
        return (
            kls[0]
            + p[..., 0] * kls[1]
            + p[..., 1] * (kls[2] + kls[3] + kls[4])
            + p[..., 2] * kls[5]
        )


class HierarchicalPPOModule(TorchRLModule, ValueFunctionAPI):
    """PPO RLModule: pos-MLP + trunk shared by the gated pi head and a vf head.

    model_config keys (all optional):
        pos_hidden_dim: int = 64      # legacy DinoFeaturesExtractor default
        trunk_hiddens: list[int] = [256, 256]
    """

    @override(TorchRLModule)
    def setup(self):
        img_dim = int(self.observation_space["image_features"].shape[0])
        pos_dim = int(self.observation_space["pos_state"].shape[0])
        nvec = [int(n) for n in self.action_space.nvec]

        pos_hidden = int(self.model_config.get("pos_hidden_dim", 64))
        trunk_hiddens = list(self.model_config.get("trunk_hiddens", [256, 256]))

        # Legacy DinoFeaturesExtractor: identity on image features, MLP on pos.
        self._pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, pos_hidden),
            nn.LayerNorm(pos_hidden),
            nn.GELU(),
            nn.Linear(pos_hidden, pos_hidden),
            nn.GELU(),
        )

        layers: list[nn.Module] = []
        in_dim = img_dim + pos_hidden
        for h in trunk_hiddens:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self._trunk = nn.Sequential(*layers)

        self._pi_head = nn.Linear(in_dim, int(np.sum(nvec)))
        if not self.inference_only:
            self._vf_head = nn.Linear(in_dim, 1)

        self.action_dist_cls = HierarchicalMultiCategorical.for_nvec(nvec)

    def _embed(self, batch: Dict[str, Any]) -> torch.Tensor:
        obs = batch[Columns.OBS]
        pos = self._pos_mlp(obs["pos_state"])
        return self._trunk(torch.cat([obs["image_features"], pos], dim=-1))

    @override(TorchRLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return {Columns.ACTION_DIST_INPUTS: self._pi_head(self._embed(batch))}

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        embeddings = self._embed(batch)
        return {
            Columns.ACTION_DIST_INPUTS: self._pi_head(embeddings),
            Columns.EMBEDDINGS: embeddings,
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> torch.Tensor:
        if embeddings is None:
            embeddings = self._embed(batch)
        return self._vf_head(embeddings).squeeze(-1)
