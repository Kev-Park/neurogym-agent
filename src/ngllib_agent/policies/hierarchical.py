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

Spatial variant (2026-10-01): when the obs also carries
`patch_features: Box(P, G, G, C_in)` (projected DINO patch tokens, see
`obs/dino_encoder.py`), the panes are laid side by side (width = P*G, the same
layout as the click grid), run through two 3x3 convs, and
  - per-pane mean+max pools (+ CLS, + pos) feed the trunk, so verb / rotate /
    zoom / value are conditioned on WHERE things are, not just what is there;
  - the click-cell logits are a per-location score map (1x1 conv over the
    spatial map with the trunk embedding broadcast in) bilinearly resized to
    the click grid, plus a per-cell bias.
The CLS-only path keeps its parameter names so old checkpoints still load.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.core.columns import Columns
from ray.rllib.core.distribution.torch.torch_distribution import (
    TorchMultiCategorical,
)
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override


class HierarchicalMultiCategorical(TorchMultiCategorical):
    """TorchMultiCategorical with verb-gated logp/entropy/kl.

    Component order must be `[type(3|4), cell, rot_x, rot_y, rot_z, zoom]`.
    Use `for_nvec(nvec)` to bind the logit split sizes (RLlib instantiates
    distribution classes via `from_logits(logits)` with no extra args).
    """

    _input_lens: List[int] = []
    _normalize_entropy: bool = False
    _cell_entropy_scale: float = 1.0

    @classmethod
    def for_nvec(cls, nvec, normalize_entropy: bool = False,
                 cell_entropy_scale: float = 1.0) -> type:
        lens = [int(n) for n in nvec]
        # 3 verbs (right_click / rotate / zoom), 4 (+ double_click, 2026-09-10)
        # or 5 (+ xs_zoom, the 2D pane's zoom, 2026-09-24). The verb count sizes
        # the head, so it comes from the ActionSpec the config declares -- a
        # 3-verb checkpoint loads only into a 3-verb module.
        if len(lens) != 6 or lens[0] not in (3, 4, 5):
            raise ValueError(f"expected nvec [3|4|5, cells, R, R, R, Z]; got {lens}")

        class _Bound(cls):
            _input_lens = lens
            _normalize_entropy = normalize_entropy
            # Multiplier on the click-cell branch's entropy bonus only. The
            # bonus pays for a FLAT cell head; with a spatial input the head
            # has a real gradient to sharpen on, so <1 lets it aim sooner.
            _cell_entropy_scale = float(cell_entropy_scale)

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

    def _cell_weight(self, p: torch.Tensor) -> torch.Tensor:
        """Probability mass on the verbs that use the cell head."""
        return p[..., 0] + (p[..., 3] if p.shape[-1] >= 4 else 0.0)

    def _zoom_weight(self, p: torch.Tensor) -> torch.Tensor:
        """Probability mass on the verbs that spend the zoom head.

        Verb 2 zooms the 3D camera and verb 4 the 2D pane; they share one bin
        head (the bin means "how much", the verb means "which zoom"), so both
        have to credit it or xs_zoom would train an ungated head.
        """
        return p[..., 2] + (p[..., 4] if p.shape[-1] >= 5 else 0.0)

    @override(TorchMultiCategorical)
    def logp(self, value: torch.Tensor) -> torch.Tensor:
        parts = torch.unbind(value, dim=-1)
        typ = parts[0].long()
        lp = [cat.logp(act) for cat, act in zip(self._cats, parts)]
        # Verbs 0 (right_click) and 3 (double_click) both spend the cell head;
        # verbs 2 (3D zoom) and 4 (2D zoom) both spend the zoom head.
        is_click = ((typ == 0) | (typ == 3)).float()
        is_rotate = (typ == 1).float()
        is_zoom = ((typ == 2) | (typ == 4)).float()
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
        if self._normalize_entropy:
            # Per-branch max-entropy normalization (2026-08-24): unnormalized,
            # the gated bonus p_click*H(1024-way) offers ~ln1024=6.9 nats vs
            # zoom's ln9=2.2, so the entropy regularizer itself pushes verb
            # mass toward click — v7 used zoom on 0.14% of steps. Normalized,
            # every branch offers the same [0,1] bonus and verb allocation is
            # entropy-neutral; exploration of zoom survives on equal terms.
            # Range is [0,2] not ~[0,8] — scale entropy_coeff up ~4x (config).
            import math

            n_verb, n_cell, r, _, _, n_zoom = self._input_lens
            return (
                h[0] / math.log(n_verb)
                + self._cell_entropy_scale * self._cell_weight(p) * h[1] / math.log(n_cell)
                + p[..., 1] * (h[2] + h[3] + h[4]) / (3.0 * math.log(r))
                + self._zoom_weight(p) * h[5] / math.log(n_zoom)
            )
        return (
            h[0]
            + self._cell_entropy_scale * self._cell_weight(p) * h[1]
            + p[..., 1] * (h[2] + h[3] + h[4])
            + self._zoom_weight(p) * h[5]
        )

    @override(TorchMultiCategorical)
    def kl(self, other: "HierarchicalMultiCategorical") -> torch.Tensor:
        kls = [cat.kl(oth) for cat, oth in zip(self._cats, other._cats)]
        p = self._type_probs()
        return (
            kls[0]
            + self._cell_weight(p) * kls[1]
            + p[..., 1] * (kls[2] + kls[3] + kls[4])
            + self._zoom_weight(p) * kls[5]
        )


class HierarchicalPPOModule(TorchRLModule, ValueFunctionAPI):
    """PPO RLModule: pos-MLP + trunk shared by the gated pi head and a vf head.

    model_config keys (all optional):
        pos_hidden_dim: int = 64      # legacy DinoFeaturesExtractor default
        trunk_hiddens: list[int] = [256, 256]
        normalize_entropy: bool = False
        cell_entropy_scale: float = 1.0
      spatial (only read when obs has patch_features):
        spatial_channels: int = 64    # conv width over the patch grid
        use_cls: bool = True          # also feed CLS to the trunk
        click_grid: [rows, cols]      # default: inferred from num_cells and
                                      # the pane count (rows = sqrt(cells/P))
    """

    @override(TorchRLModule)
    def setup(self):
        img_dim = int(self.observation_space["image_features"].shape[0])
        pos_dim = int(self.observation_space["pos_state"].shape[0])
        nvec = [int(n) for n in self.action_space.nvec]
        patch_space = getattr(self.observation_space, "spaces", {}).get("patch_features")
        self._spatial = patch_space is not None

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

        in_dim = pos_hidden
        if self._spatial:
            n_panes, g_rows, g_cols, c_in = (int(s) for s in patch_space.shape)
            c = int(self.model_config.get("spatial_channels", 64))
            self._use_cls = bool(self.model_config.get("use_cls", True))
            self._grid = (n_panes, g_rows, g_cols)
            self._conv = nn.Sequential(
                nn.Conv2d(c_in, c, 3, padding=1), nn.ReLU(),
                nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
            )
            in_dim += 2 * c * n_panes + (img_dim if self._use_cls else 0)
            self._click_grid = self._resolve_click_grid(nvec[1], n_panes)
        else:
            in_dim += img_dim

        layers: list[nn.Module] = []
        for h in trunk_hiddens:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self._trunk = nn.Sequential(*layers)

        if self._spatial:
            n_verb, n_cell, r, _, _, n_zoom = nvec
            self._pi_rest = nn.Linear(in_dim, n_verb + 3 * r + n_zoom)
            self._embed_to_map = nn.Linear(in_dim, c)
            self._cell_score = nn.Conv2d(2 * c, 1, 1)
            self._cell_bias = nn.Parameter(torch.zeros(n_cell))
        else:
            self._pi_head = nn.Linear(in_dim, int(np.sum(nvec)))
        if not self.inference_only:
            self._vf_head = nn.Linear(in_dim, 1)

        self.action_dist_cls = HierarchicalMultiCategorical.for_nvec(
            nvec,
            normalize_entropy=bool(self.model_config.get("normalize_entropy", False)),
            cell_entropy_scale=float(self.model_config.get("cell_entropy_scale", 1.0)),
        )

    def _resolve_click_grid(self, n_cell: int, n_panes: int) -> tuple[int, int]:
        """(rows, cols) of the click grid the score map is resized to. The
        panes sit side by side in the map, so the grid is expected to span the
        same panes with cols = n_panes * rows; anything else must be given
        explicitly as model_config.click_grid."""
        cg = self.model_config.get("click_grid")
        if cg is not None:
            rows, cols = int(cg[0]), int(cg[1])
        else:
            rows = int(round((n_cell / n_panes) ** 0.5))
            cols = n_panes * rows
        if rows * cols != n_cell:
            raise ValueError(
                f"click grid {rows}x{cols} != {n_cell} cells; set model.click_grid")
        return rows, cols

    def _features(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """(trunk embedding, spatial map or None)."""
        obs = batch[Columns.OBS]
        pos = self._pos_mlp(obs["pos_state"])
        if not self._spatial:
            return self._trunk(torch.cat([obs["image_features"], pos], dim=-1)), None
        n_panes, g_rows, g_cols = self._grid
        x = obs["patch_features"]                              # (B, P, R, C, D)
        b = x.shape[0]
        # panes side by side along width: (B, D, R, P*C), width = p*C + col
        x = x.permute(0, 4, 2, 1, 3).reshape(b, -1, g_rows, n_panes * g_cols)
        smap = self._conv(x)                                   # (B, c, R, P*C)
        per_pane = smap.reshape(b, smap.shape[1], g_rows, n_panes, g_cols)
        pooled = torch.cat(
            [per_pane.mean(dim=(2, 4)).flatten(1), per_pane.amax(dim=(2, 4)).flatten(1)],
            dim=-1)                                            # (B, 2*c*P)
        parts = [pooled, pos]
        if self._use_cls:
            parts.insert(1, obs["image_features"])
        return self._trunk(torch.cat(parts, dim=-1)), smap

    def _logits(self, embed: torch.Tensor, smap: Optional[torch.Tensor]) -> torch.Tensor:
        if not self._spatial:
            return self._pi_head(embed)
        n_verb = self.action_dist_cls._input_lens[0]
        rest = self._pi_rest(embed)
        ctx = self._embed_to_map(embed)[:, :, None, None].expand(-1, -1, *smap.shape[2:])
        score = self._cell_score(torch.cat([smap, ctx], dim=1))   # (B, 1, R, P*C)
        score = F.interpolate(score, size=self._click_grid, mode="bilinear",
                              align_corners=False)
        cell = score.flatten(1) + self._cell_bias
        return torch.cat([rest[:, :n_verb], cell, rest[:, n_verb:]], dim=-1)

    def _embed(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self._features(batch)[0]

    @override(TorchRLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return {Columns.ACTION_DIST_INPUTS: self._logits(*self._features(batch))}

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        embeddings, smap = self._features(batch)
        return {
            Columns.ACTION_DIST_INPUTS: self._logits(embeddings, smap),
            Columns.EMBEDDINGS: embeddings,
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> torch.Tensor:
        if embeddings is None:
            embeddings = self._embed(batch)
        return self._vf_head(embeddings).squeeze(-1)
