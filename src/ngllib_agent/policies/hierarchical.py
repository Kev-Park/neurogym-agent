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

Spatial variant (2026-10-01, reviewed design): when the obs carries
`patch_features: Box(P, G, G, 384)` — the raw DINO patch-token grid per pane
(fp16, no CLS; see `obs/dino_encoder.py`) — the module is a CNN on top of the
frozen token grid (AlphaStar-shaped: conv trunk, non-spatial heads from one
vector, the location head as a map conditioned on that vector):
  encoder (per pane, shared weights + a learned pane embedding):
      1x1 conv 384->C, then `spatial_blocks` residual blocks
      [GN, ReLU, 3x3, GN, ReLU, 3x3] at C channels  ->  map (C, G, G)
  shared vector g:
      1x1 conv C->`flat_channels`, flatten over (P, G, G) (position-preserving),
      Linear -> `hidden`, ReLU; concat pos-MLP; Linear -> `hidden`, ReLU
  heads from g only: verb, rotate x3, zoom, value (linear)
  click head: per-location logit = 1x1 conv over [map ; broadcast Linear(g->C)],
      one logit per token -> the click grid MUST be G x (P*G) (panes side by
      side, EM then 3D, row-major), no interpolation.
Trunk+heads run under bf16 autocast on CUDA. The CLS-only path keeps its
parameter names so old checkpoints still load.
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
        trunk_hiddens: list[int] = [256, 256]   # CLS-only path
        normalize_entropy: bool = False
        cell_entropy_scale: float = 1.0
      spatial (only read when obs has patch_features; see module docstring):
        spatial_channels: int = 128   # C, conv width over the token grid
        spatial_blocks: int = 3       # residual 3x3 blocks per pane
        flat_channels: int = 16       # 1x1 reduction before the flatten
        hidden: int = 512             # width of the shared vector g
        autocast_bf16: bool = True    # bf16 autocast for trunk+heads on CUDA
    """

    @override(TorchRLModule)
    def setup(self):
        pos_dim = int(self.observation_space["pos_state"].shape[0])
        nvec = [int(n) for n in self.action_space.nvec]
        spaces = getattr(self.observation_space, "spaces", {})
        patch_space = spaces.get("patch_features")
        self._spatial = patch_space is not None
        self._autocast = bool(self.model_config.get("autocast_bf16", True))

        pos_hidden = int(self.model_config.get("pos_hidden_dim", 64))

        # Legacy DinoFeaturesExtractor: identity on image features, MLP on pos.
        self._pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, pos_hidden),
            nn.LayerNorm(pos_hidden),
            nn.GELU(),
            nn.Linear(pos_hidden, pos_hidden),
            nn.GELU(),
        )

        if self._spatial:
            self._setup_spatial(patch_space, nvec, pos_hidden)
        else:
            img_dim = int(spaces["image_features"].shape[0])
            trunk_hiddens = list(self.model_config.get("trunk_hiddens", [256, 256]))
            layers: list[nn.Module] = []
            in_dim = img_dim + pos_hidden
            for h in trunk_hiddens:
                layers += [nn.Linear(in_dim, h), nn.ReLU()]
                in_dim = h
            self._trunk = nn.Sequential(*layers)
            self._pi_head = nn.Linear(in_dim, int(np.sum(nvec)))
            self._g_dim = in_dim

        if not self.inference_only:
            self._vf_head = nn.Linear(self._g_dim, 1)

        self.action_dist_cls = HierarchicalMultiCategorical.for_nvec(
            nvec,
            normalize_entropy=bool(self.model_config.get("normalize_entropy", False)),
            cell_entropy_scale=float(self.model_config.get("cell_entropy_scale", 1.0)),
        )

    def _setup_spatial(self, patch_space, nvec: List[int], pos_hidden: int) -> None:
        n_panes, g_rows, g_cols, c_in = (int(s) for s in patch_space.shape)
        c = int(self.model_config.get("spatial_channels", 128))
        n_blocks = int(self.model_config.get("spatial_blocks", 3))
        flat_c = int(self.model_config.get("flat_channels", 16))
        hidden = int(self.model_config.get("hidden", 512))
        n_verb, n_cell, r, _, _, n_zoom = nvec
        # One logit per token: the click grid is the token grid with the panes
        # side by side (EM cols 0..G-1, 3D cols G..2G-1). The ActionSpec's
        # cell_to_pixel maps row-major cells over click_bounds, which spans
        # both panes in that order, so no interpolation or re-indexing.
        if n_cell != g_rows * n_panes * g_cols:
            raise ValueError(
                f"spatial policy needs a {g_rows}x{n_panes * g_cols} click grid "
                f"(one cell per token); action space has {n_cell} cells")
        self._grid = (n_panes, g_rows, g_cols)

        self._in_proj = nn.Conv2d(c_in, c, 1)
        self._pane_embed = nn.Parameter(torch.zeros(n_panes, c, 1, 1))
        self._blocks = nn.ModuleList(_ResBlock(c) for _ in range(n_blocks))
        self._flat_proj = nn.Conv2d(c, flat_c, 1)
        self._g1 = nn.Linear(flat_c * n_panes * g_rows * g_cols, hidden)
        self._g2 = nn.Linear(hidden + pos_hidden, hidden)
        self._g_dim = hidden

        self._pi_rest = nn.Linear(hidden, n_verb + 3 * r + n_zoom)
        self._g_to_map = nn.Linear(hidden, c)
        self._cell_score = nn.Conv2d(2 * c, 1, 1)

    def _features(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """(shared vector g, per-pane feature map (B*P, C, G, G) or None)."""
        obs = batch[Columns.OBS]
        pos = self._pos_mlp(obs["pos_state"].float())
        if not self._spatial:
            return self._trunk(torch.cat([obs["image_features"], pos], dim=-1)), None
        n_panes, g_rows, g_cols = self._grid
        x = obs["patch_features"]                                  # (B, P, G, G, D)
        b = x.shape[0]
        x = x.reshape(b * n_panes, g_rows, g_cols, -1).permute(0, 3, 1, 2)
        x = self._in_proj(x.to(self._in_proj.weight.dtype))        # (B*P, C, G, G)
        x = x + self._pane_embed.repeat(b, 1, 1, 1)
        for blk in self._blocks:
            x = blk(x)
        fmap = x
        flat = F.relu(self._flat_proj(fmap)).reshape(b, -1)        # (B, flat_c*P*G*G)
        g = F.relu(self._g1(flat))
        g = F.relu(self._g2(torch.cat([g, pos], dim=-1)))
        return g, fmap

    def _logits(self, g: torch.Tensor, fmap: Optional[torch.Tensor]) -> torch.Tensor:
        if not self._spatial:
            return self._pi_head(g)
        n_panes, g_rows, g_cols = self._grid
        n_verb = self.action_dist_cls._input_lens[0]
        b = g.shape[0]
        rest = self._pi_rest(g)
        ctx = self._g_to_map(g)                                    # (B, C)
        ctx = ctx[:, None, :, None, None].expand(b, n_panes, -1, g_rows, g_cols)
        ctx = ctx.reshape(b * n_panes, -1, g_rows, g_cols)
        score = self._cell_score(torch.cat([fmap, ctx], dim=1))    # (B*P, 1, G, G)
        # (B, P, G, G) -> (B, G, P*G): row-major over [row][pane][col]
        cell = score.reshape(b, n_panes, g_rows, g_cols).permute(0, 2, 1, 3).reshape(b, -1)
        return torch.cat([rest[:, :n_verb], cell, rest[:, n_verb:]], dim=-1)

    def _autocast_ctx(self, batch: Dict[str, Any]):
        dev = batch[Columns.OBS]["pos_state"].device
        if self._spatial and self._autocast and dev.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cpu", enabled=False)

    def _embed(self, batch: Dict[str, Any]) -> torch.Tensor:
        with self._autocast_ctx(batch):
            return self._features(batch)[0].float()

    @override(TorchRLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        with self._autocast_ctx(batch):
            logits = self._logits(*self._features(batch))
        return {Columns.ACTION_DIST_INPUTS: logits.float()}

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        with self._autocast_ctx(batch):
            g, fmap = self._features(batch)
            logits = self._logits(g, fmap)
        return {
            Columns.ACTION_DIST_INPUTS: logits.float(),
            Columns.EMBEDDINGS: g.float(),
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> torch.Tensor:
        if embeddings is None:
            embeddings = self._embed(batch)
        return self._vf_head(embeddings.float()).squeeze(-1)


class _ResBlock(nn.Module):
    """Pre-activation residual block: [GN, ReLU, 3x3, GN, ReLU, 3x3] + skip."""

    def __init__(self, c: int, groups: int = 8):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(groups, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.relu(self.n1(x)))
        h = self.c2(F.relu(self.n2(h)))
        return x + h
