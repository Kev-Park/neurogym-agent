"""MultiDiscrete -> ngllib 0.2 Dict action translation (agent_plan.md §10).

Policy-facing action: `MultiDiscrete([3, num_cells, R, R, R, Z])` — three mutually
exclusive verbs (right_click / rotate / zoom). Decoded into ngllib's Dict action
space. Targets an **euler-orientation** `Environment` so the three rotate bins map
onto `delta_orient` (length 3); quaternion mode is rejected.

Extends the legacy `action_translator.py` (which had only click+rotate) with the
zoom verb via `delta_proj_scale`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ngllib Dict `action_type` codes (from Environment._build_action_space):
#   0=left_click, 1=right_click, 2=double_click, 3=edit_state
_NGL_RIGHT_CLICK = 1
_NGL_EDIT_STATE = 3


@dataclass(frozen=True)
class ActionSpec:
    grid_rows: int = 32
    grid_cols: int = 32
    pane_x0: float = 900.0
    pane_y0: float = 0.0
    pane_x1: float = 1800.0
    pane_y1: float = 900.0
    rotation_bins_per_axis: int = 9
    rotation_step_rad: float = 0.08
    zoom_bins: int = 9
    zoom_step: float = 500.0

    @property
    def num_cells(self) -> int:
        return self.grid_rows * self.grid_cols

    def nvec(self) -> list[int]:
        # [action_type, click_cell, rot_x, rot_y, rot_z, zoom]
        r = self.rotation_bins_per_axis
        return [3, self.num_cells, r, r, r, self.zoom_bins]


def _bin_to_signed(bin_index: int, bins_per_axis: int, step: float) -> float:
    """Center bin (bins//2) = 0; symmetric signed magnitude in `step` units."""
    return (bin_index - bins_per_axis // 2) * step


def cell_to_pixel(cell: int, spec: ActionSpec) -> tuple[float, float]:
    """Grid cell index -> pixel (x, y) at the cell center on the 3D pane."""
    row = cell // spec.grid_cols
    col = cell % spec.grid_cols
    cell_w = (spec.pane_x1 - spec.pane_x0) / spec.grid_cols
    cell_h = (spec.pane_y1 - spec.pane_y0) / spec.grid_rows
    x = spec.pane_x0 + (col + 0.5) * cell_w
    y = spec.pane_y0 + (row + 0.5) * cell_h
    return float(x), float(y)


def decode(md_action, spec: ActionSpec, orient_dim: int = 3) -> dict[str, Any]:
    """Translate a MultiDiscrete sample into an ngllib Dict action.

    Verbs are mutually exclusive; unused fields stay at their neutral zero.
    """
    a_type, cell, dx, dy, dz, dzoom = (int(v) for v in md_action)
    act: dict[str, Any] = {
        "action_type": 0,
        "mouse_xy": np.zeros(2, dtype=np.float32),
        "modifiers": np.zeros(3, dtype=np.int8),
        "delta_pos": np.zeros(3, dtype=np.float32),
        "delta_xs_scale": np.zeros(1, dtype=np.float32),
        "delta_orient": np.zeros(orient_dim, dtype=np.float32),
        "delta_proj_scale": np.zeros(1, dtype=np.float32),
    }

    if a_type == 0:  # right_click on a 3D-pane cell
        act["action_type"] = _NGL_RIGHT_CLICK
        x, y = cell_to_pixel(cell, spec)
        act["mouse_xy"] = np.array([x, y], dtype=np.float32)
    elif a_type == 1:  # rotate (euler deltas)
        act["action_type"] = _NGL_EDIT_STATE
        r, s = spec.rotation_bins_per_axis, spec.rotation_step_rad
        act["delta_orient"][:3] = (
            _bin_to_signed(dx, r, s),
            _bin_to_signed(dy, r, s),
            _bin_to_signed(dz, r, s),
        )
    elif a_type == 2:  # zoom (projection scale delta)
        act["action_type"] = _NGL_EDIT_STATE
        act["delta_proj_scale"][0] = _bin_to_signed(dzoom, spec.zoom_bins, spec.zoom_step)
    else:  # pragma: no cover - MultiDiscrete can't emit this
        raise ValueError(f"action_type out of range: {a_type}")

    return act


class MultiDiscreteActionWrapper:
    """gymnasium `ActionWrapper` exposing `MultiDiscrete` to the policy and
    decoding to ngllib's Dict action on `step`. Imported lazily to keep the
    decode logic free of a gymnasium dependency for unit tests."""

    def __new__(cls, env, spec: ActionSpec | None = None):
        import gymnasium as gym
        from gymnasium import spaces

        spec = spec or ActionSpec()
        orientation = getattr(env.unwrapped, "orientation", "euler")
        if orientation != "euler":
            raise ValueError(
                "MultiDiscreteActionWrapper requires an euler-orientation Environment "
                f"(delta_orient dim 3); got orientation={orientation!r}"
            )

        class _Impl(gym.ActionWrapper):
            def __init__(self, env, spec):
                super().__init__(env)
                self.spec = spec
                self.action_space = spaces.MultiDiscrete(spec.nvec())

            def action(self, action):
                return decode(action, self.spec, orient_dim=3)

        return _Impl(env, spec)
