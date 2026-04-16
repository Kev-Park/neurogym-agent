from dataclasses import dataclass

import numpy as np


@dataclass
class ActionSpec:
    grid_rows: int = 32
    grid_cols: int = 32
    pane_x0: int = 900
    pane_y0: int = 0
    pane_x1: int = 1800
    pane_y1: int = 900
    rotation_bins_per_axis: int = 9
    rotation_step_rad: float = 0.08

    @property
    def num_cells(self) -> int:
        return self.grid_rows * self.grid_cols

    def multidiscrete_nvec(self) -> list[int]:
        return [
            self.num_cells,
            2,
            self.rotation_bins_per_axis,
            self.rotation_bins_per_axis,
            self.rotation_bins_per_axis,
        ]


def _bin_to_signed_magnitude(bin_index: int, bins_per_axis: int, step: float) -> float:
    half = bins_per_axis // 2
    return (bin_index - half) * step


def cell_to_pixel(cell: int, spec: ActionSpec) -> tuple[float, float]:
    row = cell // spec.grid_cols
    col = cell % spec.grid_cols
    cell_w = (spec.pane_x1 - spec.pane_x0) / spec.grid_cols
    cell_h = (spec.pane_y1 - spec.pane_y0) / spec.grid_rows
    x = spec.pane_x0 + (col + 0.5) * cell_w
    y = spec.pane_y0 + (row + 0.5) * cell_h
    return float(x), float(y)


def decode(md_action, spec: ActionSpec) -> tuple[list, bool]:
    """
    Translate a MultiDiscrete sample into the full 17-element neurogym action vector
    (Euler-angle mode). Returns (action_vector, right_click_fired).

    Index layout of the returned vector:
        0 left_click (0)
        1 right_click
        2 double_click (0)
        3 x
        4 y
        5 shift (0)
        6 ctrl  (0)
        7 alt   (0)
        8 json_change (0)
        9,10,11 delta_position_xyz (0)
        12 delta_crossSectionScale (0)
        13,14,15 delta_orientation_euler
        16 delta_projectionScale (0)
    """
    cell, click_type, d_ex, d_ey, d_ez = (int(v) for v in md_action)
    right_click = 1 if click_type == 1 else 0
    x, y = cell_to_pixel(cell, spec)

    dex = _bin_to_signed_magnitude(d_ex, spec.rotation_bins_per_axis, spec.rotation_step_rad)
    dey = _bin_to_signed_magnitude(d_ey, spec.rotation_bins_per_axis, spec.rotation_step_rad)
    dez = _bin_to_signed_magnitude(d_ez, spec.rotation_bins_per_axis, spec.rotation_step_rad)

    vec = [0.0] * 17
    vec[1] = float(right_click)
    vec[3] = x
    vec[4] = y
    vec[8] = 1.0 if (right_click == 0 and (dex != 0 or dey != 0 or dez != 0)) else 0.0
    vec[13] = dex
    vec[14] = dey
    vec[15] = dez
    return vec, bool(right_click)


def sample_reset_perturbation(
    spec: ActionSpec,
    rng: np.random.Generator,
    rotation_perturb_rad: float,
    zoom_perturb_frac: float,
) -> list:
    vec = [0.0] * 17
    vec[8] = 1.0
    vec[13] = float(rng.uniform(-rotation_perturb_rad, rotation_perturb_rad))
    vec[14] = float(rng.uniform(-rotation_perturb_rad, rotation_perturb_rad))
    vec[15] = float(rng.uniform(-rotation_perturb_rad, rotation_perturb_rad))
    vec[16] = float(rng.uniform(-zoom_perturb_frac, zoom_perturb_frac))
    return vec
