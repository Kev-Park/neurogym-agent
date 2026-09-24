"""Could the simulator DRAW Chrome's UI instead of ngllib masking it out?

The masked regions are only worth reproducing if they are mostly constant: a
constant region can be baked from a Chrome capture once and composited, while a
state-dependent one needs glyph rendering that even two Chrome builds disagree
on (SS4.3: the axis labels differ appspot-vs-fork by font rasterisation alone).

So measure, over frames of the same scene at different states: which masked
pixels never change, which do, and what the varying ones are.

    uv run --no-sync python native/probe_ui_regions.py /scratch/kp0374/fork_match fork
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image

UI_REGIONS = (   # ngllib.simulator.pane2d, captured px in the 900x450 frame
    ("top strip (tabs, coords, icons)", 0, 32, 0, 900),
    ("2D pane left edge", 0, 450, 0, 16),
    ("2D scale bar", 416, 450, 0, 80),
    ("3D pane top-right buttons", 16, 48, 868, 900),
    ("3D 'Sections' control", 416, 450, 820, 900),
    ("3D pane left edge (UNMASKED today)", 0, 450, 450, 466),
)


def main() -> int:
    d = sys.argv[1] if len(sys.argv) > 1 else "/scratch/kp0374/fork_match"
    tag = sys.argv[2] if len(sys.argv) > 2 else "fork"
    cases = sys.argv[3:] or ["base", "zoom_in", "zoom_out", "rotated", "moved"]
    frames = [np.asarray(Image.open(f"{d}/{tag}_{c}.png").convert("RGB")) for c in cases]
    stack = np.stack(frames)
    print(f"{tag}: {len(frames)} frames {frames[0].shape} cases={','.join(cases)}")
    const = (stack == stack[0]).all(axis=0).all(axis=-1)   # per-pixel, all states equal
    print(f"whole frame: {const.mean()*100:.2f}% of pixels identical across states")
    total_px = 0
    for name, y0, y1, x0, x1 in UI_REGIONS:
        c = const[y0:y1, x0:x1]
        px = c.size
        total_px += px
        varying = np.argwhere(~c)
        where = ""
        if varying.size:
            ys, xs = varying[:, 0], varying[:, 1]
            where = (f" varying bbox y[{y0+ys.min()},{y0+ys.max()}] x[{x0+xs.min()},{x0+xs.max()}]")
        print(f"  {name:38s} {px:6d}px  constant={c.mean()*100:6.2f}%{where}")
    print(f"  masked total: {total_px}px = {total_px/const.size*100:.2f}% of the frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
