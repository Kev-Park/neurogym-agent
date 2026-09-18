"""Which 3D projection is off by ~3 px -- the mesh or the section plane?

SS4.4 found the simulator's coloured-mesh mask aligns best with Chrome's at
+3 rows, while shifting the WHOLE pane made mean|diff| worse. That is only
possible if the mesh and the section plane disagree with each other, so measure
them separately:

  - mesh mask      = saturated pixels (segment colours)
  - section plane  = grey, non-black pixels (the EM slice drawn in the 3D pane)

and do it across a zoom sweep: an offset constant in pixels points at the
viewport/principal point, one that scales with zoom points at the projection
matrix (FOV / near-far / scale factor).

Needs a GPU node (Chromium/ANGLE + EGL).

    uv run --no-sync python native/probe_3d_offset.py --out-dir /scratch/kp0374/offset3d
"""

from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def masks(pane: np.ndarray, sat: int = 25, dark: int = 20):
    """(mesh, section-plane) boolean masks of a 3D pane."""
    mx = pane.max(axis=2).astype(np.int16)
    mn = pane.min(axis=2).astype(np.int16)
    mesh = (mx - mn) > sat
    plane = ((mx - mn) <= sat) & (mx > dark)
    return mesh, plane


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = float((a | b).sum())
    return float((a & b).sum()) / u if u else 1.0


def best_offset(a: np.ndarray, b: np.ndarray, radius: int = 8):
    best = (0, 0, iou(a, b))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sa = a[max(0, dy):a.shape[0] + min(0, dy), max(0, dx):a.shape[1] + min(0, dx)]
            sb = b[max(0, -dy):b.shape[0] + min(0, -dy), max(0, -dx):b.shape[1] + min(0, -dx)]
            v = iou(sa, sb)
            if v > best[2]:
                best = (dy, dx, v)
    return best


def centroid(m: np.ndarray):
    ys, xs = np.nonzero(m)
    return (float(ys.mean()), float(xs.mean())) if ys.size else (float("nan"),) * 2


def render(make, tag, states, settle_s, out_dir):
    from PIL import Image

    r = make()
    frames = {}
    try:
        r.open()
        for name, st in states.items():
            r.reset_to(st)
            time.sleep(settle_s)
            _, img = r.observe()
            frames[name] = np.asarray(img).astype(np.uint8)
            Image.fromarray(frames[name]).save(f"{out_dir}/{tag}_{name}.png")
            print(f"  [{tag}] {name}", flush=True)
    finally:
        r.close()
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/scratch/kp0374/offset3d")
    ap.add_argument("--viewer-dist", default=os.environ.get("NGL_DIST", "/scratch/kp0374/ngl_fork_dist"))
    ap.add_argument("--settle-s", type=float, default=4.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from ngllib.chrome import ChromeRenderer
    from ngllib.simulator import SimulatorRenderer

    layout = dict(window_size=(1800, 900), capture_scale=0.5, left_pane=True, right_pane=True)
    base = ChromeRenderer(viewer_dist=args.viewer_dist, screenshot_format="png",
                          **layout).default_state()
    ps = float(base["projectionScale"])
    states = {}
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
        st = copy.deepcopy(base)
        st["projectionScale"] = ps * mult
        states[f"ps_x{mult:g}"] = st

    cf = render(lambda: ChromeRenderer(viewer_dist=args.viewer_dist,
                                       screenshot_format="png", **layout),
                "chrome", states, args.settle_s, args.out_dir)
    sf = render(lambda: SimulatorRenderer(**layout), "sim", states, args.settle_s, args.out_dir)

    print()
    print("=== 3D pane: mesh vs section plane, independently ===", flush=True)
    print(f"{'state':10s} {'part':6s} {'IoU@0':>7s} {'best dy':>8s} {'dx':>4s} {'IoU*':>7s} "
          f"{'cy chrome':>10s} {'cy sim':>8s} {'dcy':>6s}", flush=True)
    for name in states:
        mid = cf[name].shape[1] // 2
        cm, cp = masks(cf[name][:, mid:])
        sm, sp = masks(sf[name][:, mid:])
        for part, (a, b) in (("mesh", (cm, sm)), ("plane", (cp, sp))):
            dy, dx, best = best_offset(a, b)
            cy_a, _ = centroid(a)
            cy_b, _ = centroid(b)
            print(f"{name:10s} {part:6s} {iou(a, b):7.4f} {dy:+8d} {dx:+4d} {best:7.4f} "
                  f"{cy_a:10.2f} {cy_b:8.2f} {cy_b - cy_a:+6.2f}", flush=True)
    print("OFFSET3D-DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
