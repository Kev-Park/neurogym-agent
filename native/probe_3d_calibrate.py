"""Fit the 3D pane's top offset, and characterise the section plane's
zoom-dependent error.

SS4.4/probe_3d_offset established (a) a zoom-invariant +3-row offset of the whole
3D pane and (b) a plane-only error that grows with projectionScale. This probe
fixes both numbers instead of inferring them:

  1. Chrome frames are rendered once per zoom. For each candidate top offset
     the simulator's 3D pane is re-composited (shift the pane down by k rows)
     and scored -- a sweep over the one constant that `pane2d.TOOLBAR` fixes
     for the 2D pane but that the 3D panel may not share.
  2. Plane geometry is reported as a bounding box per renderer per zoom, so a
     size error (bbox w/h ratio != 1) is distinguishable from a position error
     (bbox centre offset).

Needs a GPU node.

    uv run --no-sync python native/probe_3d_calibrate.py --out-dir /scratch/kp0374/cal3d
"""

from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def masks(pane: np.ndarray, sat: int = 25, dark: int = 45):
    """(mesh, section-plane) masks of a 3D pane.

    `dark` is 45, not 20: DARK-SHADED MESH pixels (e.g. 10,34,10) have
    saturation under `sat` and so read as "grey". At 20 they dominated
    Chrome's plane mask and produced a fake zoom-dependent plane error
    (2026-09-18) that the rendered frames disproved.
    """
    mx = pane.max(axis=2).astype(np.int16)
    mn = pane.min(axis=2).astype(np.int16)
    mesh = (mx - mn) > sat
    return mesh, (~mesh) & (mx > dark)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = float((a | b).sum())
    return float((a & b).sum()) / u if u else 1.0


def bbox(m: np.ndarray):
    ys, xs = np.nonzero(m)
    if not ys.size:
        return None
    return (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))


def fmt_bbox(b):
    if b is None:
        return "empty"
    y0, y1, x0, x1 = b
    return f"y[{y0},{y1}] x[{x0},{x1}] h={y1-y0+1} w={x1-x0+1} cy={(y0+y1)/2:.1f} cx={(x0+x1)/2:.1f}"


def shift_down(pane: np.ndarray, k: int) -> np.ndarray:
    """Composite the pane k rows lower, blanking what moves off the top."""
    if k == 0:
        return pane
    out = np.zeros_like(pane)
    if k > 0:
        out[k:] = pane[:-k]
    else:
        out[:k] = pane[-k:]
    return out


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
    ap.add_argument("--out-dir", default="/scratch/kp0374/cal3d")
    ap.add_argument("--viewer-dist", default=os.environ.get("NGL_DIST"),
                    help="viewer build to serve; default = ngllib's packaged one")
    ap.add_argument("--settle-s", type=float, default=4.0)
    ap.add_argument("--max-shift", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from ngllib.chrome import ChromeRenderer
    from ngllib.simulator import SimulatorRenderer
    from ngllib.simulator.pane2d import TOOLBAR, mask_ui

    layout = dict(window_size=(1800, 900), capture_scale=0.5, left_pane=True, right_pane=True)
    base = ChromeRenderer(viewer=args.viewer_dist, screenshot_format="png",
                          **layout).default_state()
    ps = float(base["projectionScale"])
    states = {}
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
        st = copy.deepcopy(base)
        st["projectionScale"] = ps * mult
        states[f"ps_x{mult:g}"] = st

    cf = render(lambda: ChromeRenderer(viewer=args.viewer_dist,
                                       screenshot_format="png", **layout),
                "chrome", states, args.settle_s, args.out_dir)
    sf = render(lambda: SimulatorRenderer(**layout), "sim", states, args.settle_s, args.out_dir)

    print()
    print(f"=== 1. top-offset sweep (current pane2d.TOOLBAR={TOOLBAR}) ===", flush=True)
    print(f"{'state':10s} " + " ".join(f"k={k:+d}".rjust(8) for k in range(0, args.max_shift + 1)),
          flush=True)
    totals = {k: [] for k in range(0, args.max_shift + 1)}
    for name in states:
        cframe = mask_ui(cf[name])
        mid = cframe.shape[1] // 2
        cm, cp = masks(cframe[:, mid:])
        row = []
        for k in range(0, args.max_shift + 1):
            sframe = mask_ui(shift_down(sf[name], k))
            sm, sp = masks(sframe[:, mid:])
            score = (iou(cm, sm) + iou(cp, sp)) / 2.0     # mesh and plane together
            totals[k].append(score)
            row.append(f"{score:8.4f}")
        print(f"{name:10s} " + " ".join(row), flush=True)
    print("mean       " + " ".join(f"{np.mean(v):8.4f}" for v in totals.values()), flush=True)
    best_k = max(totals, key=lambda k: np.mean(totals[k]))
    print(f"-> best shift k={best_k}, i.e. 3D pane top = TOOLBAR + {best_k} = {TOOLBAR + best_k} "
          f"(capture px; x2 at full res)", flush=True)

    print()
    print("=== 2. section-plane geometry, unshifted ===", flush=True)
    for name in states:
        cframe, sframe = mask_ui(cf[name]), mask_ui(sf[name])
        mid = cframe.shape[1] // 2
        _, cp = masks(cframe[:, mid:])
        _, sp = masks(sframe[:, mid:])
        bc, bs = bbox(cp), bbox(sp)
        print(f"{name:10s} chrome {fmt_bbox(bc)}", flush=True)
        print(f"{'':10s} sim    {fmt_bbox(bs)}", flush=True)
        if bc and bs:
            hc, wc = bc[1] - bc[0] + 1, bc[3] - bc[2] + 1
            hs, ws = bs[1] - bs[0] + 1, bs[3] - bs[2] + 1
            print(f"{'':10s} ratio  h={hs/hc:.3f} w={ws/wc:.3f}  "
                  f"dcy={((bs[0]+bs[1])-(bc[0]+bc[1]))/2:+.1f} "
                  f"dcx={((bs[2]+bs[3])-(bc[2]+bc[3]))/2:+.1f}", flush=True)
    print("CAL3D-DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
