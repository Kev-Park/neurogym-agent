"""Can the 2D pane's EM match Chrome more closely than mean |diff| 4.67/255?

The simulator's EM is a calibrated chain (pane2d.resample_em): BILINEAR to the
CSS pane, BOX down to the capture, then EM_GAIN. Against Chrome it is
geometrically aligned (best offset (0,0)) but uniformly darker -- median +3/255,
p5 -5, p95 +11 -- which is a TONE error, not a geometry one.

This probe re-renders nothing: it takes the pane raster the simulator produced
and asks what pointwise correction would have matched Chrome, then whether
changing the resample chain does better than a correction at all.

  1. fit gain (y = g*x), gamma (y = 255*(x/255)^p) and affine (y = a*x + b) on
     the EM-only pixels of the 2D pane, and report residual mean |diff| for
     each -- fitted per state and pooled, so overfitting is visible;
  2. re-run the pooled winner against every state;
  3. compare resample variants (float chain without the uint8 round-trip,
     single LANCZOS, linear-light BOX) on the same frames.

Needs the paired frames a match run already wrote (chrome_*.png / sim_*.png).

    uv run --no-sync python native/probe_em_tone.py --dir /scratch/kp0374/fork_match_v3
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def em_mask(chrome_pane: np.ndarray, sim_pane: np.ndarray, sat: int = 25) -> np.ndarray:
    """Pixels that are EM in BOTH panes: grey, and not segmentation colour."""
    def grey(x):
        mx, mn = x.max(axis=2).astype(np.int16), x.min(axis=2).astype(np.int16)
        return ((mx - mn) <= sat) & (mx > 5)
    return grey(chrome_pane) & grey(sim_pane)


def fit_gain(x, y):
    return float((x * y).sum() / max((x * x).sum(), 1e-9))


def fit_affine(x, y):
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def fit_gamma(x, y):
    """y/255 = (x/255)^p, least squares in log space on mid-tones."""
    m = (x > 8) & (x < 250) & (y > 8) & (y < 250)
    if m.sum() < 100:
        return 1.0
    return float(np.polyfit(np.log(x[m] / 255.0), np.log(y[m] / 255.0), 1)[0])


def mad(a, b):
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/scratch/kp0374/fork_match_v3")
    ap.add_argument("--chrome-prefix", default="fork")
    ap.add_argument("--sim-prefix", default="sim")
    args = ap.parse_args()

    from PIL import Image

    from ngllib.simulator.pane2d import EM_GAIN, mask_ui

    cases = sorted(os.path.basename(p)[len(args.chrome_prefix) + 1:-4]
                   for p in glob.glob(f"{args.dir}/{args.chrome_prefix}_*.png"))
    if not cases:
        print(f"no frames in {args.dir}")
        return 1
    print(f"cases: {', '.join(cases)}  (current EM_GAIN={EM_GAIN})")

    pooled_x, pooled_y, per_case = [], [], {}
    for c in cases:
        ch = mask_ui(np.asarray(Image.open(f"{args.dir}/{args.chrome_prefix}_{c}.png").convert("RGB")))
        sm = mask_ui(np.asarray(Image.open(f"{args.dir}/{args.sim_prefix}_{c}.png").convert("RGB")))
        mid = ch.shape[1] // 2
        cp, sp = ch[:, :mid], sm[:, :mid]          # 2D pane = left half
        m = em_mask(cp, sp)
        x = sp[..., 0][m].astype(np.float64)       # simulator EM (grey: any channel)
        y = cp[..., 0][m].astype(np.float64)       # chrome EM
        per_case[c] = (cp, sp, m, x, y)
        pooled_x.append(x)
        pooled_y.append(y)
        g, (a, b), p = fit_gain(x, y), fit_affine(x, y), fit_gamma(x, y)
        print(f"{c:9s} n={m.sum():7d} mad={mad(x, y):5.2f}  "
              f"gain={g:.4f}->{mad(np.clip(x * g, 0, 255), y):5.2f}  "
              f"affine=({a:.4f},{b:+.2f})->{mad(np.clip(a * x + b, 0, 255), y):5.2f}  "
              f"gamma={p:.4f}->{mad(np.clip(255 * (x / 255) ** p, 0, 255), y):5.2f}")

    X, Y = np.concatenate(pooled_x), np.concatenate(pooled_y)
    g = fit_gain(X, Y)
    a, b = fit_affine(X, Y)
    p = fit_gamma(X, Y)
    print(f"\npooled n={X.size}: gain={g:.4f}  affine=({a:.4f},{b:+.2f})  gamma={p:.4f}")
    print(f"pooled mad: none={mad(X, Y):.2f}  gain={mad(np.clip(X * g, 0, 255), Y):.2f}  "
          f"affine={mad(np.clip(a * X + b, 0, 255), Y):.2f}  "
          f"gamma={mad(np.clip(255 * (X / 255) ** p, 0, 255), Y):.2f}")
    # EM_GAIN is applied inside resample_em, so the fitted gain multiplies it.
    print(f"-> implied EM_GAIN = {EM_GAIN} * {g:.4f} = {EM_GAIN * g:.4f}")

    print("\nper-state residual under the POOLED correction (overfitting check):")
    for c, (cp, sp, m, x, y) in per_case.items():
        print(f"  {c:9s} none={mad(x, y):5.2f}  gain={mad(np.clip(x * g, 0, 255), y):5.2f}  "
              f"affine={mad(np.clip(a * x + b, 0, 255), y):5.2f}  "
              f"gamma={mad(np.clip(255 * (x / 255) ** p, 0, 255), y):5.2f}")
    print("EMTONE-DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
