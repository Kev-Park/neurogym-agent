"""Decompose the 2D (left) pane's simulator-vs-Chrome parity error.

Whole-pane left SSIM sits at ~0.898. Before changing any calibration constant
this answers two questions in order:

  1. WHAT IS THE CEILING? Each collected state has TWO browser frames (_a, _b)
     of the same state. Chrome's own a-vs-b left-pane similarity is the noise
     floor: the 2D pane streams chunks, so two captures of one state need not
     agree. If sim-vs-Chrome is already at that floor there is nothing to fix
     and any "improvement" is fitting Chrome's own jitter.

  2. WHERE DOES THE RESIDUAL LIVE? Attribute it across the pieces of
     pane2d.compose_left, one at a time:
       registration  - grid-search (dy, dx) shifts; is LEFT_SHIFT_PX optimal?
       intensity     - least-squares gain/offset on plain EM pixels; is
                       EM_GAIN optimal?
       overlays      - re-score with the crosshair and the segment tint masked
                       out, isolating the EM filter chain itself
       resample      - what is left after the above is the
                       900x867 BILINEAR -> 450x433 BOX chain vs Chrome's own

Reports gap-to-CEILING, not gap-to-1.0.

    uv run --no-sync python native/probe_left_pane_parity.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 24
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Global SSIM on greyscale float images, optionally over a mask only."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if mask is not None:
        a, b = a[mask], b[mask]
    if a.size == 0:
        return float("nan")
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2))
                 / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def grey(img: np.ndarray) -> np.ndarray:
    return img[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def sharpness(g: np.ndarray, mask: np.ndarray) -> float:
    """Mean |gradient| over `mask` -- a blunt proxy for retained detail.

    If the simulator's value sits well BELOW Chrome's, we are sampling a coarser
    mip (or over-blurring in the resample chain) and the lost high frequencies
    cannot be recovered by any gain or shift.
    """
    gy, gx = np.gradient(g)
    mag = np.hypot(gy, gx)
    return float(mag[mask].mean()) if mask.any() else float("nan")


def best_shift(sim: np.ndarray, ref: np.ndarray, radius: int = 6):
    """(dy, dx, ssim) maximizing similarity — is the baked LEFT_SHIFT_PX right?"""
    best = (0, 0, -1.0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(sim, dy, axis=0), dx, axis=1)
            m = radius + 1
            s = ssim(shifted[m:-m, m:-m], ref[m:-m, m:-m])
            if s > best[2]:
                best = (dy, dx, s)
    return best


def fit_gain(sim: np.ndarray, ref: np.ndarray, mask: np.ndarray):
    """Least-squares ref ~= gain*sim + offset over plain-EM pixels."""
    x, y = sim[mask], ref[mask]
    if x.size < 32:
        return float("nan"), float("nan")
    A = np.stack([x, np.ones_like(x)], axis=1)
    gain, offset = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(gain), float(offset)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--shift-radius", type=int, default=6)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    from ngllib.native import pane2d
    from ngllib.native.em import worker_visuals

    PANE, TOOLBAR, PANE_H = pane2d.PANE, pane2d.TOOLBAR, pane2d.PANE_H

    records = [json.loads(l) for l in
               open(os.path.join(args.pairs_dir, "states.jsonl"))]
    if args.limit:
        records = records[:args.limit]

    # Overlay masks are deterministic from geometry (see pane2d.compose_left):
    # a one-sided crosshair at the pane centre, drawn below the toolbar strip.
    cy, cx = TOOLBAR + PANE_H // 2, PANE // 2
    length = int(min(900, 867) / 4 / 2)
    cross = np.zeros((PANE, PANE), bool)
    cross[cy, cx:cx + length] = True
    cross[cy:cy + length, cx] = True
    toolbar = np.zeros((PANE, PANE), bool)
    toolbar[:TOOLBAR] = True

    rows = []
    for rec in records:
        st = rec["requested_state"]
        fa = os.path.join(args.pairs_dir, "frames", f"{rec['idx']:04d}_a.png")
        fb = os.path.join(args.pairs_dir, "frames", f"{rec['idx']:04d}_b.png")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        a = np.asarray(Image.open(fa))[:, :PANE, :3]
        b = np.asarray(Image.open(fb))[:, :PANE, :3]

        try:
            canvas, _plane = worker_visuals(
                args.cache_dir, list(st["position"]),
                float(st["crossSectionScale"]), str(st["segments"][0]))
        except Exception as e:  # noqa: BLE001
            print(f"[{rec['idx']:04d}] render failed: {e}", flush=True)
            continue

        ga, gb, gs = grey(a), grey(b), grey(canvas)
        # Tint pixels: where the simulator painted segment colour (not grey).
        tint = canvas[..., :3].std(axis=2) > 6
        plain = ~(cross | toolbar | tint)

        if gs.std() < 1.0:   # blank/failed tile fetch -- not a parity datum
            print(f"[{rec['idx']:04d}] SKIP: simulator pane is blank "
                  f"(std={gs.std():.2f}); tile fetch failed", flush=True)
            continue

        ceiling = ssim(ga, gb)                    # Chrome vs itself
        parity = ssim(gs, ga)                     # simulator vs Chrome
        parity_plain = ssim(gs, ga, plain)        # overlays excluded
        dy, dx, parity_shift = best_shift(gs, ga, args.shift_radius)
        gain, offset = fit_gain(gs, ga, plain)

        sh_sim, sh_chrome = sharpness(gs, plain), sharpness(ga, plain)
        rows.append(dict(idx=rec["idx"], ceiling=ceiling, parity=parity,
                         parity_plain=parity_plain, parity_shift=parity_shift,
                         dy=dy, dx=dx, gain=gain, offset=offset,
                         tint_frac=float(tint.mean()),
                         sharp_sim=sh_sim, sharp_chrome=sh_chrome,
                         sharp_ratio=sh_sim / sh_chrome if sh_chrome else float("nan")))
        print(f"[{rec['idx']:04d}] ceiling(a,b)={ceiling:.4f}  parity={parity:.4f}  "
              f"plain={parity_plain:.4f}  best_shift=({dy:+d},{dx:+d})->{parity_shift:.4f}  "
              f"gain={gain:.4f} off={offset:+.1f} sharp={sh_sim:.2f}/{sh_chrome:.2f}",
              flush=True)

    if not rows:
        print("no usable pairs")
        return 1

    def med(k):
        return float(np.median([r[k] for r in rows]))

    print("\n================ 2D-pane parity decomposition ================")
    print(f"pairs                       : {len(rows)}")
    print(f"CEILING  Chrome a vs b      : {med('ceiling'):.4f}   <- the achievable target")
    print(f"parity   sim vs Chrome      : {med('parity'):.4f}")
    print(f"  gap to ceiling            : {med('ceiling') - med('parity'):+.4f}")
    print(f"parity, overlays masked     : {med('parity_plain'):.4f}")
    print(f"parity, best re-registration: {med('parity_shift'):.4f}")
    print(f"  median best shift (dy,dx) : ({med('dy'):+.1f}, {med('dx'):+.1f})  "
          f"[LEFT_SHIFT_PX={pane2d.LEFT_SHIFT_PX}]")
    print(f"fitted gain                 : {med('gain'):.4f}   [EM_GAIN={pane2d.EM_GAIN}]")
    print(f"fitted offset               : {med('offset'):+.2f}")
    print(f"tint coverage               : {100 * med('tint_frac'):.1f}% of pane")
    print(f"detail  sim / chrome        : {med('sharp_sim'):.2f} / {med('sharp_chrome'):.2f}"
          f"   ratio {med('sharp_ratio'):.3f}")
    print("  ratio <1 => simulator is BLURRIER (coarser mip or over-smoothing resample);")
    print("  ratio >1 => simulator is SHARPER (finer mip than NG displays).")
    print("\nReading: if `gap to ceiling` is ~0 the pane is already at parity and the "
          "remaining error is Chrome's own chunk-streaming jitter, not a calibration "
          "defect. A non-zero median best shift means LEFT_SHIFT_PX is mis-set; a "
          "fitted gain far from EM_GAIN means the intensity constant is.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
