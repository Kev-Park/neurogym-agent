"""Is the 2D pane's VERTICAL geometry wrong, and is that the 0.10 gap?

The visual audit put the ceiling at 1.0000 (Chrome reproduces itself exactly on
settled frames) and our 2D pane at 0.8990 excluding the toolbar. JPEG explains
+0.003 of that and the toolbar strip +0.029, so most of the 0.10 is
unattributed.

The DOM measurement says why it might be. Read off the live page
(probe_select_parity --mode rects): the data panels sit at page y=47 and are
853 CSS px tall. At capture_scale 0.5 that is captured rows 23.5..450, i.e.
426.5 rows. We compose 17 black rows and 433 EM rows, and fetch an extent of
xs * 867 * 4 nm. So the pane is drawn ~6 rows too high and ~1.5% too tall, and
LEFT_SHIFT_PX = (-3, 0) has been absorbing the offset half of that empirically
while the SCALE error stays.

A 1.5% scale error over 433 rows is ~3 px of displacement at the top and bottom
edges, which block_ssim punishes hard on EM texture. This scores the corrected
geometry against Chrome, per state, and sweeps the toolbar offset so the answer
does not depend on guessing 23 vs 24.

    uv run --no-sync python native/probe_pane_geometry.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 10
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def block_ssim(a, b, block: int = 16) -> float:
    ga = np.asarray(a, np.float64)
    gb = np.asarray(b, np.float64)
    if ga.ndim == 3:
        ga = ga.mean(axis=2)
    if gb.ndim == 3:
        gb = gb.mean(axis=2)
    h = (ga.shape[0] // block) * block
    w = (ga.shape[1] // block) * block
    A = ga[:h, :w].reshape(h // block, block, w // block, block)
    B = gb[:h, :w].reshape(h // block, block, w // block, block)
    ma, mb = A.mean(axis=(1, 3)), B.mean(axis=(1, 3))
    va, vb = A.var(axis=(1, 3)), B.var(axis=(1, 3))
    cov = (A * B).mean(axis=(1, 3)) - ma * mb
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float((((2 * ma * mb + c1) * (2 * cov + c2))
                  / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    from PIL import Image

    from ngllib.native import pane2d
    from ngllib.native.em import EMTiles

    PANE = pane2d.PANE
    em = EMTiles(args.cache_dir)
    records = [json.loads(line) for line in
               open(os.path.join(args.pairs_dir, "states.jsonl"))][:args.limit]

    # (label, view height in CSS px, rows of toolbar, EM rows)
    variants = [
        ("v0_current    867/17/433", 867.0, 17, 433),
        ("v1_dom        853/23/427", 853.0, 23, 427),
        ("v2_dom_t24    853/24/426", 853.0, 24, 426),
        ("v3_dom_t22    853/22/428", 853.0, 22, 428),
        ("v4_853_t17    853/17/433", 853.0, 17, 433),
    ]
    acc: dict[str, list] = {v[0]: [] for v in variants}

    for rec in records:
        st = rec["requested_state"]
        fa = os.path.join(args.pairs_dir, "frames", f"{rec['idx']:04d}_a.png")
        if not os.path.exists(fa):
            continue
        ref = np.asarray(Image.open(fa))[:, :PANE, :3].mean(axis=2)

        xs = float(st["crossSectionScale"])
        pos = np.asarray(st["position"], np.float64) * pane2d.VOXEL_NM
        line = [f"[{rec['idx']:04d}]"]
        for label, view_h, tb, rows in variants:
            ext = (xs * pane2d.CSS_PANE * 4.0, xs * view_h * 4.0)
            shifted = pos + np.array([
                pane2d.LEFT_SHIFT_PX[1] * ext[0] / PANE,
                pane2d.LEFT_SHIFT_PX[0] * ext[1] / rows, 0.0])
            try:
                tile = em.tile(shifted, ext[0], ext[1], 1024, True)
            except Exception as e:  # noqa: BLE001
                print(f"[{rec['idx']:04d}] {label}: {e}", flush=True)
                continue
            big = Image.fromarray(tile).resize(
                (900, int(round(view_h))), Image.BILINEAR)
            img = np.asarray(big.resize((PANE, rows), Image.BOX)
                             ).astype(np.float32) * pane2d.EM_GAIN
            canvas = np.zeros((450, PANE), dtype=np.float32)
            canvas[tb:tb + rows] = np.clip(img, 0, 255)
            s = block_ssim(canvas, ref)
            acc[label].append(s)
            line.append(f"{label.split()[0]}={s:.4f}")
        print("  ".join(line), flush=True)

    print("\n============== 2D pane vertical geometry ==============")
    print("ceiling (Chrome vs itself, settled) is 1.0000")
    best = None
    for label, *_ in variants:
        if not acc[label]:
            continue
        med = float(np.median(acc[label]))
        print(f"  {label:<26}{med:.4f}")
        if best is None or med > best[1]:
            best = (label, med)
    if best:
        base = float(np.median(acc[variants[0][0]])) if acc[variants[0][0]] \
            else float("nan")
        print(f"\nbest: {best[0]} at {best[1]:.4f}, "
              f"{best[1] - base:+.4f} against what ships")
    print("\nThe crosshair and the segment tint are deliberately left out: this "
          "isolates the EM raster's placement, which is what a scale error "
          "would move. If the DOM geometry wins clearly, the shipping "
          "constants are simply wrong and LEFT_SHIFT_PX has been hiding half "
          "of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
