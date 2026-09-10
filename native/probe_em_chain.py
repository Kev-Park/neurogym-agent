"""A/B the 2D pane's EM resample chain against Chrome.

The decomposition probe showed registration is already optimal and the residual
is in the EM texture: the simulator is ~13.5% blurrier than Chrome (mean
|gradient| ratio 0.865). Two candidates -- a mip coarser than Neuroglancer
displays, or the two-step resample in pane2d.compose_left
(tile -> BILINEAR 900x867 -> BOX 450x433) smoothing more than Chrome's single
area-average capture downscale.

This scores candidate chains on the EM region only (crosshair, segment tint and
toolbar masked out) so the comparison isolates the filter chain from the
overlays. Reports SSIM to Chrome and the detail ratio for each.

    uv run --no-sync python native/probe_em_chain.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 10
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def ssim(a, b, mask=None):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if mask is not None:
        a, b = a[mask], b[mask]
    if a.size == 0:
        return float("nan")
    mu_a, mu_b = a.mean(), b.mean()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2))
                 / ((mu_a ** 2 + mu_b ** 2 + c1) * (a.var() + b.var() + c2)))


def grey(img):
    return np.asarray(img)[..., :3].astype(np.float64) @ [0.299, 0.587, 0.114]


def detail(g, mask):
    gy, gx = np.gradient(g)
    return float(np.hypot(gy, gx)[mask].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    from ngllib.native import pane2d
    from ngllib.native.em import EMTiles

    PANE, TOOLBAR, PANE_H = pane2d.PANE, pane2d.TOOLBAR, pane2d.PANE_H
    GAIN = pane2d.EM_GAIN

    em = EMTiles(args.cache_dir)
    records = [json.loads(l) for l in
               open(os.path.join(args.pairs_dir, "states.jsonl"))][:args.limit]

    # same overlay geometry compose_left uses, in EM-region coordinates
    cy, cx = PANE_H // 2, PANE // 2
    length = int(min(900, 867) / 4 / 2)
    over = np.zeros((PANE_H, PANE), bool)
    over[cy, cx:cx + length] = True
    over[cy:cy + length, cx] = True

    variants = ["v0_current", "v1_direct_box", "v2_direct_lanczos", "v3_finer_lanczos"]
    acc = {v: {"ssim": [], "detail": []} for v in variants}
    chrome_detail = []
    mips = []

    for rec in records:
        st = rec["requested_state"]
        fa = os.path.join(args.pairs_dir, "frames", f"{rec['idx']:04d}_a.png")
        if not os.path.exists(fa):
            continue
        ref_full = grey(Image.open(fa))[:, :PANE]
        ref = ref_full[TOOLBAR:]                      # EM region of Chrome's pane

        pos = np.asarray(st["position"], np.float64) * pane2d.VOXEL_NM
        xs = float(st["crossSectionScale"])
        ex, ey = pane2d.pane_extents_nm(xs)
        shifted = pane2d.shifted_fetch_center_nm(pos, (ex, ey))

        try:
            t_std = em.tile(shifted, ex, ey, 1024, True)      # what ships today
            t_fine = em.tile(shifted, ex, ey, 4096, True)     # finest available
        except Exception as e:  # noqa: BLE001
            print(f"[{rec['idx']:04d}] fetch failed: {e}", flush=True)
            continue
        if t_std.std() < 1.0:
            print(f"[{rec['idx']:04d}] SKIP blank tile", flush=True)
            continue
        mips.append((t_std.shape[1], t_fine.shape[1]))

        def chain(tile, mode):
            im = Image.fromarray(tile)
            if mode == "v0_current":
                im = im.resize((900, 867), Image.BILINEAR)
                im = im.resize((PANE, PANE_H), Image.BOX)
            elif mode == "v1_direct_box":
                im = im.resize((PANE, PANE_H), Image.BOX)
            else:
                im = im.resize((PANE, PANE_H), Image.LANCZOS)
            return np.asarray(im).astype(np.float64) * GAIN

        imgs = {
            "v0_current": chain(t_std, "v0_current"),
            "v1_direct_box": chain(t_std, "v1_direct_box"),
            "v2_direct_lanczos": chain(t_std, "v2_direct_lanczos"),
            "v3_finer_lanczos": chain(t_fine, "v3_finer_lanczos"),
        }
        plain = ~over
        for v, img in imgs.items():
            acc[v]["ssim"].append(ssim(img, ref, plain))
            acc[v]["detail"].append(detail(img, plain))
        chrome_detail.append(detail(ref, plain))
        print(f"[{rec['idx']:04d}] tile {t_std.shape[1]}px (fine {t_fine.shape[1]}px)  " +
              "  ".join(f"{v.split('_',1)[0]}={acc[v]['ssim'][-1]:.4f}" for v in variants),
              flush=True)

    if not chrome_detail:
        print("no usable pairs")
        return 1

    cd = float(np.median(chrome_detail))
    print("\n============== EM resample chain A/B (EM region only) ==============")
    print(f"pairs {len(chrome_detail)}   median tile px "
          f"{int(np.median([m[0] for m in mips]))} (finest available "
          f"{int(np.median([m[1] for m in mips]))})")
    print(f"{'variant':<20}{'SSIM':>9}{'detail':>9}{'ratio':>8}")
    for v in variants:
        s = float(np.median(acc[v]["ssim"]))
        d = float(np.median(acc[v]["detail"]))
        print(f"{v:<20}{s:>9.4f}{d:>9.2f}{d / cd:>8.3f}")
    print(f"{'chrome (ref)':<20}{'-':>9}{cd:>9.2f}{1.0:>8.3f}")
    print("\nPick the variant with the highest SSIM whose detail ratio is nearest 1.0. "
          "If v3 wins on detail but not SSIM, we are sharper than Chrome and the "
          "current mip is right; if v1/v2 beat v0, the two-step resample is the defect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
