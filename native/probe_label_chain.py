"""A/B the 2D pane's SEGMENT-TINT chain against Chrome.

The dynamics run left one steady-state difference: the simulator tints 0.032 of
the 2D pane where Chrome tints 0.059, flat across every frame. Hypothesis:
Neuroglancer renders the segmentation at CSS resolution (900x867) and Chrome's
capture area-averages that down to 450, so a process one pixel wide survives as
a partially tinted pixel. `label_tile` instead picks a mip for a 450-wide tile
and NEAREST-resizes the mask, which drops those outright.

Variants, all composed the same way otherwise so only the mask chain differs:

  v0_current    mip for a 450 px tile, NEAREST mask -> 450x433, binary alpha
  v1_css_box    mip for a 900 px tile, NEAREST mask -> 900x867, BOX down to
                450x433 -> FRACTIONAL alpha (what Chrome's pipeline produces)
  v2_same_mip   as v1 but keeping v0's coarser mip, so the two effects --
                finer mip vs fractional coverage -- are separated

Scored against Chrome's own tint on the same states: tinted area, mask IoU, and
block_ssim of the whole 2D pane.

    uv run --no-sync python native/probe_label_chain.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 12
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

COLOURED = 18.0   # channel spread above this is segment tint, not grey EM


def block_ssim(a, b, block: int = 16) -> float:
    ga = np.asarray(a, np.float64).mean(axis=2)
    gb = np.asarray(b, np.float64).mean(axis=2)
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


def iou(a, b) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    from PIL import Image

    from ngllib.simulator import pane2d
    from ngllib.simulator.colors import segment_color
    from ngllib.simulator.em import EMTiles, Source

    PANE, TOOLBAR, PANE_H = pane2d.PANE, pane2d.TOOLBAR, pane2d.PANE_H
    em = EMTiles(Source.calibrated(args.cache_dir))
    records = [json.loads(line) for line in
               open(os.path.join(args.pairs_dir, "states.jsonl"))][:args.limit]

    # crosshair geometry, to keep it out of every tint mask
    cy, cx = PANE_H // 2, PANE // 2
    length = int(min(900, 867) / 4 / 2)
    cross = np.zeros((PANE_H, PANE), bool)
    cross[cy, cx:cx + length] = True
    cross[cy:cy + length, cx] = True

    variants = ["v0_current", "v1_css_box", "v2_same_mip"]
    acc = {v: {"area": [], "iou": [], "ssim": []} for v in variants}
    chrome_area = []

    for rec in records:
        st = rec["requested_state"]
        fa = os.path.join(args.pairs_dir, "frames", f"{rec['idx']:04d}_a.png")
        if not os.path.exists(fa):
            continue
        ref = np.asarray(Image.open(fa))[:, :PANE, :3][TOOLBAR:]
        rid = int(st["segments"][0])
        pos_nm = np.asarray(st["position"], np.float64) * pane2d.VOXEL_NM
        xs = float(st["crossSectionScale"])
        ext = pane2d.pane_extents_nm(xs)
        shifted = pane2d.shifted_fetch_center_nm(pos_nm, ext)

        try:
            tile = em.tile(shifted, ext[0], ext[1], 1024, True)
        except Exception as e:  # noqa: BLE001
            print(f"[{rec['idx']:04d}] EM fetch failed: {e}", flush=True)
            continue
        if tile.std() < 1.0:
            print(f"[{rec['idx']:04d}] SKIP blank tile", flush=True)
            continue

        base = np.asarray(Image.fromarray(tile).resize((900, 867), Image.BILINEAR)
                          .resize((PANE, PANE_H), Image.BOX)
                          ).astype(np.float32) * pane2d.EM_GAIN
        base = np.repeat(base[..., None], 3, axis=2)
        col = np.asarray(segment_color(rid)) * 255.0

        # Chrome's own tint, crosshair excluded
        ref_tint = (np.asarray(ref, np.float64).std(axis=2) > COLOURED) & ~cross
        chrome_area.append(float(ref_tint.mean()))

        alphas = {}
        m0 = em.label_tile(shifted, ext[0], ext[1], rid, (PANE, PANE_H))
        alphas["v0_current"] = None if m0 is None else m0.astype(np.float32)

        for name, out_px, res_div in (("v1_css_box", (900, 867), 900),
                                      ("v2_same_mip", (900, 867), PANE)):
            # label_tile picks its mip from extent/out_px[0]; res_div selects
            # which resolution that decision is made at, so v2 keeps v0's mip
            # while still producing fractional coverage.
            lab = em._label_cutout(shifted, ext[0], ext[1], (res_div, res_div))
            if lab is None:
                alphas[name] = None
                continue
            mask = (lab == rid).astype(np.uint8) * 255
            big = Image.fromarray(mask).resize(out_px, Image.NEAREST)
            alphas[name] = (np.asarray(big.resize((PANE, PANE_H), Image.BOX))
                            .astype(np.float32) / 255.0)

        line = [f"[{rec['idx']:04d}] chrome {ref_tint.mean():.4f}"]
        for v in variants:
            a = alphas.get(v)
            if a is None:
                continue
            rgb = base.copy()
            w = (0.5 * a)[..., None]
            rgb = rgb * (1.0 - w) + col[None, None, :] * w
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            tint = (a > 0.02) & ~cross
            acc[v]["area"].append(float(tint.mean()))
            acc[v]["iou"].append(iou(tint, ref_tint))
            acc[v]["ssim"].append(block_ssim(rgb, ref))
            line.append(f"{v.split('_')[0]} {tint.mean():.4f}/"
                        f"iou {acc[v]['iou'][-1]:.3f}")
        print("  ".join(line), flush=True)

    if not chrome_area:
        print("no usable pairs")
        return 1

    ca = float(np.median(chrome_area))
    print("\n============== 2D segment-tint chain ==============")
    print(f"pairs {len(chrome_area)}")
    print(f"{'variant':<14}{'area':>9}{'ratio':>8}{'IoU':>8}{'ssim':>8}")
    for v in variants:
        if not acc[v]["area"]:
            continue
        a = float(np.median(acc[v]["area"]))
        print(f"{v:<14}{a:>9.4f}{a / ca:>8.2f}"
              f"{float(np.median(acc[v]['iou'])):>8.3f}"
              f"{float(np.median(acc[v]['ssim'])):>8.4f}")
    print(f"{'chrome (ref)':<14}{ca:>9.4f}{1.0:>8.2f}")
    print("\nIf v1 lands near ratio 1.00 with a higher IoU, Chrome's tint area "
          "really is antialiased coverage we are throwing away. If v2 also "
          "does, the fractional alpha alone explains it and the finer mip is "
          "not needed -- which matters, because the finer mip costs 4x the "
          "label voxels on the fetch path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
