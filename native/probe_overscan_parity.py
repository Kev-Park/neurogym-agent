"""Does the overscan+crop path change the observation?

The region cache only earns its place if a pane cropped out of an overscanned
fetch is the SAME pane a direct fetch would have produced. The nm-per-pixel is
held equal by construction (extent and max_px scale together, so the same mip
is chosen), but two things could still shift pixels: the crop rounds the offset
to whole raster pixels, and the BILINEAR->BOX chain now runs over a larger
image before the crop.

So compare, on real data: direct fetch at the pane extent, against a crop of an
overscanned fetch at the same centre, and at centres offset by a fraction of
the pane -- the case the cache exists for.

    uv run --no-sync python native/probe_overscan_parity.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 6
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
        ga, gb = ga.mean(axis=2), gb.mean(axis=2)
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
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--overscan", type=float, default=1.5)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    from ngllib.native import pane2d
    from ngllib.native.em import unpack_ids, worker_pane_parts

    records = [json.loads(line) for line in
               open(os.path.join(args.pairs_dir, "states.jsonl"))][:args.limit]

    same, moved, idmatch = [], [], []
    ref_vs_chrome, crop_vs_chrome = [], []
    for rec in records:
        st = rec["requested_state"]
        pos = list(st["position"])
        xs = float(st["crossSectionScale"])
        ext = pane2d.pane_extents_nm(xs)

        try:
            em_ref, ids_ref_p, _pl, c_ref, e_ref = worker_pane_parts(
                args.cache_dir, pos, xs, 1024, True, 1.0)
            em_big, ids_big_p, _pl2, c_big, e_big = worker_pane_parts(
                args.cache_dir, pos, xs, 1024, True, args.overscan)
        except Exception as e:  # noqa: BLE001
            print(f"[{rec['idx']:04d}] fetch failed: {e}", flush=True)
            continue
        if em_ref is None or em_big is None:
            continue

        want = np.asarray(c_ref, dtype=np.float64)
        crop = pane2d.crop_pane(em_big, np.asarray(c_big), e_big, want)
        if crop is None:
            print(f"[{rec['idx']:04d}] centre crop did not fit", flush=True)
            continue
        s0 = block_ssim(crop, em_ref)
        same.append(s0)

        # The question that decides this is NOT crop-vs-direct: both are
        # approximations of Chrome, and if they are equally close to it then a
        # sub-pixel difference between them costs nothing. Score both against
        # the browser frame this state was collected from.
        fa = os.path.join(args.pairs_dir, "frames", f"{rec['idx']:04d}_a.png")
        if os.path.exists(fa):
            from PIL import Image

            ref_frame = np.asarray(Image.open(fa))[
                pane2d.TOOLBAR:, :pane2d.PANE, :3].mean(axis=2)
            ref_vs_chrome.append(block_ssim(em_ref, ref_frame))
            crop_vs_chrome.append(block_ssim(crop, ref_frame))

        # ids must survive the crop exactly -- they are label values, and a
        # single wrong id tints the wrong neuron.
        ids_big = unpack_ids(ids_big_p)
        ids_ref = unpack_ids(ids_ref_p)
        if ids_big is not None and ids_ref is not None:
            ic = pane2d.crop_pane(ids_big, np.asarray(c_big), e_big, want)
            if ic is not None:
                idmatch.append(float((ic == ids_ref).mean()))

        # the case the cache exists for: a move of 15% of the pane, cropped
        # from the SAME overscanned fetch, against a direct fetch there
        dx = ext[0] * 0.15
        moved_pos = [pos[0] + dx / pane2d.VOXEL_NM[0], pos[1], pos[2]]
        try:
            em_mref, _i, _p, c_mref, _e = worker_pane_parts(
                args.cache_dir, moved_pos, xs, 1024, True, 1.0)
        except Exception as e:  # noqa: BLE001
            print(f"[{rec['idx']:04d}] moved fetch failed: {e}", flush=True)
            continue
        mcrop = pane2d.crop_pane(em_big, np.asarray(c_big), e_big,
                                 np.asarray(c_mref, dtype=np.float64))
        if mcrop is None:
            print(f"[{rec['idx']:04d}] 15% move did NOT fit in the overscan",
                  flush=True)
            continue
        s1 = block_ssim(mcrop, em_mref)
        moved.append(s1)
        print(f"[{rec['idx']:04d}] centre {s0:.4f}   moved-15% {s1:.4f}",
              flush=True)

    if not same:
        print("no usable states")
        return 1
    print("\n============== overscan crop vs direct fetch ==============")
    print(f"states                      : {len(same)}")
    print(f"block_ssim, same centre     : {np.median(same):.4f}")
    if ref_vs_chrome:
        rc, cc = np.median(ref_vs_chrome), np.median(crop_vs_chrome)
        print(f"vs CHROME, direct fetch     : {rc:.4f}")
        print(f"vs CHROME, overscan crop    : {cc:.4f}   (delta {cc - rc:+.4f})")
    if moved:
        print(f"block_ssim, 15% move        : {np.median(moved):.4f}")
    if idmatch:
        print(f"id map pixels identical     : {100 * np.median(idmatch):.2f}%")
    print("\n1.0000 means the crop path is the same observation and the cache "
          "is free to use. Anything materially below it changes what every run "
          "sees and has to go through the same gates as a calibration change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
