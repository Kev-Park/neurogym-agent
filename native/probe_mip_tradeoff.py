"""Is there a mip that is both sharp enough AND fast enough?

The successive-move curve killed caching as a route to Chrome's step-0 response
on the 2D pane: each move walks to genuinely new data, so there is nothing to
re-crop or prewarm (native 11, 41, 41, 41, 40.5, 41 against Chrome's 0 six
times). Chrome answers instantly because it renders the moved view from COARSE
mips it already holds and refines afterwards -- multiscale slice rendering.

This codebase implements that as NGL_NATIVE_PANE_MODE=progressive, and the
pane-mode campaign already measured it: a blurry current pane cost 11pp against
a sharp stale one. But `progressive` only ever tested the extreme, max_px=256 --
a 16x voxel reduction. The middle of the curve was never probed.

So: for each max_px, how long does the fetch take, and how close is the
resulting pane to Chrome? If some intermediate lands near Chrome's latency
while staying close to the 0.89 the shipping resolution achieves, there is a
setting that is both current and sharp, and the 11pp result does not apply to
it. If sharpness falls off a cliff with latency, the trade is real and atomic
stays the right default.

    uv run --no-sync python native/probe_mip_tradeoff.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 6
"""

from __future__ import annotations

import argparse
import json
import os
import time

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


def sharpness(g) -> float:
    gy, gx = np.gradient(np.asarray(g, np.float64))
    return float(np.hypot(gy, gx).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    from PIL import Image

    from ngllib.simulator import pane2d
    from ngllib.simulator.em import EMTiles, Source

    mips = [512, 1024, 1536, 2048, 3072]
    lat: dict[int, list] = {m: [] for m in mips}
    ssim: dict[int, list] = {m: [] for m in mips}
    sharp: dict[int, list] = {m: [] for m in mips}
    chrome_sharp = []

    records = [json.loads(line) for line in
               open(os.path.join(args.pairs_dir, "states.jsonl"))][:args.limit]
    for rec in records:
        st = rec["requested_state"]
        fa = os.path.join(args.pairs_dir, "frames", f"{rec['idx']:04d}_a.png")
        if not os.path.exists(fa):
            continue
        ref = np.asarray(Image.open(fa))[pane2d.TOOLBAR:, :pane2d.PANE, :3]
        ref = ref.mean(axis=2)
        chrome_sharp.append(sharpness(ref))

        xs = float(st["crossSectionScale"])
        ext = pane2d.pane_extents_nm(xs)
        pos = np.asarray(st["position"], np.float64) * pane2d.VOXEL_NM
        shifted = pane2d.shifted_fetch_center_nm(pos, ext)

        line = [f"[{rec['idx']:04d}]"]
        for m in mips:
            em = EMTiles(Source.calibrated(args.cache_dir))      # cold, as a move would be
            t0 = time.monotonic()
            try:
                tile = em.tile(shifted, ext[0], ext[1], m, True)
            except Exception as e:  # noqa: BLE001
                print(f"[{rec['idx']:04d}] max_px {m} failed: {e}", flush=True)
                continue
            dt = time.monotonic() - t0
            raster = pane2d.resample_em(tile)
            lat[m].append(dt)
            ssim[m].append(block_ssim(raster, ref))
            sharp[m].append(sharpness(raster))
            line.append(f"{m}:{dt:4.2f}s/{ssim[m][-1]:.3f}")
        print("  ".join(line), flush=True)

    if not chrome_sharp:
        print("no usable states")
        return 1
    cs = float(np.median(chrome_sharp))
    print("\n============== mip latency / fidelity trade ==============")
    print(f"{'max_px':>8}{'fetch s':>10}{'vs Chrome':>12}{'sharpness':>12}"
          f"{'vs Chrome':>12}")
    for m in mips:
        if not lat[m]:
            continue
        print(f"{m:>8}{np.median(lat[m]):>10.2f}{np.median(ssim[m]):>12.4f}"
              f"{np.median(sharp[m]):>12.2f}{np.median(sharp[m]) / cs:>12.2f}")
    print(f"{'chrome':>8}{'-':>10}{'-':>12}{cs:>12.2f}{1.0:>12.2f}")
    print("\nThe shipping setting is 1024. `progressive` used 256 and cost 11pp "
          "in the pane-mode campaign. If an intermediate holds most of 1024's "
          "block_ssim at a fraction of its latency, it is a current-AND-sharp "
          "option the campaign never tested; if fidelity tracks latency all the "
          "way down, the trade is real and atomic stays right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
