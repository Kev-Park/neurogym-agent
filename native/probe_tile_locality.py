"""Is a NEARBY tile fetch already cheap, or does every move pay full price?

The per-verb sweep found the 2D pane takes 61+ steps to reflect a
move-to-mouse-position where Chrome takes 0, and the assumed cause was that
Chrome re-renders from cached chunks while we refetch. But EMTiles holds a
chunk LRU of its own, and a 200 px move overlaps the previous view heavily, so
most of the chunks a nearby fetch needs may already be in RAM.

If a nearby refetch costs ~50 ms, the 61-step lag is a PIPELINE artefact -- one
fetch in flight, adopted only once complete, queued behind pool work -- and the
fix is scheduling, not caching. If it costs ~700 ms like a cold one, the data
really is being refetched and only a cache can help.

    uv run --no-sync python native/probe_tile_locality.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 4
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    from ngllib.native import pane2d
    from ngllib.native.em import EMTiles

    records = [json.loads(line) for line in
               open(os.path.join(args.pairs_dir, "states.jsonl"))][:args.limit]

    fracs = [0.0, 0.05, 0.15, 0.25, 0.50, 1.00]
    acc: dict[float, list] = {f: [] for f in fracs}
    cold = []
    for rec in records:
        st = rec["requested_state"]
        em = EMTiles(args.cache_dir)        # fresh: cold chunk LRU
        pos = np.asarray(st["position"], np.float64) * pane2d.VOXEL_NM
        xs = float(st["crossSectionScale"])
        ext = pane2d.pane_extents_nm(xs)

        t0 = time.monotonic()
        try:
            em.tile(pos, ext[0], ext[1], 1024, True)
        except Exception as e:  # noqa: BLE001
            print(f"[{rec['idx']:04d}] cold fetch failed: {e}", flush=True)
            continue
        dt_cold = time.monotonic() - t0
        cold.append(dt_cold)

        line = [f"[{rec['idx']:04d}] cold {dt_cold:5.2f}s"]
        for f in fracs:
            p = pos + np.array([ext[0] * f, 0.0, 0.0])
            t1 = time.monotonic()
            try:
                em.tile(p, ext[0], ext[1], 1024, True)
            except Exception:  # noqa: BLE001
                continue
            dt = time.monotonic() - t1
            acc[f].append(dt)
            line.append(f"+{int(f * 100):>3}% {dt:5.2f}s")
        print("  ".join(line), flush=True)

    if not cold:
        print("no usable states")
        return 1
    print("\n============== tile fetch vs distance moved ==============")
    print(f"cold (empty chunk LRU) : {np.median(cold):.2f}s")
    for f in fracs:
        if acc[f]:
            print(f"move {int(f * 100):>3}% of pane   : "
                  f"{np.median(acc[f]):.2f}s")
    print("\nA move of 15-25% costing far less than the cold fetch means the "
          "chunks are already resident and the 61-step lag is the PIPELINE, "
          "not the data -- in which case the fix is to stop waiting for a "
          "whole fetch before showing anything, not to cache regions and pay "
          "0.12 of block_ssim for the privilege.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
