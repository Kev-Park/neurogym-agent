"""Where does worker_pane_parts actually spend its time?

Dropping the EM tile to a coarser mip (1024 -> 768, half the voxels) did not
move the 2D response step at all: 31 steps against the previous run's 5, i.e.
inside the noise. So fetch RESOLUTION is not the bottleneck, and something else
dominates.

worker_pane_parts does THREE volume reads back to back -- the EM tile at the
registration-shifted centre, the segmentation ids at the same centre, and the
plane tile at the unshifted centre -- and the client cannot show the 2D pane
until the whole bundle lands. If they are roughly equal and sequential, the
pane waits about three times longer than it needs to, and the fix is to run
them in parallel and adopt each as it arrives rather than to fetch less data.

    uv run --no-sync python native/probe_parts_breakdown.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 5
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
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    from ngllib.simulator import pane2d
    from ngllib.simulator.em import EMTiles, Source

    records = [json.loads(line) for line in
               open(os.path.join(args.pairs_dir, "states.jsonl"))][:args.limit]
    t_em, t_ids, t_plane = [], [], []
    for rec in records:
        st = rec["requested_state"]
        xs = float(st["crossSectionScale"])
        ext = pane2d.pane_extents_nm(xs)
        pos_nm = np.asarray(st["position"], np.float64) * pane2d.VOXEL_NM
        shifted = pane2d.shifted_fetch_center_nm(pos_nm, ext)

        em = EMTiles(Source.calibrated(args.cache_dir))          # cold, as a move would be
        t0 = time.monotonic()
        try:
            em.tile(shifted, ext[0], ext[1], 1024, True)
        except Exception as e:  # noqa: BLE001
            print(f"[{rec['idx']:04d}] em failed: {e}", flush=True)
            continue
        d_em = time.monotonic() - t0

        t1 = time.monotonic()
        em.label_ids(shifted, ext[0], ext[1], (pane2d.PANE, pane2d.PANE_H))
        d_ids = time.monotonic() - t1

        t2 = time.monotonic()
        em.tile(pos_nm, ext[0], ext[1], 1024, False)
        d_plane = time.monotonic() - t2

        t_em.append(d_em)
        t_ids.append(d_ids)
        t_plane.append(d_plane)
        print(f"[{rec['idx']:04d}] em {d_em:5.2f}s  ids {d_ids:5.2f}s  "
              f"plane {d_plane:5.2f}s", flush=True)

    if not t_em:
        print("no usable states")
        return 1
    me = float(np.median(t_em))
    mi = float(np.median(t_ids))
    mp = float(np.median(t_plane))
    print("\n============== worker_pane_parts breakdown ==============")
    print(f"EM tile (shifted centre)  : {me:.2f}s")
    print(f"segmentation ids          : {mi:.2f}s")
    print(f"plane tile (pos centre)   : {mp:.2f}s")
    print(f"sequential total          : {me + mi + mp:.2f}s")
    print(f"if run in PARALLEL        : {max(me, mi, mp):.2f}s")
    print(f"2D pane needs only em+ids : {max(me, mi):.2f}s parallel vs "
          f"{me + mi:.2f}s sequential")
    print("\nThe 2D pane cannot show a move until the WHOLE bundle lands, "
          "including the plane tile it does not use. If the three reads are "
          "comparable, running them in parallel and adopting each as it "
          "arrives cuts the wait substantially -- without fetching one voxel "
          "less, so not a single pixel changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
