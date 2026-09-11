"""How long does ONE mesh fetch actually take, with nothing queued behind it?

The 200-step dynamics run put the simulator's 3D response at 171.5 steps
against Chrome's 21.5 -- but the two do not spend the same wall time per step
(0.007 s vs 0.031 s), so in SECONDS that is ~1.20 s against ~0.67 s. Before
adding a dedicated mesh queue to remove head-of-line blocking behind tile
fetches, this checks whether there is any blocking to remove: time
`worker_mesh` on its own, cold and warm.

If an isolated cold fetch already costs ~1.2 s, the download and Draco decode
ARE the latency and a separate pool buys nothing. If it costs far less, the
shared pool is the problem and a dedicated queue is worth the extra process.

    uv run --no-sync python native/probe_mesh_latency.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 6
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    import numpy as np

    from ngllib.simulator.em import Source, worker_mesh

    records = [json.loads(line) for line in
               open(os.path.join(args.pairs_dir, "states.jsonl"))][:args.limit]
    seen, cold, warm, sizes = set(), [], [], []
    for rec in records:
        rid = str(rec["requested_state"]["segments"][0])
        if rid in seen:
            continue
        seen.add(rid)
        t0 = time.monotonic()
        try:
            v, _vn, f = worker_mesh(Source.calibrated(args.cache_dir), rid)
        except Exception as e:  # noqa: BLE001
            print(f"[{rid}] failed: {e}", flush=True)
            continue
        dt = time.monotonic() - t0
        t1 = time.monotonic()
        worker_mesh(Source.calibrated(args.cache_dir), rid)          # second call: cache warm
        dt2 = time.monotonic() - t1
        cold.append(dt); warm.append(dt2)
        sizes.append(len(v))
        print(f"[{rid}] cold {dt:6.2f}s  warm {dt2:6.2f}s  "
              f"{len(v):,} verts / {len(f):,} faces", flush=True)

    if not cold:
        print("no meshes fetched")
        return 1
    print("\n============== isolated mesh fetch ==============")
    print(f"meshes            : {len(cold)}")
    print(f"cold  median      : {np.median(cold):.2f}s  "
          f"(min {min(cold):.2f} max {max(cold):.2f})")
    print(f"warm  median      : {np.median(warm):.2f}s")
    print(f"vertices median   : {int(np.median(sizes)):,}")
    print("\nIn the 200-step run the simulator's 3D pane responded after "
          "~1.20 s and Chrome's after ~0.67 s. A cold fetch near 1.2 s here "
          "means the download IS the latency and a dedicated mesh pool would "
          "not help; well under that means the fetch is queueing behind tile "
          "work in the shared pool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
