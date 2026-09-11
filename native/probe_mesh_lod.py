"""Does the mesh source have coarse LODs, and are they fast enough to matter?

Under production stepping (883638, --obs dino) the 3D pane's response to a
select is 71.5 steps for the simulator against Chrome's 2.0 -- and unlike the
raw-obs run, that gap is NOT explained by cheaper simulator steps. In wall
time it is ~1.0 s against ~0.08 s, and Chrome's slowest genuine misses were
still only 8-10 steps (~0.35 s) where ours were 59-126 (0.8-1.8 s).

The likely reason is what Neuroglancer actually does and we do not: the
precomputed multi-resolution mesh format stores several levels of detail, and
NG fetches the coarsest one adequate for the current view, refining later. We
fetch the FULL mesh every time -- measured at 73k-340k vertices -- and then
compute vertex normals in Python.

So, before building progressive mesh loading, check the two things it depends
on: that this source really is multi-LOD, and that a coarse level lands fast
enough to reach Chrome's ~0.1-0.35 s.

    uv run --no-sync python native/probe_mesh_lod.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 4
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

SEG_URL = "precomputed://gs://flywire_v141_m783"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--max-lod", type=int, default=4)
    args = ap.parse_args()

    import numpy as np
    from cloudvolume import CloudVolume

    vol = CloudVolume(SEG_URL, use_https=True, cache=False, progress=False)
    info = getattr(vol.mesh, "meta", None)
    print("mesh class :", type(vol.mesh).__name__)
    try:
        mi = vol.mesh.meta.info
        print("mesh info  :", json.dumps(mi, indent=2)[:600])
    except Exception as e:  # noqa: BLE001
        print("mesh info unavailable:", e)
    print("has lod arg:", "lod" in getattr(vol.mesh.get, "__doc__", "" ) or "")
    print()

    records = [json.loads(line) for line in
               open(os.path.join(args.pairs_dir, "states.jsonl"))][:args.limit]
    seen = set()
    rows = []
    for rec in records:
        rid = int(rec["requested_state"]["segments"][0])
        if rid in seen:
            continue
        seen.add(rid)
        for lod in range(args.max_lod + 1):
            t0 = time.monotonic()
            try:
                m = vol.mesh.get(rid, lod=lod)
            except TypeError:
                print("  mesh.get takes no lod argument on this source")
                lod = None
                break
            except Exception as e:  # noqa: BLE001
                print(f"[{rid}] lod {lod}: {type(e).__name__}: "
                      f"{str(e)[:80]}", flush=True)
                continue
            dt = time.monotonic() - t0
            mesh = m[rid] if isinstance(m, dict) else m
            n = len(mesh.vertices)
            rows.append((rid, lod, dt, n))
            print(f"[{rid}] lod {lod}: {dt:6.2f}s  {n:,} verts", flush=True)
        if lod is None:
            break

    if not rows:
        print("\nNo LOD-addressable fetches: this source exposes a single "
              "level through CloudVolume, so progressive mesh loading would "
              "need the sharded multires reader directly, or a decimation "
              "step of our own.")
        return 1

    print("\n============== mesh LOD latency ==============")
    by_lod: dict[int, list] = {}
    for _rid, lod, dt, n in rows:
        by_lod.setdefault(lod, []).append((dt, n))
    for lod in sorted(by_lod):
        d = [x[0] for x in by_lod[lod]]
        v = [x[1] for x in by_lod[lod]]
        print(f"lod {lod}: {np.median(d):6.2f}s median   "
              f"{int(np.median(v)):>9,} verts median")
    # The environment does not call vol.mesh.get directly: MeshStore.get takes
    # a lod REQUEST and walks DOWN until one decodes, because the available
    # range varies per segment. When the requested level is absent that costs a
    # failed round-trip first, so the shipping path can be slower than the
    # per-level numbers above suggest. Time what actually runs.
    from ngllib.simulator.em import MeshStore, Source

    print("\n--- MeshStore.get (the shipping path, with fallback) ---")
    for req in (0, 1, 2):
        ds = []
        for rid in list(seen):
            store = MeshStore(Source.calibrated())      # fresh store: no LRU hit
            t0 = time.monotonic()
            try:
                store.get(str(rid), req)
            except Exception as e:  # noqa: BLE001
                print(f"  lod<={req} {rid}: {type(e).__name__}", flush=True)
                continue
            ds.append(time.monotonic() - t0)
        if ds:
            print(f"  request lod<={req}: {np.median(ds):5.2f}s median "
                  f"(n={len(ds)})", flush=True)

    print("\nChrome's genuine mesh misses landed in ~0.08-0.35 s. A coarse LOD "
          "at or under that makes progressive loading worth building: show the "
          "coarse level on arrival, refine when the fine one lands, which is "
          "what NG does. If every level costs about the same, the format is "
          "not the bottleneck and the decode is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
