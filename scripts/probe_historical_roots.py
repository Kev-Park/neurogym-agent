"""Feasibility probe: can PRE-EDIT (historical) roots be resolved + fetched?

The pre-edit-state task idea: load the segmentation object as it existed
BEFORE a proofreading operation (the erroneous state), and use that
operation's coordinates as the success flag. Supervoxels are immutable and
edits only rewire the agglomeration, so every historical root_id remains a
queryable object — IF our access path serves it. This probe verifies, for a
few operations from the pilot harvest:

  1. before_root_ids are recoverable from the change log;
  2. the historical root still resolves server-side (L2 leaves);
  3. spawn-point geometry exists for it (l2cache rep_coord_nm);
  4. mesh fetch works for it (CloudVolume graphene, if installed) — what the
     native renderer / Neuroglancer would display.

Read-only, ~1 min, login-node safe.

    uv run --no-sync python scripts/probe_historical_roots.py \
        --edits edit_sites_pilot.parquet --n 3
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edits", default="edit_sites_pilot.parquet")
    ap.add_argument("--datastack", default="flywire_fafb_public")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    from caveclient import CAVEclient

    client = CAVEclient(args.datastack)
    tbl = pq.read_table(args.edits, columns=["root_id", "operation_id"])
    df = tbl.to_pylist()
    # Pick the first N distinct roots from the pilot bank.
    seen, picks = set(), []
    for r in df:
        if r["root_id"] not in seen:
            seen.add(r["root_id"])
            picks.append(r["root_id"])
        if len(picks) >= args.n:
            break

    ok = {"before_ids": 0, "resolves": 0, "l2cache": 0, "mesh": 0}
    tried_mesh = False
    for rid in picks:
        print(f"\n== root {rid} ==", flush=True)
        log = client.chunkedgraph.get_tabular_change_log(int(rid))[int(rid)]
        cols = list(log.columns)
        print(f"  change-log columns: {cols}", flush=True)
        # Prefer a mid-history split (splits are the find-the-error case).
        cands = log[~log["is_merge"]] if "is_merge" in cols else log
        row = (cands.iloc[len(cands) // 2] if len(cands) else log.iloc[0])
        before = row.get("before_root_ids")
        if before is None or (hasattr(before, "__len__") and len(before) == 0):
            print("  no before_root_ids on this row; columns above tell us "
                  "which field carries it", flush=True)
            continue
        old_root = int(np.atleast_1d(before)[0])
        ok["before_ids"] += 1
        print(f"  op {row['operation_id']}: before_root={old_root}", flush=True)

        try:
            l2 = client.chunkedgraph.get_leaves(old_root, stop_layer=2)
            ok["resolves"] += 1
            print(f"  historical root RESOLVES: {len(l2)} L2 nodes", flush=True)
        except Exception as e:  # noqa: BLE001 — the failure mode is the finding
            print(f"  get_leaves FAILED: {type(e).__name__}: {e}"[:160], flush=True)
            continue

        try:
            data = client.l2cache.get_l2data(
                [int(x) for x in l2[:5]], attributes=["rep_coord_nm"])
            got = sum(1 for v in data.values() if v.get("rep_coord_nm"))
            ok["l2cache"] += 1 if got else 0
            print(f"  l2cache rep_coord_nm: {got}/5 sampled L2 ids -> "
                  f"spawn points {'OK' if got else 'MISSING'}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  l2cache FAILED: {type(e).__name__}: {e}"[:160], flush=True)

        if not tried_mesh:
            tried_mesh = True
            try:
                from cloudvolume import CloudVolume

                seg_src = client.info.segmentation_source()
                cv = CloudVolume(seg_src, use_https=True, progress=False)
                mesh = cv.mesh.get(old_root)[old_root]
                ok["mesh"] += 1
                print(f"  MESH FETCH OK: {len(mesh.vertices)} vertices "
                      f"(source {seg_src[:60]}...)", flush=True)
            except ImportError:
                print("  cloud-volume not installed in this venv — mesh check "
                      "skipped (run from the native-renderer env)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  mesh fetch FAILED: {type(e).__name__}: {e}"[:200],
                      flush=True)

    print(f"\n[probe] before_ids {ok['before_ids']}/{len(picks)} | "
          f"resolves {ok['resolves']}/{len(picks)} | "
          f"l2cache {ok['l2cache']}/{len(picks)} | "
          f"mesh {'OK' if ok['mesh'] else 'not verified'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
