"""Pilot harvest of FlyWire proofreading edit history (design-question probe).

Pulls the ChunkedGraph change log for a set of root_ids (default: our frozen
eval pool) via CAVEclient and writes one row PER EDIT COORDINATE to a parquet:

    (root_id, operation_id, is_merge, timestamp, role, x, y, z)

role = source|sink — the points the proofreader placed to specify the
operation, in segmentation voxel space (FAFB 4x4x40nm — the same grid as
segment_positions.parquet; the summary validates this against skeleton
bounding boxes). These coordinates are the candidate "success states" for
the find-the-edit-site task.

Also prints the design-question summary: edits/neuron distribution,
split-vs-merge mix, coordinate-space sanity, edit-site distance to the
nearest skeleton node (navigability on our spawn graph), and z-position of
edit sites within each neuron's extent.

Per-root REST calls with throttling + incremental flush — this is the
PILOT path; the full corpus should come from an internal export of the
operation log (Seung Lab runs the ChunkedGraph).

    uv run --no-sync python scripts/harvest_edit_history.py \
        --root-ids-from eval_d0_v1.parquet --limit 200 \
        --skeleton segment_positions.parquet \
        --output edit_sites_pilot.parquet
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VOXEL_NM = np.array([4.0, 4.0, 40.0])  # FAFB anisotropy for nm distances


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-ids-from", default="eval_d0_v1.parquet")
    ap.add_argument("--skeleton", default="segment_positions.parquet")
    ap.add_argument("--datastack", default="flywire_fafb_public")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=0.7,
                    help="Seconds between roots (politeness to the API).")
    ap.add_argument("--output", default="edit_sites_pilot.parquet")
    args = ap.parse_args()

    from caveclient import CAVEclient

    client = CAVEclient(args.datastack)
    print(f"[harvest] datastack={args.datastack} "
          f"server={client.server_address}", flush=True)

    ids = pq.read_table(args.root_ids_from, columns=["root_id"])
    root_ids = [str(r) for r in ids.column("root_id").to_pylist()][: args.limit]
    print(f"[harvest] {len(root_ids)} root_ids from {args.root_ids_from}",
          flush=True)

    rows: list[dict] = []
    failed: list[tuple[str, str]] = []
    op_meta: dict = {}
    for i, rid in enumerate(root_ids):
        try:
            log = client.chunkedgraph.get_tabular_change_log(int(rid))[int(rid)]
            if len(log) == 0:
                continue
            op_ids = [int(x) for x in log["operation_id"]]
            meta = {int(r["operation_id"]): r for _, r in log.iterrows()}
            details = client.chunkedgraph.get_operation_details(op_ids)
            for op_id_s, d in details.items():
                op_id = int(op_id_s)
                m = meta.get(op_id, {})
                is_merge = bool(m.get("is_merge", "added_edges" in d))
                ts = int(m.get("timestamp", 0))
                for role in ("source", "sink"):
                    for c in d.get(f"{role}_coords", []) or []:
                        rows.append({
                            "root_id": rid, "operation_id": op_id,
                            "is_merge": is_merge, "timestamp": ts,
                            "role": role,
                            "x": float(c[0]), "y": float(c[1]),
                            "z": float(c[2]),
                        })
            op_meta[rid] = len(op_ids)
        except Exception as e:  # noqa: BLE001 — record and continue; the
            # failure MODE (bad id vs auth vs rate limit) is itself pilot data
            failed.append((rid, f"{type(e).__name__}: {e}"[:120]))
        if (i + 1) % 10 == 0:
            print(f"[harvest] {i + 1}/{len(root_ids)} roots "
                  f"({len(rows)} points, {len(failed)} failed)", flush=True)
            if rows:
                pq.write_table(pa.Table.from_pylist(rows), args.output)
        time.sleep(args.sleep)

    if rows:
        pq.write_table(pa.Table.from_pylist(rows), args.output)
    print(f"\n[harvest] DONE: {len(rows)} edit points from "
          f"{len(op_meta)} roots with edits; {len(failed)} roots failed",
          flush=True)
    for rid, err in failed[:5]:
        print(f"  failed {rid}: {err}", flush=True)

    if not rows:
        return 1

    # ---- design-question summary ------------------------------------------
    import duckdb

    counts = np.asarray(list(op_meta.values()))
    print("\n== Edits per neuron (roots with >=1 edit) ==", flush=True)
    print(f"  n={len(counts)} median={np.median(counts):.0f} "
          f"q25={np.quantile(counts, .25):.0f} q75={np.quantile(counts, .75):.0f} "
          f"max={counts.max()}", flush=True)
    merges = sum(1 for r in rows if r["is_merge"])
    print(f"  points: {len(rows)} total | merge-ops {merges} "
          f"({100 * merges / len(rows):.0f}%) vs split {len(rows) - merges}",
          flush=True)

    con = duckdb.connect()
    esc = args.skeleton.replace("'", "''")
    print("\n== Coordinate-space sanity + navigability (sample of 12 roots) ==",
          flush=True)
    sample = list(op_meta)[:12]
    inside, dists_nm, zfracs = 0, [], []
    for rid in sample:
        res = con.execute(
            f"SELECT x, y, z FROM read_parquet('{esc}') "
            "WHERE CAST(root_id AS VARCHAR) = ?", [rid]).fetchnumpy()
        skel = np.stack([res["x"], res["y"], res["z"]], axis=1)
        pts = np.asarray([[r["x"], r["y"], r["z"]] for r in rows
                          if r["root_id"] == rid])
        lo, hi = skel.min(0), skel.max(0)
        pad = (hi - lo) * 0.1 + 1
        inside += int(np.all((pts >= lo - pad) & (pts <= hi + pad)))
        d = np.linalg.norm(
            (pts[:, None, :] - skel[None, :, :]) * VOXEL_NM, axis=2).min(1)
        dists_nm.extend(d.tolist())
        span = (hi[2] - lo[2]) or 1
        zfracs.extend(((pts[:, 2] - lo[2]) / span).tolist())
    print(f"  roots whose edit points fall inside skeleton bbox(+10%): "
          f"{inside}/{len(sample)}", flush=True)
    dists_nm = np.asarray(dists_nm)
    print(f"  edit-point -> nearest skeleton node (nm): "
          f"median {np.median(dists_nm):.0f} q90 {np.quantile(dists_nm, .9):.0f} "
          f"max {dists_nm.max():.0f}", flush=True)
    zf = np.asarray(zfracs)
    print(f"  edit-site z within neuron extent: median {np.median(zf):.2f} "
          f"(0=bottom 1=top; uniform ~0.5 means unlike the old z-max task)",
          flush=True)
    print(f"\n[harvest] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
