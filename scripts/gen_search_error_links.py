"""Generate pre-edit "find the segmentation error" NG link pairs for manual checks.

Per SEARCH_ENV_CONSIDERATIONS.md (per-operation before-state). For each sampled
proofreading operation:

  * START url  — the segmentation state BEFORE the fix: a graphene layer whose
    segments are the operation's `before_root_ids` (a split's one wrongly-merged
    object, or a merge's two separate pieces). Loading those historical roots IS
    the un-fixed geometry (they resolve + mesh forever); a layer `timestamp` just
    before the op is added as belt-and-suspenders. Viewer parked at the object
    centroid, zoomed out to survey — the error is somewhere in view, not marked.
  * EVAL url   — identical geometry, viewer parked EXACTLY at the operation
    coordinate (the proofreader's source/sink point), zoomed in: where the error
    actually is, for scoring.

Data is verified server-side (roots resolve, coords, meshes). BROWSER CAVEAT:
graphene + `flywire_public` needs a FlyWire/CAVE (middleauth) login in the Chrome
you open these in — fine for a real-browser CUA test. Host prefix is swappable
(demo host below; ngl.flywire.ai / spelunker also render the same #! state).

    uv run --no-sync python scripts/gen_search_error_links.py --n 5 --seed 20260917
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from ngllib.dataset import state_to_url

# ChunkedGraph op coords are 16x16x40 nm voxels; x4 in x/y for the 4x4x40 grid
# NG position lives in (harvest_edit_history.py). Mesh verts are in nm.
EDIT_TO_SKEL = np.array([4.0, 4.0, 1.0])
VOXEL_NM = np.array([4.0, 4.0, 40.0])
L2_MIN, L2_MAX = 20, 3000          # skip stubs and merge-monsters
MESH_VERT_CAP = 600_000            # skip absurdly large meshes for a quick check


def ng_graphene(seg_src: str) -> str:
    # Browser NG host is prodv1 (prod is a server-only alias). ngl.flywire.ai
    # authenticates graphene via the logged-in session, so DO NOT add the
    # middleauth+ prefix — that path returns empty in-browser ("HTTP error 0").
    return seg_src.replace("prod.flywire-daf.com", "prodv1.flywire-daf.com")


def build_state(em_src, seg_src, seg_ids, pos_vox, zoom2d, zoom3d):
    # ngl.flywire.ai runs the OLD neuroglancer state format (verified from a
    # normalized state it produced, 2026-09-17): navigation.pose.position.
    # voxelCoordinates (+ voxelSize), NOT the modern top-level dimensions/
    # position — those are silently ignored. Graphene layer type must be
    # segmentation_with_graph.
    return {
        "layers": [
            {"source": em_src, "type": "image", "name": "EM"},
            {"source": seg_src, "type": "segmentation_with_graph",
             "segments": [str(s) for s in seg_ids],
             "name": "flywire_public (pre-edit)"},
        ],
        "navigation": {
            "pose": {"position": {
                "voxelSize": [4, 4, 40],
                "voxelCoordinates": [float(v) for v in pos_vox]}},
            "zoomFactor": float(zoom2d),
        },
        "perspectiveZoom": float(zoom3d),
        "showDefaultAnnotations": False,
        "layout": "xy-3d",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--edits", default="edit_sites_pilot.parquet")
    ap.add_argument("--datastack", default="flywire_fafb_public")
    ap.add_argument("--merges", type=int, default=2, help="Aim for this many merge ops.")
    args = ap.parse_args()

    from caveclient import CAVEclient
    from cloudvolume import CloudVolume

    client = CAVEclient(args.datastack)
    seg_src = client.info.segmentation_source()
    ng_seg = ng_graphene(seg_src)
    ng_host = client.info.viewer_site()           # authoritative FlyWire NG host
    em_src = client.info.image_source()
    cv = CloudVolume(seg_src, use_https=True, progress=False)
    print(f"# datastack={args.datastack}\n# host={ng_host}\n"
          f"# graphene(NG)={ng_seg}\n# em={em_src}\n", flush=True)

    roots = [str(r) for r in
             pq.read_table(args.edits, columns=["root_id"]).column("root_id").to_pylist()]
    rng = np.random.default_rng(args.seed)
    roots = list(dict.fromkeys(roots))
    rng.shuffle(roots)

    def sized_ok(root_id) -> int | None:
        try:
            n = len(client.chunkedgraph.get_leaves(int(root_id), stop_layer=2))
        except Exception:
            return None
        return n if L2_MIN <= n <= L2_MAX else None

    pairs, want_merges = [], args.merges
    for rid in roots:
        if len(pairs) >= args.n:
            break
        try:
            log = client.chunkedgraph.get_tabular_change_log(int(rid))[int(rid)]
        except Exception:
            continue
        order = list(range(len(log)))
        rng.shuffle(order)
        for idx in order:
            if len(pairs) >= args.n:
                break
            row = log.iloc[idx]
            is_merge = bool(row["is_merge"])
            have_merges = sum(1 for p in pairs if p["is_merge"])
            # Steer the split/merge mix toward the requested count.
            if is_merge and have_merges >= want_merges:
                continue
            if not is_merge and (len(pairs) - have_merges) >= (args.n - want_merges):
                continue
            before = [int(x) for x in np.atleast_1d(row["before_root_ids"])]
            after = [int(x) for x in np.atleast_1d(row["after_root_ids"])]
            # Geometry must match the task: a real merge joins >=2 pieces (load
            # them all); a real split produces >=2 pieces from one object.
            if is_merge and len(before) < 2:
                continue
            if not is_merge and len(after) < 2:
                continue
            sizes = [sized_ok(b) for b in before]
            if any(s is None for s in sizes):
                continue
            op_id = int(row["operation_id"])
            ts = int(row["timestamp"]) // 1000  # change-log ts is ms; NG wants s
            try:
                det = client.chunkedgraph.get_operation_details([op_id])[str(op_id)]
                coords = [c for role in ("source_coords", "sink_coords")
                          for c in (det.get(role) or [])]
                if not coords:
                    continue
                err_vox = np.mean(coords, axis=0) * EDIT_TO_SKEL
            except Exception:
                continue
            try:
                verts = []
                for b in before:
                    m = cv.mesh.get(b)[b]
                    verts.append(np.asarray(m.vertices))
                verts = np.concatenate(verts, axis=0)
            except Exception:
                continue
            if len(verts) > MESH_VERT_CAP:
                continue
            centroid_vox = verts.mean(axis=0) / VOXEL_NM
            bbox_vox = (verts.max(0) - verts.min(0)) / VOXEL_NM
            proj = float(np.clip(np.linalg.norm(bbox_vox) * 0.45, 12000, 60000))
            dist_nm = float(np.linalg.norm((centroid_vox - err_vox) * VOXEL_NM))

            ts_layer = ts - 1
            start = build_state(em_src, ng_seg, before, centroid_vox, proj, 4.0, ts_layer)
            evals = build_state(em_src, ng_seg, before, err_vox, 3000.0, 0.8, ts_layer)
            pairs.append({
                "op_id": op_id, "is_merge": is_merge, "ts": ts,
                "before": before, "after": after, "sizes": sizes,
                "n_coords": len(coords), "err_vox": err_vox,
                "centroid_dist_nm": dist_nm, "n_verts": len(verts),
                "start_url": state_to_url(ng_host, start),
                "eval_url": state_to_url(ng_host, evals),
            })
            print(f"[{len(pairs)}/{args.n}] op {op_id} "
                  f"{'MERGE' if is_merge else 'SPLIT'} "
                  f"{time.strftime('%Y-%m-%d', time.gmtime(ts))} "
                  f"before={before} L2={sizes} verts={len(verts)} "
                  f"centroid->error {dist_nm/1000:.1f}um", flush=True)
            time.sleep(0.3)

    print("\n" + "=" * 70, flush=True)
    for i, p in enumerate(pairs, 1):
        kind = "MERGE (join the pieces)" if p["is_merge"] else "SPLIT (cut the wrong merger)"
        print(f"\n## Pair {i} — {kind}", flush=True)
        print(f"   op {p['op_id']}  date {time.strftime('%Y-%m-%d', time.gmtime(p['ts']))}"
              f"  before_roots={p['before']} (L2 {p['sizes']})"
              f"  error points={p['n_coords']}", flush=True)
        print(f"   error voxel (4x4x40) = [{p['err_vox'][0]:.0f}, "
              f"{p['err_vox'][1]:.0f}, {p['err_vox'][2]:.0f}]  "
              f"survey-start is ~{p['centroid_dist_nm']/1000:.1f}um away", flush=True)
        print(f"   START: {p['start_url']}", flush=True)
        print(f"   EVAL : {p['eval_url']}", flush=True)
    print(f"\n[done] {len(pairs)} pairs (seed={args.seed})", flush=True)
    return 0 if pairs else 1


if __name__ == "__main__":
    sys.exit(main())
