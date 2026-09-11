"""Does the simulator render NG's SHOW_ALL_SEGMENTS pane at all?

When `visibleSegments` is empty, Neuroglancer colours the WHOLE slice
(segmentation_renderlayer.ts sets SHOW_ALL_SEGMENTS and the shader forces
has=true everywhere). Chrome measures ~95% of the 2D pane tinted; in the parity
run the simulator's pane stayed at its pre-click value on every deselect.

That has two possible causes and the parity probe cannot separate them: the
render path is broken, or it works and the streamed pane simply had not caught
up within the settle steps. This composes the pane DIRECTLY from the fetch
worker's parts -- no env, no streaming, no staleness -- so the answer is
unambiguous. It now also checks the point of the parts split: one fetch, two
selections, so a selection change costs no fetch at all.

    uv run --no-sync python native/probe_show_all.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 3 \
        --out-dir /scratch/kp0374/native_spike/show_all
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def tint_frac(canvas, toolbar: int) -> float:
    a = np.asarray(canvas, dtype=np.float32)[toolbar:, :, :3]
    return float((a.std(axis=2) > 6).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    from PIL import Image

    from ngllib.simulator import pane2d
    from ngllib.simulator.em import EMTiles, Source, unpack_ids, worker_pane_parts

    records = [json.loads(line) for line in
               open(os.path.join(args.pairs_dir, "states.jsonl"))][:args.limit]
    em = EMTiles(Source.calibrated(args.cache_dir))
    ok = 0
    for rec in records:
        st = rec["requested_state"]
        pos, xs = list(st["position"]), float(st["crossSectionScale"])
        rid = str(st["segments"][0])

        # the id tile itself: how many distinct segments does it even see?
        pos_nm = np.asarray(pos, np.float64) * pane2d.VOXEL_NM
        ext = pane2d.pane_extents_nm(xs)
        ids = em.label_ids(pane2d.shifted_fetch_center_nm(pos_nm, ext),
                           ext[0], ext[1], (pane2d.PANE, pane2d.PANE_H))
        n_ids = 0 if ids is None else int(np.unique(ids).size)

        em_gray, ids_packed, _ = worker_pane_parts(Source.calibrated(args.cache_dir), pos, xs)
        tile_ids = unpack_ids(ids_packed)
        one = pane2d.compose_left_parts(em_gray, tile_ids, (rid,))
        allc = pane2d.compose_left_parts(em_gray, tile_ids, ())
        t1 = tint_frac(one, pane2d.TOOLBAR)
        ta = tint_frac(allc, pane2d.TOOLBAR)
        ok += ta > 0.5
        print(f"[{rec['idx']:04d}] id-tile {'None' if ids is None else 'ok'} "
              f"({n_ids} distinct ids)  tint: one-selected {t1:.4f}  "
              f"none-selected {ta:.4f}", flush=True)
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            Image.fromarray(allc).save(
                os.path.join(args.out_dir, f"{rec['idx']:04d}_show_all.png"))

    print("\n================ SHOW_ALL render ================")
    print(f"states with a colourized pane: {ok}/{len(records)}")
    print("Chrome measures ~0.95 tinted with nothing selected. A value near the "
          "one-selected number means the render path is broken; ~0.95 means it "
          "works and the parity run was measuring a pane that had not streamed "
          "in yet.")
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
