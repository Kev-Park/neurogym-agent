"""Validate EMTiles.segment_at against states whose selected segment is known.

Self-checking: every collected state was built by placing the viewer ON a
skeleton node of `segments[0]`, so a query at the pane CENTRE should return
that same root id. Anything else means the point query disagrees with what the
overlay draws -- which would make double-click select the wrong neuron.

Also samples off-centre points to confirm the query returns *other* ids and
background (None) rather than echoing the centre value.

    uv run --no-sync python native/probe_segment_at.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 12
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    from ngllib.native import pane2d
    from ngllib.native.em import EMTiles

    em = EMTiles(args.cache_dir)
    records = [json.loads(l) for l in
               open(os.path.join(args.pairs_dir, "states.jsonl"))][:args.limit]

    hit = miss = bg = 0
    others = 0
    for rec in records:
        st = rec["requested_state"]
        want = int(st["segments"][0])
        pos = np.asarray(st["position"], np.float64) * pane2d.VOXEL_NM
        ex, ey = pane2d.pane_extents_nm(float(st["crossSectionScale"]))

        got = em.segment_at(pos)
        tag = "HIT " if got == want else ("BG  " if got is None else "MISS")
        if got == want:
            hit += 1
        elif got is None:
            bg += 1
        else:
            miss += 1

        # a point a quarter-pane away should usually be a DIFFERENT segment or
        # background -- if it always echoes the centre, the query is broken
        off = pos + np.array([ex / 4.0, ey / 4.0, 0.0])
        got_off = em.segment_at(off)
        if got_off not in (None, want):
            others += 1

        print(f"[{rec['idx']:04d}] {tag} want={want} got={got} "
              f"off-centre={got_off}", flush=True)

    n = hit + miss + bg
    print("\n================ segment_at validation ================")
    print(f"states            : {n}")
    print(f"centre == segments[0] : {hit}/{n}")
    print(f"centre == background  : {bg}/{n}")
    print(f"centre == other id    : {miss}/{n}")
    print(f"off-centre found a different segment: {others}/{n}")
    print("\nExpect centre HIT on nearly every state: the viewer was placed on a "
          "skeleton node of that neuron. Background or a different id at the "
          "centre means the mip/resolution or the coordinate mapping is wrong.")
    return 0 if hit >= max(1, int(0.8 * n)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
