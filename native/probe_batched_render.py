"""Batched-rendering ceiling probe (experiment 2; no env logic, no RLlib).

ONE process, ONE EGL context renders N views per pipeline step (sequential
draws — no per-env context switching), then ONE DINO forward over all 2N
pane frames (per-runner batched encoding, the experiment-1 winner). With
fetches now ~free/async, render+encode IS the step cost, so this measures
the node-level sps ceiling of a batched single-process design (Phase-2
minus CUDA-GL interop, which would further remove the readbacks).

    uv run --no-sync python native/probe_batched_render.py --views 32
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main() -> int:
    import pyarrow.parquet as pq

    from ngllib.native.colors import segment_color
    from ngllib.native.em import MeshStore
    from ngllib.native.render3d import MeshRenderer
    from ngllib_agent.obs import get_dino_encoder

    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=32)
    ap.add_argument("--meshes", type=int, default=8)
    ap.add_argument("--secs", type=float, default=60.0)
    ap.add_argument("--pool", default="/scratch/kp0374/neurogym-agent/eval_d0_v1.parquet")
    args = ap.parse_args()

    rids = [str(r) for r in
            pq.read_table(args.pool, columns=["root_id"])
            .column("root_id").to_pylist()][:args.meshes]
    rend = MeshRenderer(450, 433)
    store = MeshStore()
    centers = {}
    for rid in rids:
        v, f = store.get(rid)
        rend.load_mesh(rid, v, f)
        centers[rid] = v.mean(axis=0)
        print(f"[br] mesh {rid}: {len(v)} verts", flush=True)
    enc = get_dino_encoder()

    rng = np.random.default_rng(3)
    views = []
    for i in range(args.views):
        rid = rids[i % len(rids)]
        q = rng.normal(size=4)
        views.append({"rid": rid, "pos": centers[rid],
                      "q": q / np.linalg.norm(q), "zoom": 14000 * 4.07})

    def spin(v):
        d = rng.normal(scale=0.05, size=4)
        q = v["q"] + d
        v["q"] = q / np.linalg.norm(q)

    t_render = t_dino = 0.0
    n = 0
    # warmup
    for v in views:
        rend.render(v["rid"], v["pos"], v["q"], v["zoom"],
                    segment_color(int(v["rid"])))
    t_end = time.monotonic() + args.secs
    while time.monotonic() < t_end:
        t0 = time.monotonic()
        frames = []
        for v in views:
            spin(v)
            frames.append(rend.render(v["rid"], v["pos"], v["q"], v["zoom"],
                                      segment_color(int(v["rid"]))))
        t1 = time.monotonic()
        enc.encode(frames + frames)  # both panes per env-step equivalent
        t2 = time.monotonic()
        t_render += t1 - t0
        t_dino += t2 - t1
        n += 1
    total = t_render + t_dino
    sps = args.views * n / total
    print(f"RENDERBATCH N={args.views} sps={sps:.1f} "
          f"render={1000 * t_render / (n * args.views):.2f}ms/view "
          f"dino={1000 * t_dino / n:.1f}ms/step "
          f"({n} pipeline steps)", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
