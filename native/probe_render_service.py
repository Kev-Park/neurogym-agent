"""Per-node render+encode SERVICE probe (experiment 3).

One service process owns ONE GL context + ONE DINO; K client processes run
pure env logic (state arithmetic + click picks) and exchange tiny messages:
  state (pos/quat/zoom/rid, ~100B)  ->  DINO features (2x384 f32, ~3KB)
  pick request (state + pixel)      ->  picked position or None

This composes the two batched designs and removes both their weaknesses:
no frame IPC (experiment 1's service shipped 1.2MB/step), and no serial
render->encode alternation (experiment 2's single process) — clients step
concurrently while the service pipelines render batches into encode calls.
No EM tiles here (amortized/memoized in the real env): this is the
render+encode+IPC pipeline ceiling.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np


def _client(i, secs, rids, centers, req_q, resp_q, out_q, barrier):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "geom", "/scratch/kp0374/wt/neurogym-native/src/ngllib/utils/geom.py")
    geom = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geom)

    rng = np.random.default_rng(i)
    rid = rids[i % len(rids)]
    pos = list(centers[i % len(rids)])
    q = rng.normal(size=4)
    quat = list(q / np.linalg.norm(q))
    ps = 14000.0
    barrier.wait()
    n, t0 = 0, time.monotonic()
    while time.monotonic() - t0 < secs:
        verb = rng.integers(3)
        if verb == 0:  # click: pick RPC, then adopt position on a hit
            px, py = int(rng.integers(450)), int(rng.integers(433))
            req_q.put(("pick", i, rid, pos, quat, ps, px, py))
            hit = resp_q.get()
            if hit is not None:
                pos = list(hit)
        elif verb == 1:  # rotate (local, exact env arithmetic)
            e = geom.quaternion_to_euler(quat)
            d = (rng.integers(9, size=3) - 4) * 0.08
            quat = list(geom.euler_to_quaternion(
                [e[0] + d[0], e[1] + d[1], e[2] + d[2]]))
        else:  # zoom (local; floor mirrors NativeEnvironment's clamp)
            ps = max(1.0, min(500_000.0,
                              ps + (float(rng.integers(9)) - 4) * 500.0))
        req_q.put(("obs", i, rid, pos, quat, ps))
        resp_q.get()  # features
        n += 1
    out_q.put((i, n, time.monotonic() - t0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--meshes", type=int, default=8)
    ap.add_argument("--secs", type=float, default=90.0)
    ap.add_argument("--window-ms", type=float, default=5.0)
    ap.add_argument("--pipeline", action="store_true",
                    help="overlap render (main/GL thread) with encode "
                         "(worker thread): serial alternation caps ~176 sps")
    ap.add_argument("--pool", default="/scratch/kp0374/neurogym-agent/eval_d0_v1.parquet")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    from ngllib.native.colors import segment_color
    from ngllib.native.em import MeshStore
    from ngllib.native.render3d import MeshRenderer
    from ngllib_agent.obs import get_dino_encoder

    rids = [str(r) for r in
            pq.read_table(args.pool, columns=["root_id"])
            .column("root_id").to_pylist()][:args.meshes]
    rend = MeshRenderer(450, 433)
    store = MeshStore()
    centers = []
    for rid in rids:
        v, f = store.get(rid)
        rend.load_mesh(rid, v, f)
        centers.append([float(x) for x in
                        (v.mean(axis=0) / np.array([4.0, 4.0, 40.0]))])
    enc = get_dino_encoder()
    zoom = lambda ps: ps * 4.07  # noqa: E731
    VOX = np.array([4.0, 4.0, 40.0])

    ctx = mp.get_context("spawn")
    req_q = ctx.Queue()
    out_q = ctx.Queue()
    resp_qs = [ctx.Queue() for _ in range(args.k)]
    barrier = ctx.Barrier(args.k + 1)
    procs = [ctx.Process(target=_client,
                         args=(i, args.secs, rids, centers, req_q,
                               resp_qs[i], out_q, barrier))
             for i in range(args.k)]
    for p in procs:
        p.start()
    barrier.wait()

    t_render = t_enc = 0.0
    batches = frames_total = picks = 0

    enc_q = None
    if args.pipeline:
        import queue as _queue
        import threading

        enc_q = _queue.Queue(maxsize=2)
        stats_lock = threading.Lock()

        def _encoder_loop():
            nonlocal t_enc, batches, frames_total
            while True:
                item = enc_q.get()
                if item is None:
                    return
                frames_, reqs_ = item
                e0 = time.monotonic()
                feats = enc.encode(frames_ + frames_)
                for j, (_, i, *_r) in enumerate(reqs_):
                    resp_qs[i].put(feats[j])
                with stats_lock:
                    t_enc += time.monotonic() - e0
                    batches += 1
                    frames_total += len(frames_)

        enc_thread = threading.Thread(target=_encoder_loop, daemon=True)
        enc_thread.start()

    t_end = time.monotonic() + args.secs + 30
    while time.monotonic() < t_end:
        try:
            first = req_q.get(timeout=1.0)
        except Exception:
            if all(not p.is_alive() for p in procs):
                break
            continue
        group = [first]
        wend = time.monotonic() + args.window_ms / 1000.0
        while time.monotonic() < wend:
            try:
                group.append(req_q.get_nowait())
            except Exception:
                time.sleep(0.0005)
        # picks answered inline; obs requests batched into one encode
        obs_reqs = []
        t0 = time.monotonic()
        for msg in group:
            if msg[0] == "pick":
                _, i, rid, pos, quat, ps, px, py = msg
                pos_nm = np.asarray(pos) * VOX
                depth, view, proj = rend.pick_depth(rid, pos_nm, quat,
                                                    zoom(ps))
                d = depth[py, px]
                if d >= 0.9999:
                    resp_qs[i].put(None)
                else:
                    ndc = np.array([2 * (px + 0.5) / 450 - 1,
                                    1 - 2 * (py + 0.5) / 433,
                                    2 * d - 1, 1.0])
                    w = np.linalg.inv(proj @ view) @ ndc
                    resp_qs[i].put(list((w[:3] / w[3]) / VOX))
                picks += 1
            else:
                obs_reqs.append(msg)
        frames = []
        for _, i, rid, pos, quat, ps in obs_reqs:
            frames.append(rend.render(rid, np.asarray(pos) * VOX, quat,
                                      zoom(ps), segment_color(int(rid))))
        t1 = time.monotonic()
        if frames:
            if enc_q is not None:
                enc_q.put((frames, obs_reqs))  # encode overlaps next render
            else:
                feats = enc.encode(frames + frames)
                for j, (_, i, *_rest) in enumerate(obs_reqs):
                    resp_qs[i].put(feats[j])
                batches += 1
                frames_total += len(frames)
                t_enc += time.monotonic() - t1
        t_render += t1 - t0
    total_steps, max_el = 0, 1e-9
    for _ in range(args.k):
        try:
            i, n, el = out_q.get(timeout=60)
            total_steps += n
            max_el = max(max_el, el)
        except Exception:
            break
    for p in procs:
        p.join(timeout=20)
        if p.is_alive():
            p.terminate()
    print(f"RENDERSERVICE K={args.k} sps={total_steps / max_el:.1f} "
          f"mean_batch={frames_total / max(1, batches):.1f} picks={picks} "
          f"render+pick={1000 * t_render / max(1, batches):.1f}ms/batch "
          f"encode={1000 * t_enc / max(1, batches):.1f}ms/batch", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
