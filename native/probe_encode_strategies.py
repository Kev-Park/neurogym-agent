"""DINO encoding-strategy shootout (standalone; no RLlib, no training).

Modes:
  bench       — encoder frames/sec vs batch size (one process, one context).
  per-runner  — M raw native envs in one process (ThreadedVectorEnv); each
                vector step does ONE encode(2M frames) call. Launch K
                concurrent instances (sbatch script) for node totals.
  service     — per-node encoder service: ONE server process owns the
                encoder; K spawned single-env client processes send frame
                pairs over queues, the server batches whatever arrives in a
                short window. Aggregate sps printed by the parent.

RLlib mapping: per-runner == an env-to-module connector inside each
EnvRunner (idiomatic). service == a custom Ray actor called from env code
(possible, not native). This probe prices both without RLlib in the loop.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import numpy as np


def _sample_action(rng):
    return np.array([rng.integers(3), rng.integers(1024), rng.integers(9),
                     rng.integers(9), rng.integers(9), rng.integers(9)])


def mode_bench(args):
    from ngllib_agent.obs import get_dino_encoder

    enc = get_dino_encoder()
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (450, 450, 3), dtype=np.uint8)
              for _ in range(128)]
    for bs in (1, 2, 4, 8, 16, 32, 64, 128):
        enc.encode(frames[:bs])  # warmup
        n, t0 = 0, time.monotonic()
        while time.monotonic() - t0 < 5.0:
            enc.encode(frames[:bs])
            n += 1
        el = time.monotonic() - t0
        print(f"BENCH bs={bs:3d}: {n * bs / el:7.1f} frames/s "
              f"({1000 * el / n:6.1f} ms/call)", flush=True)
    return 0


def mode_per_runner(args):
    from ngllib_agent.env_build import load_config, make_env_creator
    from ngllib_agent.obs import get_dino_encoder

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = "raw"
    m = args.m
    venv = make_env_creator(cfg, vector_mode="threads")({"num_envs": m})
    enc = get_dino_encoder()
    venv.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    def stepbatch():
        acts = np.stack([_sample_action(rng) for _ in range(m)])
        obs, *_ = venv.step(acts)
        imgs = obs["image"]  # (M, 450, 900, 3)
        frames = list(imgs[:, :, :450]) + list(imgs[:, :, 450:])
        enc.encode(frames)  # ONE forward for all M envs, both panes

    for _ in range(5):
        stepbatch()
    n, t0 = 0, time.monotonic()
    while time.monotonic() - t0 < args.secs:
        stepbatch()
        n += 1
    el = time.monotonic() - t0
    print(f"PERRUNNER M={m} sps={m * n / el:.1f} per_env={n / el:.2f}",
          flush=True)
    venv.close()
    return 0


def _service_client(i, cfg_path, secs, req_q, resp_q, out_q, barrier):
    os.environ.setdefault("NGL_NATIVE_FETCH_WORKERS", "1")
    from ngllib_agent.env_build import build_env, load_config

    cfg = load_config(cfg_path)
    cfg.setdefault("obs", {})["mode"] = "raw"
    env = build_env(cfg)
    obs, _ = env.reset(seed=100 + i)
    rng = np.random.default_rng(i)
    barrier.wait()
    n, t0 = 0, time.monotonic()
    while time.monotonic() - t0 < secs:
        obs, r, term, trunc, _ = env.step(_sample_action(rng))
        if term or trunc:
            obs, _ = env.reset()
        img = obs["image"]
        req_q.put((i, img[:, :450].copy(), img[:, 450:].copy()))
        resp_q.get()  # features (synchronous step, like a wrapper would be)
        n += 1
    out_q.put((i, n, time.monotonic() - t0))
    env.close()


def mode_service(args):
    from ngllib_agent.obs import get_dino_encoder

    enc = get_dino_encoder()
    ctx = mp.get_context("spawn")
    req_q = ctx.Queue()
    out_q = ctx.Queue()
    resp_qs = [ctx.Queue() for _ in range(args.k)]
    barrier = ctx.Barrier(args.k + 1)
    procs = [ctx.Process(target=_service_client,
                         args=(i, args.config, args.secs, req_q, resp_qs[i],
                               out_q, barrier))
             for i in range(args.k)]
    for p in procs:
        p.start()
    barrier.wait()
    t_end = time.monotonic() + args.secs + 30
    batches, batched_frames = 0, 0
    while time.monotonic() < t_end:
        try:
            first = req_q.get(timeout=1.0)
        except Exception:
            if all(not p.is_alive() for p in procs):
                break
            continue
        group = [first]
        window_end = time.monotonic() + args.window_ms / 1000.0
        while time.monotonic() < window_end:
            try:
                group.append(req_q.get_nowait())
            except Exception:
                time.sleep(0.001)
        frames = [f for (_, l, r) in group for f in (l, r)]
        feats = enc.encode(frames)
        for j, (i, _, _) in enumerate(group):
            resp_qs[i].put(feats[2 * j:2 * j + 2])
        batches += 1
        batched_frames += len(frames)
    total_steps, max_el = 0, 1e-9
    for _ in range(args.k):
        try:
            i, n, el = out_q.get(timeout=60)
            total_steps += n
            max_el = max(max_el, el)
        except Exception:
            break
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    print(f"SERVICE K={args.k} sps={total_steps / max_el:.1f} "
          f"mean_batch={batched_frames / max(1, batches):.1f} frames",
          flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "per-runner", "service"],
                    required=True)
    ap.add_argument("--config", default="configs/native.yaml")
    ap.add_argument("--m", type=int, default=4, help="envs (per-runner mode)")
    ap.add_argument("--k", type=int, default=32, help="clients (service mode)")
    ap.add_argument("--secs", type=float, default=120.0)
    ap.add_argument("--window-ms", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    return {"bench": mode_bench, "per-runner": mode_per_runner,
            "service": mode_service}[args.mode](args)


if __name__ == "__main__":
    import sys
    sys.exit(main())
