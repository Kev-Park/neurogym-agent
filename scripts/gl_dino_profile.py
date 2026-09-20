"""Profile the GL-render vs DINO-encode GPU-time split for the batched pipeline.

The fastest-stable config is GPU-compute-bound (SM ~100%), but we don't know how
that time splits between GL rasterization and the DINO forward — which decides
what a dedicated-GPU disaggregation should dedicate. This times, for one batch of
B panes: render-only, encode-only, and combined, on a GPU node:

    PROF_BATCH=8 PROF_ITERS=200 uv run --no-sync python scripts/gl_dino_profile.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from ngllib.simulator.pane2d import PANE, PANE_H
from ngllib.simulator.render3d import MeshRenderer
from ngllib_agent.obs.dino_encoder import DinoEncoder


def cube(center, half):
    c = np.asarray(center, dtype="f4")
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
                 dtype="f4") * half + c
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
                 dtype="i4")
    return v, f


def scene():
    plane = (np.linspace(0, 255, 64)[:, None] * np.ones(64)[None, :]).astype(np.uint8)
    return dict(root_id=["1"], position_nm=[0.0, 0.0, 0.0], quat=[0.0, 0.0, 0.0, 1.0],
                zoom_nm=8000.0, color=(1.0, 0.4, 0.2), em_tile=plane,
                em_extent_nm=(16000.0, 16000.0), em_gain=1.0)


def _sync():
    torch.cuda.synchronize()


def main():
    B = int(os.environ.get("PROF_BATCH", "8"))
    N = int(os.environ.get("PROF_ITERS", "200"))
    mr = MeshRenderer(PANE, PANE_H, cuda_ipc=True, ipc_export=False)
    mr.enable_atlas(B)
    v, f = cube([0, 0, 0], 2500.0)
    mr.load_mesh("1", v, f)
    enc = DinoEncoder()
    scenes = [scene() for _ in range(B)]
    print(f"GL: {mr.ctx.info.get('GL_RENDERER','?')}  B={B} N={N}", flush=True)

    # warmup both paths
    for _ in range(10):
        views = mr.render_batch(scenes, to_cuda=True)
        enc.encode_gpu(views, gl_flip=True)
    _sync()

    # render-only (GL raster + GL->CUDA copy; no DINO)
    t0 = time.perf_counter()
    for _ in range(N):
        views = mr.render_batch(scenes, to_cuda=True)
    _sync()
    t_render = (time.perf_counter() - t0) / N

    # encode-only (DINO forward on the last batch's views; no render)
    t0 = time.perf_counter()
    for _ in range(N):
        enc.encode_gpu(views, gl_flip=True)
    _sync()
    t_encode = (time.perf_counter() - t0) / N

    # combined (render then encode, per step)
    t0 = time.perf_counter()
    for _ in range(N):
        vv = mr.render_batch(scenes, to_cuda=True)
        enc.encode_gpu(vv, gl_flip=True)
    _sync()
    t_comb = (time.perf_counter() - t0) / N

    r_ms, e_ms, c_ms = t_render * 1e3, t_encode * 1e3, t_comb * 1e3
    print(f"render-only : {r_ms:7.2f} ms/batch ({r_ms/B:.2f} ms/pane)", flush=True)
    print(f"encode-only : {e_ms:7.2f} ms/batch ({e_ms/B:.2f} ms/pane)", flush=True)
    print(f"combined    : {c_ms:7.2f} ms/batch", flush=True)
    print(f"sum(r+e)    : {r_ms+e_ms:7.2f} ms  -> overlap/contention = "
          f"{(r_ms+e_ms)-c_ms:+.2f} ms ({100*((r_ms+e_ms)-c_ms)/(r_ms+e_ms):+.1f}%)",
          flush=True)
    tot = r_ms + e_ms
    print(f"GPU-time split (serial est): render {100*r_ms/tot:.0f}% / "
          f"DINO {100*e_ms/tot:.0f}%", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
