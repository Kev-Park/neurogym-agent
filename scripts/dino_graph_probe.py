"""DINO-GRAPH-PARITY gate: verify the CUDA-graph encode path is numerically
identical to the eager path, across several DISTINCT random frames.

Two things must hold, and the second is the one a naive capture gets wrong:
  1. graph(frame) allclose eager(frame)                 -- capture is faithful
  2. feeding a NEW frame then replay returns the NEW result (matches eager on
     the new frame), i.e. copy_->replay actually re-reads the static buffer and
     does not return the stale output captured at build time.

Also prints a rough per-call timing (eager vs graph) at the interop batch shape.
Needs a GPU + the agent venv. Run via scripts/dino_graph_probe.slurm.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ngllib_agent.obs.dino_encoder import DinoEncoder

# Interop panes are RGBA (C=4, alpha dropped), GL-flipped, toolbar-padded. Batch
# = M envs per runner. Shape is representative; parity is shape-independent.
H, W, C = 433, 450, 4
BATCH = 2
FLIP = True
PAD = 17
N_FRAMES = 6
TOL = 1e-4


def _rand_frames(g: torch.Generator):
    return [torch.randint(0, 256, (H, W, C), dtype=torch.uint8, device="cuda", generator=g)
            for _ in range(BATCH)]


def main() -> int:
    assert torch.cuda.is_available(), "need a GPU"
    eager = DinoEncoder(use_cuda_graph=False)
    graph = DinoEncoder(use_cuda_graph=True)

    g = torch.Generator(device="cuda").manual_seed(0)
    max_diff = 0.0
    stale_ok = True
    prev_graph_out = None
    for i in range(N_FRAMES):
        imgs = _rand_frames(g)
        e = eager.encode_gpu([t.clone() for t in imgs], gl_flip=FLIP, top_pad=PAD)
        gr = graph.encode_gpu([t.clone() for t in imgs], gl_flip=FLIP, top_pad=PAD)
        d = float(np.abs(e - gr).max())
        max_diff = max(max_diff, d)
        # (2) distinct frames must yield distinct graph outputs (not a frozen replay)
        if prev_graph_out is not None:
            if np.allclose(prev_graph_out, gr, atol=TOL):
                stale_ok = False
        prev_graph_out = gr
        print(f"frame {i}: max|eager-graph|={d:.3e}")

    # timing at the interop batch shape (warm)
    imgs = _rand_frames(g)
    for _ in range(3):
        eager.encode_gpu([t.clone() for t in imgs], gl_flip=FLIP, top_pad=PAD)
        graph.encode_gpu([t.clone() for t in imgs], gl_flip=FLIP, top_pad=PAD)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        eager.encode_gpu([t.clone() for t in imgs], gl_flip=FLIP, top_pad=PAD)
    torch.cuda.synchronize()
    te = (time.perf_counter() - t0) / 50 * 1e3
    t0 = time.perf_counter()
    for _ in range(50):
        graph.encode_gpu([t.clone() for t in imgs], gl_flip=FLIP, top_pad=PAD)
    torch.cuda.synchronize()
    tg = (time.perf_counter() - t0) / 50 * 1e3

    used_graph = len(graph._graphs) > 0 and not graph._graph_failed
    print(f"per-call (batch={BATCH}): eager={te:.3f}ms graph={tg:.3f}ms "
          f"speedup={te / tg:.2f}x captured={used_graph}")

    ok = (max_diff <= TOL) and stale_ok and used_graph
    print(f"DINO-GRAPH-PARITY {'PASS' if ok else 'FAIL'} "
          f"(max_diff={max_diff:.3e} tol={TOL} distinct_outputs={stale_ok} captured={used_graph})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
