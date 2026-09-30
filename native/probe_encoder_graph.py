"""Validate + benchmark the ported CUDA-graph encode() path (obs.dino.cuda_graph).

Gate before trusting graphs in a training sweep:
  1. PARITY: graphed encode() must be BITWISE-identical to eager across MULTIPLE
     distinct random inputs (proves replay re-reads fresh pixels, not a stale
     capture).
  2. SPEEDUP: graph vs eager encode() ms/call at the co-pane batch (B=2).
Also reports fp16 (expected NON-bitwise; a dynamics-gated lever) for context.

Usage (GPU node):  uv run --no-sync python native/probe_encoder_graph.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ngllib_agent.obs.dino_encoder import DinoEncoder

PANE = (450, 450, 3)   # ~co-pane half-window after capture_scale 0.5


def _imgs(rng, b=2):
    return [rng.integers(0, 256, PANE, dtype=np.uint8) for _ in range(b)]


def _bench(enc, rng, n=100):
    enc.encode(_imgs(rng))  # warm / capture
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        enc.encode(_imgs(rng))
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / n


def main() -> int:
    assert torch.cuda.is_available(), "needs a GPU"
    rng = np.random.default_rng(0)
    print(f"[enc-graph] {torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)

    eager = DinoEncoder(use_cuda_graph=False)
    graph = DinoEncoder(use_cuda_graph=True)

    # --- parity across 10 distinct inputs -------------------------------------
    worst = 0.0
    for _ in range(10):
        ims = _imgs(rng)
        fe = eager.encode(ims)
        fg = graph.encode(ims)
        worst = max(worst, float(np.abs(fe - fg).max()))
    print(f"[enc-graph] PARITY graph-vs-eager worst_abs_diff={worst:.2e} "
          f"({'BITWISE OK' if worst == 0.0 else 'DRIFT'})", flush=True)

    # --- speedup --------------------------------------------------------------
    eager_ms = _bench(eager, rng)
    graph_ms = _bench(graph, rng)
    print(f"[enc-graph] encode() B=2  eager={eager_ms:.2f}ms  graph={graph_ms:.2f}ms  "
          f"speedup={eager_ms/graph_ms:.2f}x", flush=True)

    # --- fp16 reference (dynamics-gated; expected non-bitwise) -----------------
    try:
        fp16 = DinoEncoder(use_cuda_graph=False, use_fp16=True)
        ims = _imgs(rng)
        d = float(np.abs(eager.encode(ims) - fp16.encode(ims)).max())
        fp16_ms = _bench(fp16, rng)
        print(f"[enc-graph] fp16  encode={fp16_ms:.2f}ms  vs_eager_abs_diff={d:.2e} "
              f"(DYNAMICS-GATED)", flush=True)
    except Exception as e:
        print(f"[enc-graph] fp16 probe failed: {type(e).__name__}: {str(e)[:80]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
