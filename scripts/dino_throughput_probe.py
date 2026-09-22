"""D_cap microbench: max DINO encode throughput on ONE dedicated GPU.

The dedicated-DINO-GPU disaggregation can only beat the shared config (~317 sps)
if dedicated big-batch DINO is MORE SM-efficient than the shared GPU's implied
DINO throughput D_cap_shared = 317 / 0.83 ~= 382 img/s. This measures the UPPER
BOUND on a dedicated DINO GPU's throughput (synthetic frames back-to-back, no
cross-GPU transfer, no queue) as images/sec vs batch size, eager vs CUDA-graph.

Reads:
  images/sec at each batch = how many env-steps/sec one dedicated DINO GPU could
  serve. Compare the best to ~382:
    <= ~382  -> dedicated-DINO disagg CANNOT beat shared 317 -> don't build it.
    >  ~382  -> real headroom -> measure R_cap + build the split.

Single process (so CUDA-graph capture works, unlike 24-way MPS). Needs a GPU.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ngllib_agent.obs.dino_encoder import DinoEncoder

H, W, C = 433, 450, 4          # interop pane shape (RGBA, GL-flipped, toolbar-padded)
FLIP, PAD = True, 17
BATCHES = [2, 4, 8, 12, 24, 48]
D_CAP_SHARED = 382.0            # 317 / 0.83, the number disagg must beat


def _bench(enc: DinoEncoder, batch: int, g: torch.Generator, iters: int = 40) -> float:
    imgs = [torch.randint(0, 256, (H, W, C), dtype=torch.uint8, device="cuda", generator=g)
            for _ in range(batch)]
    for _ in range(5):                       # warm + capture
        enc.encode_gpu([t for t in imgs], gl_flip=FLIP, top_pad=PAD)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        enc.encode_gpu([t for t in imgs], gl_flip=FLIP, top_pad=PAD)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return batch / dt                        # images/sec


def main() -> int:
    assert torch.cuda.is_available(), "need a GPU"
    g = torch.Generator(device="cuda").manual_seed(0)
    best = 0.0
    for use_graph in (False, True):
        enc = DinoEncoder(use_cuda_graph=use_graph)
        tag = "graph" if use_graph else "eager"
        for b in BATCHES:
            ips = _bench(enc, b, g)
            captured = use_graph and len(enc._graphs) > 0
            best = max(best, ips)
            print(f"D_cap {tag} batch={b:3d}: {ips:7.1f} img/s "
                  f"(fwd/s={ips / b:6.1f}) captured={captured}")
    verdict = "HEADROOM -> measure R_cap + build split" if best > D_CAP_SHARED \
        else "NO HEADROOM -> dedicated-DINO disagg cannot beat shared 317"
    print(f"D_CAP_BEST={best:.1f} img/s  vs D_cap_shared~={D_CAP_SHARED}  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
