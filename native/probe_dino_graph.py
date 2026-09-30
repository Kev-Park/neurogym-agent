"""Isolated CEILING test: does a CUDA graph speed up THIS workload's DINO forward?

CUDA graphs are implemented on the throughput-scaling branch (right-pane SOTA,
+6% aggregate) but NOT on zmax-left, so v5 doesn't get them. Before porting the
encoder graph path, this measures the UPPER BOUND: the frozen ViT-S/14 forward,
eager vs a captured CUDA graph, at the batch sizes the co-pane obs actually uses
(B=2 = one env's [EM, 3D] panes; B=4 = M=2 envs x 2 panes if batched).

Caveat baked into the interpretation: an isolated per-call speedup does NOT
translate 1:1 to training sps — under N-way MPS the idle-SM gaps a graph closes
are already filled by other processes' kernels (right-pane: 1.43x isolated ->
+6% aggregate). This number is the ceiling, not the expected training gain.

Usage (GPU node):  uv run --no-sync python native/probe_dino_graph.py
"""

from __future__ import annotations

import time

import torch


def _bench(fn, n: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / n  # ms/call


def main() -> int:
    assert torch.cuda.is_available(), "needs a GPU"
    dev = torch.device("cuda")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
    model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[dino-graph] {torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)

    for B in (1, 2, 4):
        src = torch.randn(B, 3, 224, 224, device=dev)
        static_in = torch.randn(B, 3, 224, 224, device=dev)

        @torch.no_grad()
        def eager():
            src.normal_()               # fresh data each call (defeats caching)
            return model(src)

        _bench(eager, 20)               # WARM this batch shape (cudnn autotune +
                                        # lazy alloc) so the timed run is not
                                        # cold — the first-benched shape otherwise
                                        # eats all the one-time warmup.
        eager_ms = _bench(eager, 100)

        # ---- capture a CUDA graph of the forward from a static input buffer ----
        graph_ms = None
        note = ""
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(static_in)
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(g):
                static_out = model(static_in)

            def graphed():
                static_in.copy_(torch.randn_like(static_in))  # new pixels in
                g.replay()
                return static_out

            graph_ms = _bench(graphed, 100)

            # parity: graph replay must recompute on fresh input (not stale capture)
            static_in.copy_(src)
            g.replay()
            with torch.no_grad():
                ref = model(src)
            max_abs = float((static_out - ref).abs().max())
            note = f"parity max_abs_diff={max_abs:.2e}"
        except Exception as e:  # capture can fail on some ops; report, don't crash
            note = f"CAPTURE FAILED: {type(e).__name__}: {str(e)[:80]}"

        if graph_ms:
            print(f"[dino-graph] B={B}  eager={eager_ms:.2f}ms  graph={graph_ms:.2f}ms  "
                  f"speedup={eager_ms/graph_ms:.2f}x  ({note})", flush=True)
        else:
            print(f"[dino-graph] B={B}  eager={eager_ms:.2f}ms  graph=n/a  ({note})", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
