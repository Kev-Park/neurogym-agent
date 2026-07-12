"""Is per-env DINO the ~40-sps wall, and would batching fix it?

Decode and capture are ruled out (probe_decode / probe_flood). Remaining suspect:
DINO is called PER ENV at batch 2 (dino_obs.py `encode([left, right])`), so a
32-env vector-step does 32 tiny forwards + 32 host<->GPU round-trips instead of
one batched forward. This measures:

  (1) single-call encode() latency by batch size (2 = per-env; 32/64 = batched
      across a process's envs) -> images/s and env-steps/s.
  (2) threaded per-env style: T threads each looping encode(batch 2) on the
      SHARED per-process encoder (mimics ThreadedVectorEnv). Aggregate calls/s
      = env-steps/s the current design allows. If ≈40, per-env DINO IS the wall.

If batched images/s ≫ threaded-per-env env-steps/s, batching DINO at the vector
level is the fix.

    uv run --no-sync python scripts/probe_dino.py
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ngllib_agent.obs.dino_encoder import get_dino_encoder

H = W = 900  # one pane of the 1800x900 two-pane window


def med(fn, n=20):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def imgs(B):
    return [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(B)]


def main() -> int:
    enc = get_dino_encoder()
    enc.encode(imgs(2))  # warm CUDA / cuDNN autotune

    print("[dino] single-call encode() latency by batch (1 env-step = 2 panes):",
          flush=True)
    for B in [2, 8, 16, 32, 64]:
        ims = imgs(B)
        m = med(lambda: enc.encode(ims), n=20)
        print(f"[dino]   batch={B:>2}: {m:6.1f}ms  images/s={B*1000/m:5.0f}  "
              f"env_steps/s={(B/2)*1000/m:5.0f}", flush=True)

    print("[dino] threaded per-env style (T threads x encode(batch2), shared encoder):",
          flush=True)
    for Tn in [1, 8, 16, 32]:
        ims_per = [imgs(2) for _ in range(Tn)]
        counts = [0] * Tn
        deadline = time.time() + 8.0

        def worker(i):
            c = 0
            while time.time() < deadline:
                enc.encode(ims_per[i])
                c += 1
            counts[i] = c

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(Tn)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dt = time.time() - t0
        tot = sum(counts)
        print(f"[dino]   T={Tn:>2}: aggregate_env_steps/s={tot/dt:5.0f}  "
              f"per_thread={tot/dt/Tn:5.1f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
