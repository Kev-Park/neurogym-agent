"""Flood the screenshot-readback path to measure its raw ceiling directly.

Full-step throughput (~40 sps/GPU) bundles capture + DINO + state-gather +
decode. This strips all of that: build M browsers on ONE GPU with a real
Neuroglancer frame loaded, then each browser tight-loops raw CDP
`Page.captureScreenshot` (no decode, no action, no DINO). Aggregate captures/sec
at the plateau = the pure readback/encode/transport ceiling.
  - lands ≈ full-step ~40  => capture IS the wall (no headroom w/o async).
  - lands ≫ 40             => full-step rate is bound by the OTHER per-step work
                              (DINO/decode/GIL) and there's headroom to reclaim.

Playwright's sync API is thread-bound (a browser can only be driven from the
thread that created it — the greenlet dispatcher can't switch threads). So each
browser is BOTH built and flooded inside its own dedicated thread; a barrier
makes all M flood the same T-second window concurrently. base64 results are
discarded (≈no GIL-held decode), so this is close to the true readback ceiling.

    uv run --no-sync python scripts/probe_flood.py <M> <seconds>
"""

from __future__ import annotations

import sys
import threading
import time

import ngllib
from ngllib_agent.env_build import load_config
from ngllib_agent.providers import FlywireSkeletonProvider

_SHOT = {"format": "jpeg", "quality": 85}


def worker(i, cfg, T, barrier, results):
    ec = cfg["env"]
    base = None
    try:
        base = ngllib.Environment(
            headless=True, renderer="gpu", orientation="euler",
            left_pane=True, right_pane=True, window_size=(1800, 900),
            reset_state_provider=FlywireSkeletonProvider(ec["parquet_path"]),
        )
        base.reset(seed=i)
        page = base.page
        cdp = page.context.new_cdp_session(page)
        for _ in range(3):  # warm the capture path
            cdp.send("Page.captureScreenshot", _SHOT)
    except Exception as e:
        results[i] = ("BUILD_FAIL", f"{type(e).__name__}: {str(e)[:70]}")
        try:
            barrier.wait(timeout=5)  # don't deadlock the others
        except Exception:
            pass
        if base is not None:
            try:
                base.close()
            except Exception:
                pass
        return

    # all browsers built -> flood the same window together
    try:
        barrier.wait(timeout=300)
    except Exception:
        pass
    deadline = time.time() + T
    c = 0
    try:
        while time.time() < deadline:
            cdp.send("Page.captureScreenshot", _SHOT)
            c += 1
    finally:
        results[i] = ("OK", c)
        try:
            base.close()
        except Exception:
            pass


def main() -> int:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
    cfg = load_config("configs/ppo_zmax_navigate.yaml")

    results = [None] * M
    barrier = threading.Barrier(M)
    threads = [threading.Thread(target=worker, args=(i, cfg, T, barrier, results))
               for i in range(M)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = [c for (s, c) in results if s == "OK"]
    fails = [c for (s, c) in results if s != "OK"]
    eff = len(ok)
    total = sum(ok)
    caps_s = total / T if T else 0.0
    print(f"[flood] RESULT M={M} eff_browsers={eff} caps_s={caps_s:.1f} "
          f"per_browser={caps_s/eff if eff else 0:.2f} (caps={total} in {T:.0f}s, "
          f"build_fails={len(fails)})", flush=True)
    if fails:
        print(f"[flood]   fail sample: {fails[0]}", flush=True)
    return 0 if eff else 1


if __name__ == "__main__":
    raise SystemExit(main())
