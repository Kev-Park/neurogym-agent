"""WHY does the packaged viewer cost throughput, and does caching fix it?

V6 measured 34.1 sps packaged vs 43.4 hosted at Chrome 2x16 (job 923949). The
hypothesis is mechanical: `page.route` turns every viewer asset into a round
trip Chrome -> Playwright driver -> a PYTHON callback, on the same connection
the step loop uses for evaluate()/screenshot(), while `clear_cache_on_recycle`
wipes the HTTP cache every episode so all ~17 assets are re-served per recycle.

This measures the per-reset cost directly (resets are where contexts recycle),
and reports the route counters the renderer now keeps, for four arms:

  packaged            today's default
  packaged+cache      cache-control headers + no per-episode cache clear
  hosted              the Google-hosted build, cache cleared (production today)
  hosted+cache        hosted without the per-episode clear, for a fair pairing

Needs a GPU node.

    uv run --no-sync python native/probe_viewer_cost.py --resets 8
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resets", type=int, default=8)
    ap.add_argument("--steps-per-reset", type=int, default=3)
    args = ap.parse_args()

    from ngllib.chrome import ChromeRenderer

    layout = dict(window_size=(1800, 900), capture_scale=0.5,
                  left_pane=True, right_pane=True)
    arms = {
        "packaged": dict(clear_cache_on_recycle=True),
        "packaged+cache": dict(clear_cache_on_recycle=False),
        "hosted": dict(viewer="hosted", clear_cache_on_recycle=True),
        "hosted+cache": dict(viewer="hosted", clear_cache_on_recycle=False),
    }

    print(f"{'arm':16s} {'reset med':>10s} {'reset mean':>11s} {'step med':>9s} "
          f"{'route calls':>12s} {'route MB':>9s} {'route s':>8s}", flush=True)
    for name, kw in arms.items():
        r = ChromeRenderer(screenshot_format="png", **layout, **kw)
        state = r.default_state()
        resets, steps = [], []
        try:
            r.open()
            r.reset_to(state)              # first navigation: not counted (cold)
            for _ in range(args.resets):
                t0 = time.perf_counter()
                r.reset_to(state)
                resets.append(time.perf_counter() - t0)
                for _ in range(args.steps_per_reset):
                    t1 = time.perf_counter()
                    r.observe()
                    steps.append(time.perf_counter() - t1)
            stats = r.route_stats if r.viewer_dist is not None else {
                "calls": 0, "bytes": 0, "seconds": 0.0}
        finally:
            r.close()
        print(f"{name:16s} {statistics.median(resets):9.2f}s {statistics.mean(resets):10.2f}s "
              f"{statistics.median(steps) * 1000:8.0f}ms {stats['calls']:12.0f} "
              f"{stats['bytes'] / 1e6:9.1f} {stats['seconds']:8.1f}", flush=True)
    print("VIEWERCOST-DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
