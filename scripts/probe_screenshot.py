"""Quantify the screenshot lever (101ms = 84% of the step).

Times page.screenshot for varying viewport resolution and JPEG quality on a
real rendered viewer. Informs how much per-env throughput we can buy.
"""

from __future__ import annotations

import time

import numpy as np

import ngllib
from ngllib_agent.env_build import load_config
from ngllib_agent.providers import FlywireSkeletonProvider

N = 25


def timed(env, type_, quality):
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        if type_ == "jpeg":
            env.page.screenshot(type="jpeg", quality=quality)
        else:
            env.page.screenshot(type="png")
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]  # median ms


def main() -> int:
    cfg = load_config("configs/ppo_zmax_navigate.yaml")
    prov = FlywireSkeletonProvider(cfg["env"]["parquet_path"])

    for (w, h) in [(1800, 900), (1200, 600), (900, 450), (600, 300)]:
        env = ngllib.Environment(
            headless=True, renderer="gpu", orientation="euler",
            left_pane=True, right_pane=True, window_size=(w, h),
            reset_state_provider=prov,
        )
        env.reset(seed=0)
        # let a couple frames settle
        for _ in range(2):
            env.step(env.action_space.sample())
        row = f"[shot] window={w}x{h} ({w*h//1000}Kpx):"
        for q in (85, 50, 30):
            row += f"  jpeg_q{q}={timed(env, 'jpeg', q):.0f}ms"
        row += f"  png={timed(env, 'png', 0):.0f}ms"
        print(row, flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
