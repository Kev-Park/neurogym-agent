"""Investigate the screenshot-readback floor: compare capture methods.

Isolates where the ~67ms goes by timing alternative capture paths on the same
rendered viewer, all non-hacky (documented CDP APIs / Chrome flags):
  1. Playwright page.screenshot   (adds stability waits on top of CDP)
  2. raw CDP Page.captureScreenshot (no Playwright wrapper)
  3. + optimizeForSpeed:true      (CDP encode-speed hint)
  4. lower quality via raw CDP     (confirms encode isn't the cost)
  5. action->fresh-frame latency   (what the RL loop actually needs)
"""

from __future__ import annotations

import base64
import time

import numpy as np

import ngllib
from ngllib_agent.env_build import load_config
from ngllib_agent.providers import FlywireSkeletonProvider
from ngllib_agent.wrappers import ActionSpec, MultiDiscreteActionWrapper


def med(fn, n=25):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def main() -> int:
    cfg = load_config("configs/ppo_zmax_navigate.yaml")
    ec, ac = cfg["env"], cfg["action"]
    base = ngllib.Environment(
        headless=True, renderer="gpu", orientation="euler",
        left_pane=True, right_pane=True, window_size=(1800, 900),
        reset_state_provider=FlywireSkeletonProvider(ec["parquet_path"]),
    )
    x0, y0, x1, y1 = ac["pane_3d_bounds"]
    spec = ActionSpec(grid_rows=ac["grid_rows"], grid_cols=ac["grid_cols"],
                      pane_x0=x0, pane_y0=y0, pane_x1=x1, pane_y1=y1,
                      rotation_bins_per_axis=ac["rotation_bins_per_axis"],
                      rotation_step_rad=ac["rotation_step_rad"],
                      zoom_bins=ac["zoom_bins"], zoom_step=ac["zoom_step"])
    env = MultiDiscreteActionWrapper(base, spec)
    env.reset(seed=0)
    for _ in range(4):
        env.step(env.action_space.sample())

    page = base.page
    cdp = page.context.new_cdp_session(page)

    def cdp_shot(quality=85, opt=False):
        p = {"format": "jpeg", "quality": quality}
        if opt:
            p["optimizeForSpeed"] = True
        cdp.send("Page.captureScreenshot", p)

    print(f"[cap] playwright  jpeg85         : {med(lambda: page.screenshot(type='jpeg', quality=85)):.0f}ms", flush=True)
    print(f"[cap] raw-CDP     jpeg85         : {med(lambda: cdp_shot(85)):.0f}ms", flush=True)
    print(f"[cap] raw-CDP     jpeg85 optSpeed: {med(lambda: cdp_shot(85, True)):.0f}ms", flush=True)
    print(f"[cap] raw-CDP     jpeg30 optSpeed: {med(lambda: cdp_shot(30, True)):.0f}ms", flush=True)

    # 5. What the RL loop needs: after applying an action, how long until a
    #    fresh, correct frame is capturable? (captureScreenshot forces a commit.)
    lat = []
    for _ in range(15):
        t = time.perf_counter()
        env.step(env.action_space.sample())  # apply action + gather obs (screenshot inside)
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    print(f"[cap] full env.step (action->obs): median={lat[len(lat)//2]:.0f}ms", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
