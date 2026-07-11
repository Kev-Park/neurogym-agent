"""Deeper readback levers: compositor flags + async screencast.

(1) Chrome flags that remove frame throttling / vsync waits — does the 50ms
    raw-CDP floor or the post-action render-wait drop?
(2) CDP screencast: browser pushes frames as rendered (async), vs the
    synchronous per-call captureScreenshot. Measure achievable frame cadence.
All standard/documented (Chrome flags + CDP Page.startScreencast).
"""

from __future__ import annotations

import time

import ngllib
from ngllib_agent.env_build import load_config
from ngllib_agent.providers import FlywireSkeletonProvider
from ngllib_agent.wrappers import ActionSpec, MultiDiscreteActionWrapper

EXTRA_FLAGS = [
    "--disable-gpu-vsync",
    "--disable-frame-rate-limit",
    "--run-all-compositor-stages-before-draw",
]

_orig = ngllib.Environment._build_launch_args


def build(extra_flags):
    if extra_flags:
        ngllib.Environment._build_launch_args = lambda self: _orig(self) + extra_flags
    else:
        ngllib.Environment._build_launch_args = _orig
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
    return env, base


def med(fn, n=25):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    ts.sort(); return ts[len(ts) // 2]


def measure(label, extra_flags):
    env, base = build(extra_flags)
    page = base.page
    cdp = page.context.new_cdp_session(page)
    cap = med(lambda: cdp.send("Page.captureScreenshot", {"format": "jpeg", "quality": 85}))
    lat = []
    for _ in range(15):
        t = time.perf_counter(); env.step(env.action_space.sample()); lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    print(f"[cap2] {label}: raw_cdp_capture={cap:.0f}ms  full_step={lat[len(lat)//2]:.0f}ms", flush=True)

    # screencast: push frames as they render; drive renders via steps
    frames = []
    def on_frame(p):
        frames.append(time.perf_counter())
        try:
            cdp.send("Page.screencastFrameAck", {"sessionId": p["sessionId"]})
        except Exception:
            pass
    cdp.on("Page.screencastFrame", on_frame)
    cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 85, "everyNthFrame": 1})
    t0 = time.perf_counter()
    for _ in range(20):
        env.step(env.action_space.sample())
    cdp.send("Page.stopScreencast")
    dt = time.perf_counter() - t0
    print(f"[cap2] {label}: screencast frames={len(frames)} over {dt:.1f}s "
          f"({len(frames)/dt:.1f} fps pushed while stepping)", flush=True)
    env.close()


def main() -> int:
    measure("baseline    ", [])
    measure("compositor+ ", EXTRA_FLAGS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
