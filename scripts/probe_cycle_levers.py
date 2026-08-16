"""Cycle-time lever probe (2026-08-16) — capture scale / cache clear / Chrome flags.

Phase A  capture_scale in {1.0, 0.5, 0.25}: mean step wall time over 30 steps
         (2:1 aspect preserved; browser-side GPU downscale replaces Python
         decode+resize work). Saves one sample frame per scale for visual QA.
Phase B  clear_cache_on_recycle {True, False}: mean reset time across 10
         episodes (warm HTTP cache should cut app+mesh re-downloads) + Chrome
         RSS at start vs end (bounded-cache check).
Phase C  footprint flag bundle {off, on}: mean step time + WebGL renderer
         string (must stay ANGLE/NVIDIA — no SwiftShader) + sample frame +
         mean|frame| sanity (non-black).

    uv run --no-sync python scripts/probe_cycle_levers.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import psutil
from PIL import Image

import ngllib
from ngllib_agent.env_build import load_config
from ngllib_agent.providers import FlywireSkeletonProvider

OUT = "/tmp/cycle_levers"
os.makedirs(OUT, exist_ok=True)

FLAG_BUNDLE = [
    "--disable-background-networking",
    "--metrics-recording-only",
    "--mute-audio",
    "--no-first-run",
    "--disable-default-apps",
    "--disable-breakpad",
    "--disable-component-update",
    "--disable-site-isolation-trials",
]

WEBGL_JS = """() => {
  const c = document.createElement('canvas');
  const gl = c.getContext('webgl2') || c.getContext('webgl');
  if (!gl) return 'NO-GL';
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'no-debug-ext';
}"""


def build(cfg, **kw):
    ec = cfg["env"]
    return ngllib.Environment(
        headless=True, renderer="gpu", orientation="euler",
        left_pane=True, right_pane=True, window_size=(1800, 900),
        reset_state_provider=FlywireSkeletonProvider(ec["parquet_path"]),
        **kw,
    )


def chrome_rss_mb() -> float:
    total = 0
    for c in psutil.Process(os.getpid()).children(recursive=True):
        try:
            if "chrom" in c.name().lower():
                total += c.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / 1e6


def timed_steps(env, n=30):
    ts = []
    for _ in range(n):
        t0 = time.monotonic()
        try:
            env.step(env.action_space.sample())
        except Exception:
            continue
        ts.append((time.monotonic() - t0) * 1000)
    return sum(ts) / max(1, len(ts))


def main() -> int:
    cfg = load_config("configs/ppo_zmax_navigate.yaml")

    print("### Phase A — capture_scale ###", flush=True)
    for scale in (1.0, 0.5, 0.25):
        env = build(cfg, capture_scale=scale)
        obs, _ = env.reset(seed=0)
        ms = timed_steps(env, 30)
        img = obs["image"]
        Image.fromarray(img).save(f"{OUT}/scaleA_{scale}.png")
        print(f"[A] scale={scale}: step_ms={ms:.1f} shape={img.shape} "
              f"mean_px={img.mean():.1f}", flush=True)
        env.close()

    print("### Phase B — clear_cache_on_recycle ###", flush=True)
    for clear in (True, False):
        env = build(cfg, clear_cache_on_recycle=clear)
        env.reset(seed=0)
        rss0 = chrome_rss_mb()
        resets = []
        for ep in range(10):
            for _ in range(5):
                try:
                    env.step(env.action_space.sample())
                except Exception:
                    pass
            t0 = time.monotonic()
            env.reset()
            resets.append((time.monotonic() - t0) * 1000)
        rss1 = chrome_rss_mb()
        m = sorted(resets)
        print(f"[B] clear={clear}: reset_ms med={m[len(m)//2]:.0f} "
              f"mean={sum(resets)/len(resets):.0f} max={max(resets):.0f} "
              f"chrome_rss {rss0:.0f}->{rss1:.0f} MB", flush=True)
        env.close()

    print("### Phase C — footprint flag bundle ###", flush=True)
    for label, flags in (("off", None), ("on", FLAG_BUNDLE)):
        env = build(cfg, extra_launch_args=flags)
        obs, _ = env.reset(seed=0)
        renderer = env.page.evaluate(WEBGL_JS)
        ms = timed_steps(env, 30)
        img = env._prev_obs["image"]
        Image.fromarray(img).save(f"{OUT}/flagsC_{label}.png")
        print(f"[C] flags={label}: step_ms={ms:.1f} mean_px={img.mean():.1f} "
              f"renderer={str(renderer)[:70]}", flush=True)
        env.close()

    print(f"[done] sample frames in {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
