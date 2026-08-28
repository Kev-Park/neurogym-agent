"""Reset-ahead prefetch probe: provider-driven resets with timing.

Reset 1 is cold (no prefetch yet); resets 2+ should adopt the prefetched
mesh/tiles and be much faster. Steps between resets give the prefetch time
to resolve (an episode normally lasts minutes; here we sleep briefly).

    uv run --no-sync python native/probe_reset_ahead.py
"""

from __future__ import annotations

import time

import numpy as np


def main() -> int:
    from ngllib_agent.env_build import build_env, load_config

    cfg = load_config("configs/native.yaml")
    cfg.setdefault("obs", {})["mode"] = "pos"
    env = build_env(cfg)

    # Training-shaped load: click-heavy stepping (position changes -> tile
    # groups compete with the mesh prefetch on the fetch pool), no idle.
    rng = np.random.default_rng(7)
    for ep in range(6):
        t0 = time.monotonic()
        obs, info = env.reset()
        t_reset = time.monotonic() - t0
        rid = info["task_info"].get("segment_id")
        t0 = time.monotonic()
        n_steps = 60
        for i in range(n_steps):
            if i % 2 == 0:  # half clicks, half rotates
                a = np.array([0, rng.integers(1024), 4, 4, 4, 4])
            else:
                a = np.array([1, 0, rng.integers(9), rng.integers(9),
                              rng.integers(9), 4])
            obs, r, term, trunc, _ = env.step(a)
        t_steps = (time.monotonic() - t0) / n_steps
        print(f"[ra] episode {ep}: reset={t_reset:6.1f}s  step={t_steps:.2f}s  "
              f"segment={rid}", flush=True)
    env.close()
    print("[ra] PROBE-OK", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
