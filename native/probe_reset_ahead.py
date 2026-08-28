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

    rot = np.array([1, 0, 5, 2, 4, 4])
    for ep in range(5):
        t0 = time.monotonic()
        obs, info = env.reset()
        t_reset = time.monotonic() - t0
        rid = info["task_info"].get("segment_id")
        t0 = time.monotonic()
        for _ in range(6):
            obs, r, term, trunc, _ = env.step(rot)
        t_steps = (time.monotonic() - t0) / 6
        print(f"[ra] episode {ep}: reset={t_reset:6.1f}s  step={t_steps:.2f}s  "
              f"segment={rid}", flush=True)
        # give the next episode's prefetch time to land, as a real episode would
        time.sleep(20)
    env.close()
    print("[ra] PROBE-OK", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
