"""Service-mode integration smoke: Ray service actor + client env end-to-end.

    uv run --no-sync python native/probe_service_smoke.py
"""

from __future__ import annotations

import os
import time

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def main() -> int:
    import ray

    from ngllib_agent.env_build import build_env, load_config
    from ngllib_agent.service_actor import create_render_services

    ray.init(include_dashboard=False, log_to_driver=True)
    cfg = load_config("configs/native_service.yaml")
    create_render_services(cfg)
    print("[svc-smoke] service up", flush=True)

    env = build_env(cfg)
    rng = np.random.default_rng(5)
    for ep in range(3):
        t0 = time.monotonic()
        obs, info = env.reset()
        t_reset = time.monotonic() - t0
        assert "image_features" in obs and obs["image_features"].shape == (768,)
        t0 = time.monotonic()
        ret = 0.0
        for i in range(20):
            a = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(a)
            ret += r
            if term or trunc:
                break
        t_step = (time.monotonic() - t0) / (i + 1)
        print(f"[svc-smoke] ep {ep}: reset={t_reset:.1f}s step={t_step:.3f}s "
              f"feat_norm={float(np.linalg.norm(obs['image_features'])):.1f} "
              f"return={ret:.3f}", flush=True)
    env.close()
    print("[svc-smoke] SMOKE-OK", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
