"""Do service-mode observations equal local-mode observations?

native-v9-test (per-env local renderer) transfers to Chrome at 96.5%;
native-v9-plane (per-node service, same plane, same iters) at 86.8%, while
both score 99.5% on native. Either the service's OBS differ from local's,
or the difference is training dynamics (curriculum pacing). This measures
the first directly: identical states through both paths, compared in
FEATURE space (what the policy actually consumes).

Reported per state:
  cos_left / cos_right : cosine similarity of the per-pane DINO features
  l2rel                : ||svc - local|| / ||local|| over the 768-dim obs
  stale_steps          : how many steps the service's 2D canvas lagged
                         (fresh=0) under a click-heavy action stream

    uv run --no-sync python native/probe_obs_equivalence.py
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ["RAY_ADDRESS"] = "local"


def main() -> int:
    import ray

    from ngllib_agent.env_build import build_env, load_config
    from ngllib_agent.obs import get_dino_encoder
    from ngllib_agent.providers import FlywireSkeletonProvider
    from ngllib_agent.service_actor import create_render_services

    svc_cfg = load_config("configs/native_service.yaml")
    ray.init(include_dashboard=False, log_to_driver=False)
    create_render_services(svc_cfg)
    svc_env = build_env(svc_cfg)

    loc_cfg = load_config("configs/native.yaml")
    loc_cfg.setdefault("obs", {})["mode"] = "raw"
    loc_cfg["env"].update({"image_size": None, "left_pane": True,
                           "right_pane": True, "capture_scale": 0.5})
    loc_env = build_env(loc_cfg)
    enc = get_dino_encoder()

    ec = svc_cfg["env"]
    provider = FlywireSkeletonProvider(ec["parquet_path"])
    rng = np.random.default_rng(17)

    cosl, cosr, rel = [], [], []
    for k in range(6):
        state, ti = provider(rng, None)
        so, _ = svc_env.reset(options={"state": state, "task_info": ti})
        lo, _ = loc_env.reset(options={"state": state, "task_info": ti})
        img = lo["image"] if isinstance(lo, dict) and "image" in lo else None
        if img is None:
            print("[obs] local env returned no image; check obs mode")
            return 1
        left, right = img[:, :450], img[:, 450:]
        f_loc = enc.encode([left, right]).reshape(-1)
        f_svc = np.asarray(so["image_features"], np.float32)
        d = f_loc.shape[0] // 2

        def cos(a, b):
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        cl, cr = cos(f_loc[:d], f_svc[:d]), cos(f_loc[d:], f_svc[d:])
        r = float(np.linalg.norm(f_svc - f_loc) / (np.linalg.norm(f_loc) + 1e-9))
        cosl.append(cl); cosr.append(cr); rel.append(r)
        print(f"[obs] state {k}: cos_left={cl:.4f} cos_right={cr:.4f} "
              f"l2rel={r:.4f}", flush=True)

    print(f"[obs] MEDIAN cos_left={np.median(cosl):.4f} "
          f"cos_right={np.median(cosr):.4f} l2rel={np.median(rel):.4f}",
          flush=True)

    # 2D-pane staleness under a click-heavy stream, BOTH paths (local is the
    # control: without it a service number is unreadable).
    def unwrap(e):
        while hasattr(e, "env"):
            e = e.env
        return getattr(e, "unwrapped", e)

    def freshness(env, attr, steps=40, seed=20260903):
        """Both arms MUST see the same action stream: a "fresh" step is one
        where the tile key did not change, and only clicks move position, so
        a rotation-heavy stream trivially scores 40/40. Sharing one rng
        across the two calls gave them DIFFERENT streams and made service
        swing 2..40 run to run while local sat at 0..8 -- unpaired and
        uninterpretable. Seed per call so the comparison is paired."""
        r = np.random.default_rng(seed)
        base = unwrap(env)
        env.reset()
        fresh = 0
        for i in range(steps):
            a = (np.array([0, int(r.integers(1024)), 4, 4, 4, 4])
                 if i % 2 == 0 else
                 np.array([1, 0, int(r.integers(9)), int(r.integers(9)),
                           int(r.integers(9)), 4]))
            env.step(a)
            if base._tile_state_key() == getattr(base, attr, None):
                fresh += 1
        return fresh, steps - fresh

    sf, ss = freshness(svc_env, "_svc_key")
    lf, ls = freshness(loc_env, "_tile_key")
    print(f"[obs] 2D-pane freshness over 40 click-heavy steps: "
          f"SERVICE fresh={sf} stale={ss} | LOCAL fresh={lf} stale={ls}",
          flush=True)
    svc_env.close(); loc_env.close()
    print("[obs] PROBE-OK", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
