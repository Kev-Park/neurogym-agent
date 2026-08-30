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

    # canvas staleness under a click-heavy stream (the service-only effect)
    base = svc_env
    while hasattr(base, "env"):
        base = base.env
    base = getattr(base, "unwrapped", base)
    svc_env.reset()
    stale = fresh = 0
    for i in range(40):
        a = (np.array([0, int(rng.integers(1024)), 4, 4, 4, 4]) if i % 2 == 0
             else np.array([1, 0, int(rng.integers(9)), int(rng.integers(9)),
                            int(rng.integers(9)), 4]))
        svc_env.step(a)
        key = base._tile_state_key() if hasattr(base, "_tile_state_key") else None
        if key is not None and key == getattr(base, "_svc_key", None):
            fresh += 1
        else:
            stale += 1
    print(f"[obs] canvas freshness over 40 steps: fresh={fresh} stale={stale}",
          flush=True)
    svc_env.close(); loc_env.close()
    print("[obs] PROBE-OK", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
