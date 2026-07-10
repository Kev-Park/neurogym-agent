"""Localize the per-GPU throughput gap (we're ~23 sps/GPU vs legacy ~30).

Single env (M=1, no contention) so we measure pure pipeline latency, broken
into: browser step (apply action + render + screenshot + state gather) vs DINO
encode vs the state-ready poll. Run on a vulkan node.
"""

from __future__ import annotations

import time

import numpy as np

import ngllib
from ngllib_agent.env_build import load_config
from ngllib_agent.obs import get_dino_encoder
from ngllib_agent.providers import FlywireSkeletonProvider
from ngllib_agent.rewards import ZRewardConfig, make_z_reward_factory, make_z_termination_factory
from ngllib_agent.wrappers import ActionSpec, MultiDiscreteActionWrapper, split_panes

N = 40


def build_base(cfg):
    ec, rc = cfg["env"], cfg["reward"]
    rcfg = ZRewardConfig(**{k: rc[k] for k in ("z_tolerance", "success", "z_shaping_coef", "step_penalty")})
    base = ngllib.Environment(
        headless=True, renderer="gpu", orientation="euler",
        left_pane=True, right_pane=True, image_size=None,   # dino config: full 2-pane render
        reset_state_provider=FlywireSkeletonProvider(ec["parquet_path"]),
        reward_factory=make_z_reward_factory(rcfg),
        termination_factory=make_z_termination_factory(rcfg),
    )
    ac = cfg["action"]
    x0, y0, x1, y1 = ac["pane_3d_bounds"]
    spec = ActionSpec(grid_rows=ac["grid_rows"], grid_cols=ac["grid_cols"],
                      pane_x0=x0, pane_y0=y0, pane_x1=x1, pane_y1=y1,
                      rotation_bins_per_axis=ac["rotation_bins_per_axis"],
                      rotation_step_rad=ac["rotation_step_rad"],
                      zoom_bins=ac["zoom_bins"], zoom_step=ac["zoom_step"])
    return MultiDiscreteActionWrapper(base, spec)


def main() -> int:
    cfg = load_config("configs/ppo_zmax_navigate.yaml")
    env = build_base(cfg)
    enc = get_dino_encoder()
    obs, _ = env.reset(seed=0)
    print(f"[timing] warm; image shape={np.asarray(obs['image']).shape}", flush=True)

    t_step, t_dino, img_shape = [], [], None
    for i in range(N):
        a = env.action_space.sample()
        t0 = time.perf_counter()
        obs, r, term, trunc, info = env.step(a)
        t1 = time.perf_counter()
        img = np.asarray(obs["image"])
        img_shape = img.shape
        left, right = split_panes(img)
        t2 = time.perf_counter()
        enc.encode([left, right])
        t3 = time.perf_counter()
        t_step.append((t1 - t0) * 1000)
        t_dino.append((t3 - t2) * 1000)
        if term or trunc:
            obs, _ = env.reset()

    def stats(x):
        x = sorted(x)
        return f"mean={np.mean(x):.0f}ms median={x[len(x)//2]:.0f}ms p90={x[int(len(x)*0.9)]:.0f}ms"

    print(f"[timing] rendered image shape: {img_shape}", flush=True)
    print(f"[timing] browser step (apply+render+screenshot+state): {stats(t_step)}", flush=True)
    print(f"[timing] DINO encode (2 panes):                        {stats(t_dino)}", flush=True)

    # Drill into the browser step: time ngllib internals directly.
    base = env.unwrapped
    from ngllib_agent.wrappers import decode
    spec = env._action_spec
    ta, tss, tjs = [], [], []
    for i in range(N):
        act = decode(env.action_space.sample(), spec, orient_dim=3)
        t0 = time.perf_counter(); base._apply_actions(act); t1 = time.perf_counter()
        base._get_screenshot(); t2 = time.perf_counter()
        base._get_json_state(); t3 = time.perf_counter()
        ta.append((t1 - t0) * 1000); tss.append((t2 - t1) * 1000); tjs.append((t3 - t2) * 1000)
    print(f"[timing]   apply_action:  {stats(ta)}", flush=True)
    print(f"[timing]   screenshot:    {stats(tss)}", flush=True)
    print(f"[timing]   state gather:  {stats(tjs)}", flush=True)
    tot = np.mean(t_step) + np.mean(t_dino)
    print(f"[timing] total/step ~{tot:.0f}ms -> single-env ceiling ~{1000/tot:.1f} steps/s", flush=True)
    print(f"[timing] step breakdown: browser={np.mean(t_step)/tot*100:.0f}% dino={np.mean(t_dino)/tot*100:.0f}%", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
