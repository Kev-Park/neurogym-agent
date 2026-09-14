"""Pure env-stepping throughput (no Ray, no PPO) — isolates the env pipeline.

Builds a ThreadedVectorEnv of M browsers and steps it, reporting aggregate sps.
  Q1 (M ceiling):  run at M=32/40/48 single-GPU.
  Q2 (multi-GPU):  run TWO instances on one 2-GPU node (CUDA_VISIBLE_DEVICES=0
                   and =1) simultaneously; if each still ~single-GPU rate, the
                   node isn't the wall (RLlib coordination is) — else it is.

    uv run --no-sync python scripts/probe_throughput.py <M> <N_steps> [capture_scale]

THRU_CONFIG selects the config (default configs/ppo_zmax_navigate.yaml, i.e.
Chrome; configs/native.yaml for the simulator). THRU_SECS, if set, measures for
that many seconds instead of a fixed N vector-steps, so shapes with very
different per-step costs get comparable sample sizes.

THRU_POLICY=<ckpt_*.pkl> acts with that trained policy (stochastic, one batched
forward per vector-step -- the way an RLlib EnvRunner does it) instead of random
actions. Random actions click constantly and refetch tiles, so they are
fetch-bound in a way trained z-nav policies are not; the M (envs/runner) gain
they show (3.3x at 32 runners, 2026-09-12) did NOT appear in real training
(148 vs 147 sps). Measure with the policy you will train with.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from ngllib_agent.env_build import load_config, make_env_creator


def main() -> int:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    tag = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

    cfg = load_config(os.environ.get("THRU_CONFIG", "configs/ppo_zmax_navigate.yaml"))
    cfg.setdefault("obs", {})["mode"] = "dino"
    if scale != 1.0:
        cfg.setdefault("env", {})["capture_scale"] = scale
    secs = float(os.environ.get("THRU_SECS", "0") or 0)
    if M == 1:
        # make_env_creator hands back a bare env for num_envs=1 (no
        # single_action_space); a 1-env ThreadedVectorEnv keeps the loop uniform
        # (the same trap cost the 32x1 shape in the August density grid).
        from ngllib_agent.env_build import build_env
        from ngllib_agent.vector_env import ThreadedVectorEnv

        venv = ThreadedVectorEnv([lambda: build_env(cfg)])
    else:
        venv = make_env_creator(cfg, vector_mode="threads")({"num_envs": M})

    rng = np.random.default_rng(0)
    pkl = os.environ.get("THRU_POLICY")
    policy_tag = "random"
    if pkl:
        import torch
        from ray.rllib.core.columns import Columns

        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from eval_d0 import StatePklPolicy  # rebuilds the module from the pickle

        torch.manual_seed(0)
        pol = StatePklPolicy(pkl, venv.envs[0] if hasattr(venv, "envs") else venv,
                             cfg.get("model", {}), stochastic=True)
        policy_tag = "trained"
        last_obs = {}

        def acts():
            if not last_obs:
                return np.stack([venv.single_action_space.sample() for _ in range(M)])
            batch = {Columns.OBS: {k: torch.from_numpy(np.asarray(v, np.float32))
                                   for k, v in last_obs.items()}}
            with torch.no_grad():
                out = pol.module.forward_inference(batch)
            dist = pol.dist_cls.from_logits(out[Columns.ACTION_DIST_INPUTS])
            return dist.sample().cpu().numpy()
    else:
        def acts():
            # batched MultiDiscrete sample for all M envs
            return np.stack([venv.single_action_space.sample() for _ in range(M)])

    # Vector reset isn't retry-guarded (ResilientStepWrapper only guards step),
    # and a cold-start thundering herd can fail one browser's navigation.
    # Retry the reset+warmup a few times before giving up.
    for attempt in range(5):
        try:
            obs, _ = venv.reset(seed=0)
            if pkl:
                last_obs = obs
            for _ in range(8):  # warm all browsers past cold start
                obs = venv.step(acts())[0]
                if pkl:
                    last_obs = obs
            break
        except Exception as e:
            print(f"[thru] warmup attempt {attempt} failed: {type(e).__name__}: "
                  f"{str(e)[:80]}; retrying", flush=True)
            time.sleep(5)
    else:
        print(f"[thru] RESULT M={M} gpu={tag} sps=FAILED (could not warm up)", flush=True)
        venv.close()
        return 1
    print(f"[thru] M={M} gpu={tag} warm; measuring {N} vector-steps...", flush=True)

    steps = 0
    t0 = time.time()
    n = 0
    while (n < N) if not secs else (time.time() - t0 < secs):
        obs = venv.step(acts())[0]
        if pkl:
            last_obs = obs
        steps += M
        n += 1
    dt = time.time() - t0
    print(f"[thru] RESULT M={M} scale={scale} gpu={tag} policy={policy_tag} sps={steps/dt:.1f} "
          f"(env_steps={steps} in {dt:.0f}s, per_env={steps/dt/M:.2f})", flush=True)
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
