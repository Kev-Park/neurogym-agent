"""Pure env-stepping throughput (no Ray, no PPO) — isolates the env pipeline.

Builds a ThreadedVectorEnv of M browsers and steps it, reporting aggregate sps.
  Q1 (M ceiling):  run at M=32/40/48 single-GPU.
  Q2 (multi-GPU):  run TWO instances on one 2-GPU node (CUDA_VISIBLE_DEVICES=0
                   and =1) simultaneously; if each still ~single-GPU rate, the
                   node isn't the wall (RLlib coordination is) — else it is.

    uv run --no-sync python scripts/probe_throughput.py <M> <N_steps>
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
    tag = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

    cfg = load_config("configs/ppo_zmax_navigate.yaml")
    cfg.setdefault("obs", {})["mode"] = "dino"
    venv = make_env_creator(cfg, vector_mode="threads")({"num_envs": M})

    rng = np.random.default_rng(0)

    def acts():
        # batched MultiDiscrete sample for all M envs
        return np.stack([venv.single_action_space.sample() for _ in range(M)])

    venv.reset(seed=0)
    for _ in range(8):  # warm all browsers past cold start
        venv.step(acts())
    print(f"[thru] M={M} gpu={tag} warm; measuring {N} vector-steps...", flush=True)

    steps = 0
    t0 = time.time()
    for _ in range(N):
        venv.step(acts())
        steps += M
    dt = time.time() - t0
    print(f"[thru] RESULT M={M} gpu={tag} sps={steps/dt:.1f} "
          f"(env_steps={steps} in {dt:.0f}s, per_env={steps/dt/M:.2f})", flush=True)
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
