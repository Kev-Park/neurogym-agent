"""Milestone 1 sanity loop: reset + step the wrapped ngllib env end to end.

Runs a real browser -> needs a vulkan-capable GPU node (submit under SLURM).
Verifies the provider/factories/action-wrapper wire together and that reward +
z tracking are sane. Not a training run.

    uv run python scripts/sanity_env_loop.py [--config configs/ppo_zmax_navigate.yaml] [--steps 10]
"""

from __future__ import annotations

import argparse

import numpy as np

from ngllib_agent.env_build import build_env, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo_zmax_navigate.yaml")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    env = build_env(cfg)
    try:
        obs, info = env.reset(seed=args.seed)
        print("reset ok. obs keys:", sorted(obs.keys()))
        print("task_info:", info.get("task_info"))
        z_max = info["task_info"]["z_max"]
        print(f"z_max={z_max:.1f}  z0={float(obs['position'][2]):.1f}")

        rng = np.random.default_rng(args.seed)
        total = 0.0
        for t in range(args.steps):
            action = _rng_action(env, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            z = float(obs["position"][2])
            print(
                f"step {t:2d} a={list(action)} r={reward:+.4f} "
                f"z={z:8.1f} d(z,zmax)={abs(z - z_max):8.1f} "
                f"term={terminated} trunc={truncated}"
            )
            if terminated or truncated:
                obs, info = env.reset()
                z_max = info["task_info"]["z_max"]
        print(f"OK: {args.steps} steps, total_reward={total:+.4f}")
    finally:
        env.close()


def _rng_action(env, rng):
    # env.action_space is MultiDiscrete; sample with our own rng for reproducibility
    return np.array([int(rng.integers(n)) for n in env.action_space.nvec], dtype=np.int64)


if __name__ == "__main__":
    main()
