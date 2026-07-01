"""Milestone 1 single-process PPO smoke — Ray RLlib new API stack (>=2.40).

Confirms the wrapped ngllib env trains under PPO (loss moves) with a single local
env runner. Needs a real browser -> run on a vulkan-capable GPU node under SLURM.

    uv run python scripts/ppo_smoke.py [--config ...] [--iters 3] [--train-batch-size 256]

Smoke simplifications (replaced in later milestones):
- Observation reduced to the pos-state vector (position/xs_scale/orientation/
  proj_scale) so PPO's default RLModule needs no CNN/Dict handling. DINO image
  obs comes later.
- Learner on CPU (tiny MLP); the browser still uses the GPU for rendering.
"""

from __future__ import annotations

import argparse

import numpy as np


def _pos_only_wrapper(env):
    """Dict obs -> flat Box(pos_state). Keeps sanity_env_loop on the real Dict."""
    import gymnasium as gym
    from gymnasium import spaces

    space = env.observation_space
    dim = sum(
        int(np.prod(space[k].shape))
        for k in ("position", "xs_scale", "orientation", "proj_scale")
    )

    class _PosOnly(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(dim,), dtype=np.float32
            )

        def observation(self, obs):
            return np.concatenate(
                [
                    np.asarray(obs["position"], np.float32).ravel(),
                    np.asarray(obs["xs_scale"], np.float32).ravel(),
                    np.asarray(obs["orientation"], np.float32).ravel(),
                    np.asarray(obs["proj_scale"], np.float32).ravel(),
                ]
            ).astype(np.float32)

    return _PosOnly(env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo_zmax_navigate.yaml")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--train-batch-size", type=int, default=256)
    args = ap.parse_args()

    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    from ngllib_agent.env_build import build_env, load_config

    cfg = load_config(args.config)
    pc = cfg.get("ppo", {})

    register_env("ngl-znav", lambda env_config: _pos_only_wrapper(build_env(cfg)))

    ray.init(include_dashboard=False, log_to_driver=True)
    config = (
        PPOConfig()
        .environment("ngl-znav")
        .framework("torch")
        .env_runners(num_env_runners=0, rollout_fragment_length="auto")
        .learners(num_learners=0)  # learner in the driver process (CPU)
        .training(
            train_batch_size=args.train_batch_size,
            minibatch_size=min(pc.get("sgd_minibatch_size", 64), args.train_batch_size),
            num_epochs=pc.get("num_sgd_iter", 4),
            gamma=pc.get("gamma", 0.99),
            lambda_=pc.get("lambda", 0.95),
            clip_param=pc.get("clip_param", 0.2),
            lr=pc.get("lr", 3.0e-4),
            kl_target=pc.get("kl_target", 0.01),
        )
    )
    algo = config.build_algo()

    for i in range(args.iters):
        result = algo.train()
        if i == 0:
            print("result top-level keys:", sorted(result.keys()))
        er = result.get("env_runners", {}) or {}
        learners = result.get("learners", {}) or {}
        pol = learners.get("default_policy", {}) or {}
        print(
            f"iter {i}: "
            f"episode_return_mean={er.get('episode_return_mean')} "
            f"num_steps={er.get('num_env_steps_sampled')} "
            f"total_loss={pol.get('total_loss')} "
            f"policy_loss={pol.get('policy_loss')}"
        )

    print("PPO smoke complete.")
    ray.shutdown()


if __name__ == "__main__":
    main()
