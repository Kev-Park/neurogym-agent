"""PPO smoke — Ray RLlib new API stack (>=2.40).

Confirms the wrapped ngllib env trains under PPO (loss moves). Needs a real
browser -> run on a vulkan-capable GPU node under SLURM.

    # Milestone 1 (single-process, env in driver):
    uv run python scripts/ppo_smoke.py --iters 2 --train-batch-size 128
    # Milestone 2 (N remote env runners, each with its own Chrome):
    uv run python scripts/ppo_smoke.py --iters 5 --train-batch-size 512 --num-env-runners 2

Simplifications (replaced in later milestones):
- Observation reduced to the pos-state vector (position/xs_scale/orientation/
  proj_scale) so PPO's default RLModule needs no CNN/Dict handling. DINO image
  obs comes later.
- Learner on CPU (tiny MLP); the browser still uses the GPU for rendering.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# Ray >=2.43 auto-ships the CWD as a runtime_env working_dir when launched under
# `uv run`; here that's the 1.2GB repo (checkpoints/wandb/parquet) and it blows
# past the 512MB limit. The smoke is single-process (0 remote actors), so disable
# it. Multi-node milestones will set an explicit runtime_env / shared FS instead.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


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
    ap.add_argument(
        "--num-env-runners",
        type=int,
        default=0,
        help="0 = M1 (env in driver); >0 = M2 (N remote Ray-actor env runners, each with its own Chrome).",
    )
    ap.add_argument(
        "--browser-restart-every",
        type=int,
        default=None,
        help="Override env.browser_restart_every (default in ngllib = 90). "
             "Set to a small value (e.g. 5) in extended tests to exercise the "
             "Playwright refresh mechanism.",
    )
    args = ap.parse_args()

    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    from ngllib_agent.env_build import build_env, load_config

    cfg = load_config(args.config)
    if args.browser_restart_every is not None:
        cfg["env"]["browser_restart_every"] = args.browser_restart_every
    pc = cfg.get("ppo", {})

    register_env("ngl-znav", lambda env_config: _pos_only_wrapper(build_env(cfg)))

    ray.init(include_dashboard=False, log_to_driver=True)
    config = (
        PPOConfig()
        .environment("ngl-znav")
        .framework("torch")
        .env_runners(
            num_env_runners=args.num_env_runners,
            rollout_fragment_length="auto",
            # Chrome renders via Vulkan (ICD), not CUDA, so remote runners
            # don't need a Ray GPU allocation to run the browser. Learner
            # (still in driver) doesn't need CUDA either at this stage.
            num_gpus_per_env_runner=0,
        )
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
    algo = config.build_algo() if hasattr(config, "build_algo") else config.build()

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
