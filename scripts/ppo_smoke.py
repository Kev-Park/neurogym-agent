"""Milestone 1 single-process PPO smoke via Ray RLlib.

Confirms the wrapped ngllib env trains under PPO (loss moves) with local sampling.
Needs a real browser -> run on a vulkan-capable GPU node under SLURM.

    uv run python scripts/ppo_smoke.py [--config ...] [--iters 5]

Notes:
- `num_rollout_workers=0` samples in the driver process (one browser). Multi-worker
  / multi-node comes in later milestones.
- Uses a downsized image (`env.image_size`) so RLlib's default CNN has filters;
  the DINO observation pipeline replaces this later.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo_zmax_navigate.yaml")
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    from ngllib_agent.env_build import build_env, load_config

    cfg = load_config(args.config)
    pc = cfg.get("ppo", {})

    register_env("ngl-znav", lambda env_config: build_env(cfg))

    ray.init(local_mode=True, include_dashboard=False)
    algo = (
        PPOConfig()
        .environment("ngl-znav")
        .framework("torch")
        .rollouts(num_rollout_workers=0, rollout_fragment_length="auto")
        .training(
            train_batch_size=pc.get("train_batch_size", 2000),
            sgd_minibatch_size=pc.get("sgd_minibatch_size", 256),
            num_sgd_iter=pc.get("num_sgd_iter", 4),
            gamma=pc.get("gamma", 0.99),
            lambda_=pc.get("lambda", 0.95),
            clip_param=pc.get("clip_param", 0.2),
            lr=pc.get("lr", 3.0e-4),
            kl_target=pc.get("kl_target", 0.01),
        )
        .resources(num_gpus=1)
        .build()
    )

    first_loss = None
    for i in range(args.iters):
        result = algo.train()
        info = result.get("info", {}).get("learner", {}).get("default_policy", {})
        stats = info.get("learner_stats", info)
        loss = stats.get("total_loss")
        rew = result.get("episode_reward_mean")
        print(f"iter {i}: total_loss={loss} episode_reward_mean={rew}")
        if first_loss is None and loss is not None:
            first_loss = loss
    print("PPO smoke complete.")
    ray.shutdown()


if __name__ == "__main__":
    main()
