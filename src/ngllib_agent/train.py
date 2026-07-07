"""PPO training entry point (agent_plan.md §12 `train.py` rewrite).

Config-driven RLlib PPO on the wrapped ngllib env with wandb logging and async
atomic checkpointing (§14c). Runs standalone (local Ray) or against an existing
cluster via RAY_ADDRESS (the coordinator's coord_learner.sh path).

    uv run python -m ngllib_agent.train \
        --run-name dist-rl-v1 --iters 250 \
        --num-env-runners 3 --num-envs-per-env-runner 8 \
        --num-gpus-per-env-runner 0.6 \
        --train-batch-size 3072 --rollout-fragment-length 128 \
        --checkpoint-dir /scratch/kp0374/checkpoints/dist-rl-v1

Resume: `--resume` loads the latest `ckpt_*.pkl` from --checkpoint-dir and
continues the same wandb run (id kept in meta.json).
"""

from __future__ import annotations

import argparse
import numbers
import os
import time

# Ray auto-ships the CWD as a runtime_env under `uv run` (1.2GB repo) — see
# ppo_smoke.py. Must be set before ray is imported.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ngllib_agent.train")
    ap.add_argument("--config", default="configs/ppo_zmax_navigate.yaml")
    ap.add_argument("--run-name", default=f"znav-{int(time.time())}")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--obs", choices=["pos", "dino"], default="dino")
    # Scale / placement
    ap.add_argument("--num-env-runners", type=int, default=0)
    ap.add_argument("--num-envs-per-env-runner", type=int, default=1)
    ap.add_argument("--num-gpus-per-env-runner", type=float, default=0.0)
    # Sampling
    ap.add_argument("--train-batch-size", type=int, default=None,
                    help="Defaults to config ppo.train_batch_size.")
    ap.add_argument("--rollout-fragment-length", default="auto")
    ap.add_argument("--sample-timeout-s", type=float, default=600.0)
    # Checkpoint / resume
    ap.add_argument("--checkpoint-dir", default=None,
                    help="Defaults to checkpoints/<run-name> under CWD.")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    # Logging
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                    default="online")
    ap.add_argument("--wandb-project", default="neurogym-agent")
    return ap


def _scalars(d: dict, prefix: str) -> dict:
    """Flatten numeric leaves one level deep for wandb."""
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, numbers.Number):
            out[f"{prefix}/{k}"] = float(v)
    return out


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)

    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    import wandb

    from .distributed.checkpoint import (
        AsyncCheckpointer,
        atomic_json,
        latest_checkpoint,
        load_checkpoint,
    )
    from .env_build import load_config, make_env_creator

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = args.obs
    pc = cfg.get("ppo", {})
    train_batch = args.train_batch_size or pc.get("train_batch_size", 2000)
    ckpt_dir = args.checkpoint_dir or os.path.join("checkpoints", args.run_name)

    register_env("ngl-znav", make_env_creator(cfg))

    ray.init(include_dashboard=False, log_to_driver=True, ignore_reinit_error=True)

    vectorize_mode = (
        "sync" if args.num_envs_per_env_runner <= 1 else "vector_entry_point"
    )
    config = (
        PPOConfig()
        .environment("ngl-znav")
        .framework("torch")
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=args.num_envs_per_env_runner,
            gym_env_vectorize_mode=vectorize_mode,
            num_gpus_per_env_runner=args.num_gpus_per_env_runner,
            rollout_fragment_length=(
                int(args.rollout_fragment_length)
                if str(args.rollout_fragment_length).isdigit()
                else args.rollout_fragment_length
            ),
            sample_timeout_s=args.sample_timeout_s,
        )
        .learners(num_learners=0)  # learner in the driver (small MLP, CPU)
        .training(
            train_batch_size=train_batch,
            minibatch_size=min(pc.get("sgd_minibatch_size", 256), train_batch),
            num_epochs=pc.get("num_sgd_iter", 4),
            gamma=pc.get("gamma", 0.99),
            lambda_=pc.get("lambda", 0.95),
            clip_param=pc.get("clip_param", 0.2),
            lr=pc.get("lr", 3.0e-4),
            kl_target=pc.get("kl_target", 0.01),
            # float or [[timestep, value], ...] schedule (see config comment)
            entropy_coeff=pc.get("entropy_coeff", 0.0),
        )
    )
    if args.obs == "dino":
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec

        from .policies import HierarchicalPPOModule

        config = config.rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=HierarchicalPPOModule,
                model_config=cfg.get("model", {}),
            )
        )

    algo = config.build_algo() if hasattr(config, "build_algo") else config.build()

    # ---- resume ------------------------------------------------------------
    wandb_id = None
    if args.resume:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is None:
            print(f"[train] --resume but no checkpoint in {ckpt_dir}; starting fresh")
        else:
            print(f"[train] resuming from {ckpt}")
            algo.set_state(load_checkpoint(ckpt))
            meta_path = os.path.join(ckpt_dir, "meta.json")
            if os.path.exists(meta_path):
                import json

                wandb_id = json.load(open(meta_path)).get("wandb_id")

    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        id=wandb_id,
        resume="allow" if wandb_id else None,
        mode=args.wandb_mode,
        config={**cfg, "cli": vars(args)},
    )
    checkpointer = AsyncCheckpointer(ckpt_dir, every=args.checkpoint_every)

    # ---- train loop ----------------------------------------------------------
    try:
        for _ in range(args.iters):
            result = algo.train()
            it = int(result.get("training_iteration", 0))
            er = result.get("env_runners", {}) or {}
            pol = (result.get("learners", {}) or {}).get("default_policy", {}) or {}
            t_iter = result.get("time_this_iter_s")
            n_steps = er.get("num_env_steps_sampled")

            metrics = {
                **_scalars(er, "env_runners"),
                **_scalars(pol, "learner"),
                "perf/time_this_iter_s": t_iter,
                "perf/steps_per_s": (n_steps / t_iter) if (t_iter and n_steps) else None,
            }
            wandb.log({k: v for k, v in metrics.items() if v is not None}, step=it)
            # Progress heartbeat every iteration — the coordinator's
            # --target-iterations completion check reads this (REFINEMENT R1).
            atomic_json(
                {"iteration": it, "wandb_id": run.id, "run_name": args.run_name},
                os.path.join(ckpt_dir, "meta.json"),
            )
            checkpointer.maybe_save(algo, it)
            print(
                f"iter {it}: return_mean={er.get('episode_return_mean')} "
                f"steps={n_steps} t={t_iter and round(t_iter, 1)}s "
                f"sps={n_steps and t_iter and round(n_steps / t_iter, 1)} "
                f"loss={pol.get('total_loss')}",
                flush=True,
            )
    finally:
        checkpointer.finalize()
        wandb.finish()
        algo.stop()
        ray.shutdown()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
