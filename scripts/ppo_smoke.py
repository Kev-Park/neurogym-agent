"""PPO smoke — Ray RLlib new API stack (>=2.40).

Confirms the wrapped ngllib env trains under PPO (loss moves). Needs a real
browser -> run on a vulkan-capable GPU node under SLURM.

    # Milestone 1 (single-process, env in driver):
    uv run python scripts/ppo_smoke.py --iters 2 --train-batch-size 128
    # Milestone 2 (N remote env runners, each with its own Chrome):
    uv run python scripts/ppo_smoke.py --iters 5 --train-batch-size 512 --num-env-runners 2

Observation modes (--obs, see configs `obs:` section):
- pos  — scaled pos-state vector + RLlib default MLP module (infra smoke).
- dino — env-side frozen DINO split-pane features + HierarchicalPPOModule
         (real observation/policy). Pass --num-gpus-per-env-runner 0.25 so the
         runner actor gets CUDA for the encoder.
Learner stays on CPU (small MLPs either way); the browser uses the GPU via Vulkan.
"""

from __future__ import annotations

import argparse
import os

# Ray >=2.43 auto-ships the CWD as a runtime_env working_dir when launched under
# `uv run`; here that's the 1.2GB repo (checkpoints/wandb/parquet) and it blows
# past the 512MB limit. The smoke is single-process (0 remote actors), so disable
# it. Multi-node milestones will set an explicit runtime_env / shared FS instead.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


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
    ap.add_argument(
        "--sample-timeout-s",
        type=float,
        default=600.0,
        help="Max seconds RLlib waits for a rollout fragment before giving up "
             "(reported as 'No samples returned from remote workers' and produces "
             "nan iter metrics). Browser stepping is ~0.5s/step, so a 256-step "
             "fragment needs ~128s ideal; 4-5x headroom accommodates env glitches. "
             "Default 600s is deliberately generous — reduce for faster failure "
             "detection when tuning.",
    )
    ap.add_argument(
        "--rollout-fragment-length",
        default="auto",
        help="Per-worker per-fragment sample count. 'auto' = train_batch_size / "
             "num_env_runners. Small integer values (e.g. 64) trade throughput "
             "for lower per-fragment latency (helps under high glitch rate).",
    )
    ap.add_argument(
        "--obs",
        choices=["pos", "dino"],
        default=None,
        help="Override config obs.mode. 'dino' = env-side frozen DINO features + "
             "HierarchicalPPOModule (real observation/policy); 'pos' = pos-state "
             "vector + default RLModule (infra smoke).",
    )
    ap.add_argument(
        "--num-gpus-per-env-runner",
        type=float,
        default=0.0,
        help="Ray GPU share per env runner. Needed >0 in dino mode so the runner "
             "actor gets CUDA visibility for the env-side DINO encoder (e.g. 0.25). "
             "Chrome's Vulkan rendering works regardless of Ray's GPU accounting.",
    )
    args = ap.parse_args()

    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    from ngllib_agent.env_build import build_env, load_config

    cfg = load_config(args.config)
    if args.browser_restart_every is not None:
        cfg["env"]["browser_restart_every"] = args.browser_restart_every
    cfg.setdefault("obs", {}).setdefault("mode", "pos")  # this script never uses raw
    if args.obs is not None:
        cfg["obs"]["mode"] = args.obs
    obs_mode = cfg["obs"]["mode"]
    pc = cfg.get("ppo", {})

    register_env("ngl-znav", lambda env_config: build_env(cfg))

    ray.init(include_dashboard=False, log_to_driver=True)
    config = (
        PPOConfig()
        .environment("ngl-znav")
        .framework("torch")
        .env_runners(
            num_env_runners=args.num_env_runners,
            rollout_fragment_length=(
                int(args.rollout_fragment_length)
                if str(args.rollout_fragment_length).isdigit()
                else args.rollout_fragment_length
            ),
            # Chrome renders via Vulkan (ICD), not CUDA — a Ray GPU share is only
            # needed for the env-side DINO encoder (dino mode; pass e.g. 0.25).
            num_gpus_per_env_runner=args.num_gpus_per_env_runner,
            # Browser stepping (~0.5s/step) makes even short rollout fragments
            # slower than RLlib's default sample_timeout_s (60s). Without this
            # override every iter reports nan for the full training window
            # after the first ~30 min (see stress test 2026-07-02 for the
            # symptom pattern). Generous 600s default; tune down when needed.
            sample_timeout_s=args.sample_timeout_s,
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
    if obs_mode == "dino":
        # Real observation/policy: gated hierarchical RLModule over DINO features.
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec

        from ngllib_agent.policies import HierarchicalPPOModule

        config = config.rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=HierarchicalPPOModule,
                model_config=cfg.get("model", {}),
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
