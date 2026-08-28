"""PPO training entry point (agent_plan.md §12 `train.py` rewrite).

Config-driven RLlib PPO on the wrapped ngllib env with wandb logging and async
atomic checkpointing (§14c). Runs standalone (local Ray) or against an existing
cluster via RAY_ADDRESS (the coordinator's coord_learner.sh path).

    uv run python -m ngllib_agent.train \
        --run-name dist-rl-v1 --iters 250 \
        --train-batch-size 3072 --rollout-fragment-length 128 \
        --checkpoint-dir /scratch/kp0374/checkpoints/dist-rl-v1

Env-runner topology defaults to the PINNED 2-process x 16-thread/GPU config
(see the --num-env-runners note below); for N renderer nodes pass
`--num-env-runners <2*N>`.

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
# Fresh local cluster unless the coordinator/caller supplies a real address —
# stale /tmp/ray/ray_current_cluster markers on previously-used nodes make a
# plain ray.init() join a dead head and hang (see ppo_smoke.py, 2026-07-07).
os.environ.setdefault("RAY_ADDRESS", "local")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ngllib_agent.train")
    ap.add_argument("--config", default="configs/ppo_zmax_navigate.yaml")
    ap.add_argument("--run-name", default=f"znav-{int(time.time())}")
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--obs", choices=["pos", "dino"], default="dino")
    # Scale / placement.
    # PINNED default topology (2026-07-11, REFINEMENT.md R4-frontier): per GPU
    # node run 2 EnvRunner PROCESSES x 16 threaded envs each on a shared GPU.
    # Rationale: multi-process beats single-process threading ~+44% (escapes the
    # GIL on JPEG-decode + DINO-prep), and the per-node full-step ceiling (~35-40
    # sps) is host-side, NOT capture (raw capture does ~170/GPU) — so 2 procs
    # saturates the node and more GPUs/procs don't help. Multi-node: scale
    # --num-env-runners = 2 x (renderer nodes), keep the rest.
    ap.add_argument("--render-service", action="store_true",
                    help="Start one per-node render+encode service actor "
                         "(requires env.render_service: true in the config; "
                         "runners then need no GPU).")
    ap.add_argument("--learner-gpu", action="store_true",
                    help="Run the (driver-local) learner's update on the GPU "
                         "instead of CPU — halves the synchronous PPO cycle "
                         "when sampling is fast (native renderer).")
    ap.add_argument("--num-env-runners", type=int, default=2)
    ap.add_argument("--num-cpus-per-env-runner", type=float, default=1.0,
                    help="Ray CPU reservation per runner. Service-mode "
                         "runners need no GPU, so this is what forces them "
                         "to SPREAD across renderer nodes instead of "
                         "packing onto the head.")
    ap.add_argument("--num-envs-per-env-runner", type=int, default=16)
    ap.add_argument("--num-gpus-per-env-runner", type=float, default=0.5)
    ap.add_argument("--vector", choices=["spawn", "threads"], default="threads",
                    help="M>1 topology: process-per-env vs ThreadedVectorEnv (R4). "
                         "PINNED: threads (see --num-env-runners note).")
    # Sampling
    ap.add_argument("--train-batch-size", type=int, default=None,
                    help="Defaults to config ppo.train_batch_size.")
    ap.add_argument("--rollout-fragment-length", default="auto")
    ap.add_argument("--sample-timeout-s", type=float, default=600.0)
    ap.add_argument("--recovery-mode", choices=["escalate", "in_place"], default=None,
                    help="ngllib glitch-recovery strategy A/B. None = config/ngllib "
                         "default (escalate).")
    # DEFAULT ON since the 2026-08-16 mitigation sweep: stagger alone won —
    # mean 101.1±2.1 sps vs base 96.5±2.7, worst-seed 98.2 vs 92.9, stragglers
    # 8.5%->3.1%, reset waves 63-85 -> 8-13 resets/10s. Zero steady-state cost.
    ap.add_argument("--stagger-first-episode", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="M1a: evenly-spaced shorter FIRST episodes desynchronize "
                         "TimeLimit truncations across each node's envs (kills the "
                         "synchronized reset waves). One-time cost, no steady tax.")
    ap.add_argument("--reset-ahead", action="store_true",
                    help="M5: pre-navigate the next episode in a warm browser "
                         "context while the current one steps; reset swaps pages "
                         "instead of paying navigate+settle on the critical path.")
    # R10: coord-test-v7 showed RLlib will run indefinitely at ~15% throughput
    # while an EnvRunner restart churns (100 min at ~360s/iter vs 40s cruise).
    # On sustained degradation: force a checkpoint and exit 43 so the
    # coordinator respawns the workload from it (~5 min, the proven cure).
    # Non-coordinator launchers without a relaunch loop should pass
    # --no-degraded-exit or accept the early end (the run was wasting
    # walltime anyway; --resume continues it).
    ap.add_argument("--degraded-exit", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Exit 43 (checkpoint first) when the median of the "
                         "last 5 iteration times exceeds 3x the run median "
                         "(floor 180s). Capped at 3 exits/2h via meta.json — "
                         "degradation that survives full restarts is "
                         "environmental and restarting only burns progress.")
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

    from .degradation import DegradationDetector
    from .distributed.checkpoint import (
        AsyncCheckpointer,
        atomic_json,
        atomic_pickle,
        latest_checkpoint,
        load_checkpoint,
    )
    from .env_build import load_config, make_env_creator

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = args.obs
    if args.recovery_mode:
        cfg.setdefault("env", {})["recovery_mode"] = args.recovery_mode
    if args.stagger_first_episode:
        cfg.setdefault("env", {})["stagger_first_episode"] = True
    if args.reset_ahead:
        cfg.setdefault("env", {})["reset_ahead"] = True
    pc = cfg.get("ppo", {})
    train_batch = args.train_batch_size or pc.get("train_batch_size", 2000)
    ckpt_dir = args.checkpoint_dir or os.path.join("checkpoints", args.run_name)

    register_env("ngl-znav", make_env_creator(cfg, vector_mode=args.vector))

    ray.init(include_dashboard=False, log_to_driver=True, ignore_reinit_error=True)

    if args.render_service:
        from .service_actor import create_render_services

        create_render_services(cfg)

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
            num_cpus_per_env_runner=args.num_cpus_per_env_runner,
            rollout_fragment_length=(
                int(args.rollout_fragment_length)
                if str(args.rollout_fragment_length).isdigit()
                else args.rollout_fragment_length
            ),
            sample_timeout_s=args.sample_timeout_s,
        )
        # Learner stays LOGICALLY separated (local learner in the driver;
        # weights broadcast to runners each iter) — --learner-gpu only moves
        # its arithmetic to the GPU. PPO here is synchronous, so every
        # sampler idles during the update; on CPU that phase matched the
        # sampling phase (~50% duty cycle, native run 870742 measured 47 of
        # 116 available sps). Browser-era default stays CPU.
        .learners(num_learners=0,
                  num_gpus_per_learner=1 if args.learner_gpu else 0)
        # Escalation ladder for browser glitches at 32-browsers/GPU density:
        #   transient  -> absorbed by ResilientStepWrapper (step truncates, reset
        #                 retries 3x) so they never reach RLlib.
        #   persistent -> a runner whose browsers stay broken (observed: leaked
        #                 VRAM after the watchdog SIGKILLs hung Chromes -> new
        #                 browsers can't get a GPU context -> ALL 16 fail reset).
        # For persistent failure we must restart the whole EnvRunner *actor*
        # (fresh PROCESS frees the leaked VRAM). restart_failed_sub_environments
        # =True recreated the vector-env IN-PLACE, which can't free VRAM and spun
        # in an infinite recreate loop (job 845537 stalled iter 180 @ 2 sps). So:
        # keep it False and let restart_failed_env_runners do the process restart.
        .fault_tolerance(
            restart_failed_sub_environments=False,
            restart_failed_env_runners=True,
        )
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
    import json

    meta_path = os.path.join(ckpt_dir, "meta.json")
    prior_meta = {}
    if os.path.exists(meta_path):
        try:
            prior_meta = json.load(open(meta_path))
        except (json.JSONDecodeError, OSError):
            prior_meta = {}

    wandb_id = None
    if args.resume:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is None:
            print(f"[train] --resume but no checkpoint in {ckpt_dir}; starting fresh")
        else:
            print(f"[train] resuming from {ckpt}")
            algo.set_state(load_checkpoint(ckpt))
            wandb_id = prior_meta.get("wandb_id")

    # R10 degraded-throughput exit: detector + cross-restart exit cap. The cap
    # lives in meta.json because each exit is a fresh process — if 3 exits in
    # 2h haven't cured it, the cause is environmental (co-tenant load, sick
    # node) and further restarts just burn progress.
    degraded_exits = list(prior_meta.get("degraded_exits", []))
    detector = None
    if args.degraded_exit:
        recent_exits = [t for t in degraded_exits if time.time() - t < 7200]
        if len(recent_exits) >= 3:
            print(
                f"[train] degraded-exit DISABLED: {len(recent_exits)} exits in 2h "
                "did not cure the degradation — environmental cause suspected; "
                "running through it",
                flush=True,
            )
        else:
            detector = DegradationDetector()

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
            # Iteration-boundary marker (absolute wall ts) for aligning storm
            # onsets against learner/iteration boundaries in the event logs.
            print(f"MARK iter_start ts={time.time():.3f}", flush=True)
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
            # --target-iterations completion check and its progress-stall
            # timeout both read this (REFINEMENT R1/R10).
            atomic_json(
                {
                    "iteration": it,
                    "wandb_id": run.id,
                    "run_name": args.run_name,
                    "degraded_exits": degraded_exits,
                },
                meta_path,
            )
            checkpointer.maybe_save(algo, it)
            if args.checkpoint_every and it % args.checkpoint_every == 0:
                print(f"MARK checkpoint it={it} ts={time.time():.3f}", flush=True)
            # Healthy-runner count on the iter line: v7 forensics had to
            # reconstruct "an actor was down for 100 min" from scattered
            # actor-manager warnings. Never worth failing an iteration over.
            try:
                healthy = algo.env_runner_group.num_healthy_remote_workers()
            except Exception:
                healthy = None
            print(
                f"iter {it}: return_mean={er.get('episode_return_mean')} "
                f"steps={n_steps} t={t_iter and round(t_iter, 1)}s "
                f"sps={n_steps and t_iter and round(n_steps / t_iter, 1)} "
                f"loss={pol.get('total_loss')} "
                f"H={healthy if healthy is not None else '?'}/{args.num_env_runners} "
                f"ts={time.time():.3f}",
                flush=True,
            )
            if detector is not None and t_iter and detector.observe(t_iter):
                degraded_exits.append(time.time())
                print(
                    f"MARK degraded_exit it={it} t_iter={t_iter:.1f}s "
                    f"baseline={detector.baseline_s:.1f}s "
                    f"n_exits={len(degraded_exits)} ts={time.time():.3f}",
                    flush=True,
                )
                # Checkpoint synchronously (off-cadence iters would otherwise
                # lose up to checkpoint_every-1 iters of samples), persist the
                # exit record, and die so the coordinator respawns us fresh.
                atomic_pickle(
                    algo.get_state(),
                    os.path.join(ckpt_dir, f"ckpt_{it:06d}.pkl"),
                )
                atomic_json(
                    {
                        "iteration": it,
                        "wandb_id": run.id,
                        "run_name": args.run_name,
                        "degraded_exits": degraded_exits,
                    },
                    meta_path,
                )
                return 43
            # --iters is an ABSOLUTE target. RLlib's training_iteration
            # resumes from the checkpoint, but this loop used to count
            # RELATIVE iterations — a resumed run trained --iters MORE
            # (native-v9-test overran 740 -> 764+ before being caught).
            if it >= args.iters:
                print(f"MARK target_reached it={it} iters={args.iters} "
                      f"ts={time.time():.3f}", flush=True)
                break
    finally:
        checkpointer.finalize()
        wandb.finish()
        algo.stop()
        ray.shutdown()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
