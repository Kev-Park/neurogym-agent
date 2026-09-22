"""Build the wrapped ngllib Environment from a config dict.

Shared by the sanity loop and the PPO smoke so both construct the env identically.
Imports `ngllib` lazily -- only when actually building an env.

`env.backend` selects the renderer: `chrome` (default; Playwright + Chromium,
the deployment target) or `simulator` (CloudVolume + moderngl/EGL). The
pre-seam spellings `browser` / `native` are read as the same two, so the run
configs that produced existing checkpoints still build.
"""

from __future__ import annotations

import os
from typing import Any

from .providers import FlywireSkeletonProvider
from .rewards import ZRewardConfig, make_z_reward_factory, make_z_termination_factory
from .wrappers import (
    ActionSpec,
    DinoObservationWrapper,
    MultiDiscreteActionWrapper,
    PosStateWrapper,
    ResilientStepWrapper,
)

BACKENDS = {"chrome": "chrome", "browser": "chrome", "simulator": "simulator", "native": "simulator"}


def action_spec_from_config(ac: dict[str, Any]) -> ActionSpec:
    # `click_bounds` supersedes `pane_3d_bounds`: clicks may address either
    # pane now that the simulator matches Chrome on the 2D pane too. The old
    # key is still read so existing run configs stay reproducible.
    x0, y0, x1, y1 = ac.get("click_bounds") or ac["pane_3d_bounds"]
    # action.verbs sizes the policy head (3 = every checkpoint before the
    # double-click verb, 4 = with it). Legacy configs carry `pane_3d_bounds`
    # and no `verbs`; they are 3-verb by construction.
    verbs = int(ac.get("verbs", 4 if "click_bounds" in ac else 3))
    return ActionSpec(
        verbs=verbs,
        grid_rows=ac["grid_rows"],
        grid_cols=ac["grid_cols"],
        pane_x0=x0,
        pane_y0=y0,
        pane_x1=x1,
        pane_y1=y1,
        rotation_bins_per_axis=ac["rotation_bins_per_axis"],
        rotation_step_rad=ac["rotation_step_rad"],
        zoom_bins=ac["zoom_bins"],
        zoom_step=ac["zoom_step"],
    )


def build_env(cfg: dict[str, Any], first_episode_limit: int | None = None,
              dino_server_index: int | None = None):
    """Construct `TimeLimit(MultiDiscreteActionWrapper(ngllib.Environment))`.

    `dino_server_index` (the runner's worker_index) routes DINO encoding to a
    shared `DinoServer` actor when obs.dino.server is enabled; ignored otherwise.
    """
    import logging

    from ngllib import ChromeRenderer, Environment, SimulatorRenderer

    # Configure basic logging so ngllib's INFO messages (browser restarts,
    # navigation retries) surface in the driver log via Ray's log_to_driver=True.
    # Idempotent if already configured.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        )

    ec, ac, rc = cfg["env"], cfg["action"], cfg["reward"]
    oc = cfg.get("obs", {})
    obs_mode = oc.get("mode", "raw")  # raw | pos | dino

    # DINO obs needs the full two-pane render at native resolution (the wrapper
    # splits EM|3D and resizes per pane) — derive these env settings from the mode
    # rather than trusting per-key config to stay consistent.
    if obs_mode == "dino":
        # capture_scale=0.5 default (2026-08-16): browser-side GPU downscale —
        # panes 450² still ≫ DINO's 224² input, visually pristine, and +31%
        # aggregate sps (M=16 A/B: 26.5 -> 34.8; single-env step 99 -> 49ms).
        # Config env.capture_scale overrides.
        # obs.use_left_pane: false trains on the 3D pane ONLY. Default TRUE:
        # right-pane-only was tested 2026-08-31 and lost ~19pp on Chrome
        # (67.5% vs 86-87.5%) while leaving the sim2real gap unchanged, so
        # the 2D EM pane carries real signal -- plausibly a z cue, since the
        # cross-section's appearance varies with depth and the task is
        # z-navigation. Keep both panes unless re-testing that ablation.
        _use_left = bool(oc.get("use_left_pane", True))
        ec = {**ec, "left_pane": _use_left, "right_pane": True,
              "image_size": None,
              "capture_scale": ec.get("capture_scale", 0.5)}

    # env.holdout_parquet: a frozen eval pool whose root_ids are EXCLUDED from
    # training resets, so eval measures unseen-neuron generalization. The eval
    # CLI resets with explicit states and is unaffected.
    exclude = None
    if ec.get("holdout_parquet"):
        import pyarrow.parquet as pq

        exclude = [
            str(r) for r in
            pq.read_table(ec["holdout_parquet"], columns=["root_id"])
            .column("root_id").to_pylist()
        ]
    psr = ec.get("projection_scale_range")
    provider = FlywireSkeletonProvider(
        ec["parquet_path"],
        projection_scale_range=tuple(psr) if psr else None,
        spawn_curriculum=ec.get("spawn_curriculum"),
        exclude_root_ids=exclude,
    )
    rcfg = ZRewardConfig(
        z_tolerance=rc["z_tolerance"],
        success=rc["success"],
        z_shaping_coef=rc["z_shaping_coef"],
        step_penalty=rc["step_penalty"],
        z_tolerance_frac=rc.get("z_tolerance_frac"),
    )

    image_size = ec.get("image_size")
    backend = BACKENDS.get(str(ec.get("backend", "chrome")))
    if backend is None:
        raise ValueError(
            f"env.backend must be chrome|simulator (or the older browser|native); "
            f"got {ec.get('backend')!r}")
    layout = dict(
        left_pane=ec.get("left_pane", False),
        right_pane=ec.get("right_pane", True),
        image_size=tuple(image_size) if image_size else None,
    )
    if "capture_scale" in ec:
        layout["capture_scale"] = ec["capture_scale"]

    if backend == "simulator":
        # env.pane_mode: the 2D-pane fill policy (atomic | progressive |
        # concurrent | random); ngllib's default is the shipping `atomic`.
        sim_kwargs = dict(cache_dir=ec.get("cv_cache"))
        if "pane_mode" in ec:
            sim_kwargs["pane_mode"] = ec["pane_mode"]
        # On-GPU feed: each enabled GL pane stays in VRAM (no CPU readback).
        #   server + cuda_ipc  -> cross-process CUDA-IPC payloads to a DINO server
        #   cuda_local (no srv)-> RAW CUDA tensors to THIS process's DINO encoder
        #                         (encode_gpu; no server, no IPC handle).
        # Both need a small --num-gpus-per-env-runner for the interop context.
        _dino = oc.get("dino") or {}
        _srv = _dino.get("server") or {}
        server_ipc = bool(_srv.get("enabled") and _srv.get("cuda_ipc"))
        local_gpu = bool(_dino.get("cuda_local") and not _srv.get("enabled"))
        if server_ipc or local_gpu:
            sim_kwargs["cuda_ipc"] = True
            sim_kwargs["ipc_export"] = server_ipc  # False => raw tensor, in-process
        # render_batch (throughput-scaling): batch the process's envs' 3D renders
        # through one shared GL context (atlas) instead of one context per env.
        # render_batch_size MUST be >= num_envs_per_env_runner (the atlas cell
        # count); the bench sets it equal. Pairs with cuda_local for the all-VRAM
        # arm (interop) or plain readback otherwise.
        if ec.get("render_batch"):
            sim_kwargs["render_batch"] = True
            # NGL_RENDER_BATCH_SIZE (set by the bench = num_envs_per_env_runner)
            # sizes the atlas; falls back to the config value.
            sim_kwargs["render_batch_size"] = int(os.environ.get(
                "NGL_RENDER_BATCH_SIZE", ec.get("render_batch_size", 1)))
        renderer = SimulatorRenderer(**layout, **sim_kwargs)
        # The simulator has always defaulted to reset-ahead prefetch (its
        # warm work is a background fetch, free to start immediately).
        env_kwargs = dict(reset_ahead=ec.get("reset_ahead", True))
    else:
        chrome_kwargs = dict(
            headless=ec.get("headless", True),
            renderer=ec.get("renderer", "gpu"),
            **layout,
        )
        # Optional self-healing overrides — only pass if the config sets them, so
        # ngllib's defaults (browser_restart_every=90, retry_on_reset=3) apply
        # otherwise. Used by the extended smoke to force restart-mechanism firing.
        # recovery_mode: 'escalate' (default, full browser relaunch on repeated
        # glitch) vs 'in_place' (cheap context recycle at the source).
        # Cycle-time levers (2026-08-16): optional per-episode HTTP-cache clear,
        # extra Chrome flags (footprint experiments).
        for k in ("browser_restart_every", "retry_on_reset", "recovery_mode",
                  "clear_cache_on_recycle", "extra_launch_args", "state_ready_timeout_s"):
            if k in ec:
                chrome_kwargs[k] = ec[k]
        renderer = ChromeRenderer(**chrome_kwargs)
        # M5 reset-ahead (2026-08): pre-navigate the next episode in a warm
        # context off the critical path; reset swaps pages instead of paying
        # navigate+settle. Off unless the config asks.
        env_kwargs = {}
        for k in ("reset_ahead", "reset_ahead_after_steps"):
            if k in ec:
                env_kwargs[k] = ec[k]

    env = Environment(
        backend=renderer,
        orientation=ec.get("orientation", "euler"),
        reset_state_provider=provider,
        reward_factory=make_z_reward_factory(rcfg),
        termination_factory=make_z_termination_factory(rcfg),
        **env_kwargs,
    )
    env = MultiDiscreteActionWrapper(env, action_spec_from_config(ac))
    return _wrap_obs_and_limits(env, cfg, first_episode_limit, dino_server_index)


def _wrap_obs_and_limits(env, cfg: dict[str, Any], first_episode_limit: int | None,
                         dino_server_index: int | None = None):
    """Obs-mode + resilient + TimeLimit (+ stagger) stack shared by both
    backends."""
    import gymnasium as gym

    ec, oc = cfg["env"], cfg.get("obs", {})
    obs_mode = oc.get("mode", "raw")

    # Observation mode (agent_plan.md §10/Round 8). Applied under the resilient
    # wrapper so glitch-truncation returns an already-transformed obs.
    scale = oc.get("pos_state_scale")
    if obs_mode == "dino":
        dc = oc.get("dino", {})
        server_cfg = dc.get("server") or {}
        if server_cfg.get("enabled"):
            # Route encoding to a shared DinoServer actor (dino-server
            # experiment). Runner worker_index % M picks the server; envs that
            # share a runner share a server, so their concurrent encode() calls
            # batch together. feature_dim can be given to avoid a startup RPC.
            from .obs.dino_server import DinoServerClient

            m = max(1, int(server_cfg.get("instances", 1)))
            idx = int(dino_server_index or 0) % m
            encoder = DinoServerClient(idx, feature_dim=server_cfg.get("feature_dim"))
        else:
            from .obs import get_dino_encoder  # torch import stays lazy

            encoder = get_dino_encoder(
                model_name=dc.get("model_name", "dinov2_vits14"),
                input_size=dc.get("input_size", 224),
                device=dc.get("device"),
                use_cuda_graph=bool(dc.get("cuda_graph", False)),
                use_noop=bool(dc.get("noop", False)),
            )
        env = DinoObservationWrapper(env, encoder, pos_state_scale=scale)
    elif obs_mode == "pos":
        env = PosStateWrapper(env, pos_state_scale=scale)
    elif obs_mode != "raw":
        raise ValueError(f"obs.mode must be raw|pos|dino; got {obs_mode!r}")

    env = ResilientStepWrapper(env)  # truncate on transient viewer/browser glitches
    max_steps = ec.get("max_episode_steps", 300)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    # M1a episode-boundary stagger (2026-08): a shorter FIRST episode permanently
    # offsets this env's truncation cycle, so a vector's envs don't reset in
    # synchronized waves. Outermost so it can force truncation before TimeLimit.
    if first_episode_limit is not None and first_episode_limit < max_steps:
        from .wrappers import FirstEpisodeStagger

        env = FirstEpisodeStagger(env, first_episode_limit)
    return env


def make_env_creator(cfg: dict[str, Any], vector_mode: str = "spawn"):
    """RLlib env creator supporting the vector_entry_point path.

    RLlib registers callable envs with a vector entry point that passes
    `num_envs` in env_config. For M>1 we build the vector env ourselves; use
    with `gym_env_vectorize_mode="vector_entry_point"`.

    vector_mode:
      "spawn"   — AsyncVectorEnv, fresh interpreter per env (own CUDA context +
                  Chrome). Gym's plain 'async' FORKS: forked children inherit
                  torch/CUDA/playwright state and deadlock in reset (2026-07-03).
      "threads" — ThreadedVectorEnv (R4): one process, M browser threads, ONE
                  CUDA context + ONE shared DINO. Density-oriented topology.
    """
    if vector_mode not in ("spawn", "threads"):
        raise ValueError(f"vector_mode must be spawn|threads; got {vector_mode!r}")

    def _creator(env_config: dict[str, Any] | None = None):
        env_config = env_config or {}
        num_envs = int(env_config.get("num_envs") or 0)
        # worker_index (EnvContext attr) both staggers resets AND, when the DINO
        # server is enabled, routes this runner's envs to server worker_index % M.
        widx = int(getattr(env_config, "worker_index", 0) or 0)
        # M1a: evenly-spaced first-episode limits desynchronize TimeLimit
        # truncations. Spread across the NODE's envs (2 runners/GPU share a
        # node): runners interleave via worker_index parity, so the node's 2M
        # envs reset ~1 at a time instead of 16-at-once waves. Deterministic
        # spacing (not random) guarantees uniformity.
        limits: list[int | None] = [None] * max(num_envs, 1)
        if cfg.get("env", {}).get("stagger_first_episode") and num_envs > 1:
            max_steps = cfg["env"].get("max_episode_steps", 300)
            spacing = max_steps / (2 * num_envs)
            limits = [
                max(5, max_steps - round((2 * i + (widx % 2)) * spacing))
                for i in range(num_envs)
            ]
        if num_envs > 1:
            # Only pass the kwarg when staggering is on — keeps build_env's
            # plain (cfg) call signature for other callers/tests.
            fns = [
                (
                    (lambda lim=lim: build_env(cfg, first_episode_limit=lim,
                                               dino_server_index=widx))
                    if lim is not None
                    else (lambda: build_env(cfg, dino_server_index=widx))
                )
                for lim in limits
            ]
            if vector_mode == "threads":
                from .vector_env import ThreadedVectorEnv

                return ThreadedVectorEnv(fns)
            import gymnasium as gym

            return gym.vector.AsyncVectorEnv(fns, context="spawn")
        return build_env(cfg, dino_server_index=widx)

    return _creator


def load_config(path: str) -> dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)
