"""Build the wrapped ngllib Environment from a config dict.

Shared by the sanity loop and the PPO smoke so both construct the env identically.
Imports `ngllib` (and thus Playwright) lazily — only when actually building an env.
"""

from __future__ import annotations

from typing import Any

from .providers import FlywireSkeletonProvider
from .rewards import ZRewardConfig, make_z_reward_factory, make_z_termination_factory
from .wrappers import (
    ActionSpec,
    DinoObservationWrapper,
    MultiDiscreteActionWrapper,
    PosStateWrapper,
    ResilientStepWrapper,
    ServiceFeaturesWrapper,
)


def action_spec_from_config(ac: dict[str, Any]) -> ActionSpec:
    x0, y0, x1, y1 = ac["pane_3d_bounds"]
    return ActionSpec(
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


def build_env(cfg: dict[str, Any], first_episode_limit: int | None = None):
    """Construct `TimeLimit(MultiDiscreteActionWrapper(ngllib.Environment))`."""
    import gymnasium as gym
    import logging

    from ngllib import Environment

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
        # obs.use_left_pane: false trains on the 3D pane ONLY. The 2D EM
        # pane is not task-essential but IS task-correlated, so feeding it
        # lets the policy depend on the 384 dims that differ most between
        # the simulator and Chrome (and whose de-sync is unmanaged today).
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

    # Backend switch (native-renderer branch): env.backend == "native" builds
    # the browser-free ngllib.native.NativeEnvironment (CloudVolume +
    # moderngl/EGL) instead of Playwright+Chrome. Same obs/action contract;
    # the browser-lifecycle kwargs below have no native counterpart.
    if ec.get("backend", "browser") == "native":
        from ngllib.native.environment import NativeEnvironment

        # env.render_service: true -> this env is a service CLIENT: no GL,
        # no DINO in the runner; states go to the per-node render service
        # (created by train.py / eval drivers via create_render_services).
        svc_factory = None
        if ec.get("render_service"):
            from .service_actor import service_factory as svc_factory  # noqa: F811

        env = NativeEnvironment(
            orientation=ec.get("orientation", "euler"),
            left_pane=ec.get("left_pane", False),
            right_pane=ec.get("right_pane", True),
            image_size=tuple(image_size) if image_size else None,
            capture_scale=ec.get("capture_scale", 0.5),
            reset_state_provider=provider,
            reward_factory=make_z_reward_factory(rcfg),
            termination_factory=make_z_termination_factory(rcfg),
            cache_dir=ec.get("cv_cache"),
            reset_ahead=ec.get("reset_ahead", True),
            render_service=svc_factory,
            service_feature_dim=int(
                oc.get("dino", {}).get("feature_dim", 384)),
        )
        env = MultiDiscreteActionWrapper(env, action_spec_from_config(ac))
        return _wrap_obs_and_limits(env, cfg, first_episode_limit)

    env_kwargs = dict(
        headless=ec.get("headless", True),
        renderer=ec.get("renderer", "gpu"),
        orientation=ec.get("orientation", "euler"),
        left_pane=ec.get("left_pane", False),
        right_pane=ec.get("right_pane", True),
        image_size=tuple(image_size) if image_size else None,
        reset_state_provider=provider,
        reward_factory=make_z_reward_factory(rcfg),
        termination_factory=make_z_termination_factory(rcfg),
    )
    # Optional self-healing overrides — only pass if the config sets them, so
    # ngllib's defaults (browser_restart_every=90, retry_on_reset=3) apply
    # otherwise. Used by the extended smoke to force restart-mechanism firing.
    if "browser_restart_every" in ec:
        env_kwargs["browser_restart_every"] = ec["browser_restart_every"]
    if "retry_on_reset" in ec:
        env_kwargs["retry_on_reset"] = ec["retry_on_reset"]
    # Glitch-recovery strategy A/B (2026-08): 'escalate' (default, full browser
    # relaunch on repeated glitch) vs 'in_place' (cheap context recycle at the
    # source, legacy-style). Only passed if set so older ngllib without the kwarg
    # still builds.
    if "recovery_mode" in ec:
        env_kwargs["recovery_mode"] = ec["recovery_mode"]
    # M5 reset-ahead (2026-08): pre-navigate the next episode in a warm context
    # off the critical path; reset swaps pages instead of paying navigate+settle.
    if "reset_ahead" in ec:
        env_kwargs["reset_ahead"] = ec["reset_ahead"]
    if "reset_ahead_after_steps" in ec:
        env_kwargs["reset_ahead_after_steps"] = ec["reset_ahead_after_steps"]
    # Cycle-time levers (2026-08-16): browser-side downscaled capture, optional
    # per-episode HTTP-cache clear, extra Chrome flags (footprint experiments).
    for k in ("capture_scale", "clear_cache_on_recycle", "extra_launch_args",
              "state_ready_timeout_s"):
        if k in ec:
            env_kwargs[k] = ec[k]
    env = Environment(**env_kwargs)

    env = MultiDiscreteActionWrapper(env, action_spec_from_config(ac))
    return _wrap_obs_and_limits(env, cfg, first_episode_limit)


def _wrap_obs_and_limits(env, cfg: dict[str, Any], first_episode_limit: int | None):
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
        if "image_features" in getattr(env.observation_space, "spaces", {}):
            # Service-mode native env: features already encoded per-node;
            # same policy-facing Dict, no torch in this process.
            env = ServiceFeaturesWrapper(
                env, feature_dim=int(dc.get("feature_dim", 384)),
                pos_state_scale=scale)
        else:
            from .obs import get_dino_encoder  # torch import stays lazy

            encoder = get_dino_encoder(
                model_name=dc.get("model_name", "dinov2_vits14"),
                input_size=dc.get("input_size", 224),
                device=dc.get("device"),
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
        # M1a: evenly-spaced first-episode limits desynchronize TimeLimit
        # truncations. Spread across the NODE's envs (2 runners/GPU share a
        # node): runners interleave via worker_index parity, so the node's 2M
        # envs reset ~1 at a time instead of 16-at-once waves. Deterministic
        # spacing (not random) guarantees uniformity.
        limits: list[int | None] = [None] * max(num_envs, 1)
        if cfg.get("env", {}).get("stagger_first_episode") and num_envs > 1:
            max_steps = cfg["env"].get("max_episode_steps", 300)
            widx = int(getattr(env_config, "worker_index", 0) or 0)
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
                    (lambda lim=lim: build_env(cfg, first_episode_limit=lim))
                    if lim is not None
                    else (lambda: build_env(cfg))
                )
                for lim in limits
            ]
            if vector_mode == "threads":
                from .vector_env import ThreadedVectorEnv

                return ThreadedVectorEnv(fns)
            import gymnasium as gym

            return gym.vector.AsyncVectorEnv(fns, context="spawn")
        return build_env(cfg)

    return _creator


def load_config(path: str) -> dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)
