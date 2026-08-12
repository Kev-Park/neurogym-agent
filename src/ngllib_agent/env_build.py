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
        ec = {**ec, "left_pane": True, "right_pane": True, "image_size": None}

    provider = FlywireSkeletonProvider(ec["parquet_path"])
    rcfg = ZRewardConfig(
        z_tolerance=rc["z_tolerance"],
        success=rc["success"],
        z_shaping_coef=rc["z_shaping_coef"],
        step_penalty=rc["step_penalty"],
    )

    image_size = ec.get("image_size")
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
    env = Environment(**env_kwargs)

    env = MultiDiscreteActionWrapper(env, action_spec_from_config(ac))

    # Observation mode (agent_plan.md §10/Round 8). Applied under the resilient
    # wrapper so glitch-truncation returns an already-transformed obs.
    scale = oc.get("pos_state_scale")
    if obs_mode == "dino":
        from .obs import get_dino_encoder  # torch import stays lazy

        dc = oc.get("dino", {})
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
            fns = [
                (lambda lim=lim: build_env(cfg, first_episode_limit=lim))
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
