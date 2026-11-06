"""Build the wrapped ngllib Environment from a config dict.

Shared by the sanity loop and the PPO smoke so both construct the env identically.
Imports `ngllib` lazily -- only when actually building an env.

`env.backend` selects the renderer: `chrome` (default; Playwright + Chromium,
the deployment target) or `simulator` (CloudVolume + moderngl/EGL). The
pre-seam spellings `browser` / `native` are read as the same two, so the run
configs that produced existing checkpoints still build.
"""

from __future__ import annotations

from typing import Any

from ngllib.dataset import DatasetSpec

from .tasks import build_task
from .data import resolve_task_config, validate_scene
from .wrappers import (
    ActionSpec,
    DinoObservationWrapper,
    MultiDiscreteActionWrapper,
    PosStateWrapper,
    ResilientStepWrapper,
)

BACKENDS = {"chrome": "chrome", "browser": "chrome", "simulator": "simulator", "native": "simulator"}


def dino_layout(ec, oc):
    """Observation config owns DINO panes and preprocessing resolution."""
    left = bool(oc.get("use_left_pane", True))
    if "left_pane" in ec and bool(ec["left_pane"]) != left:
        raise ValueError("env.left_pane conflicts with obs.use_left_pane in DINO mode")
    if ec.get("right_pane") is False:
        raise ValueError("DINO mode requires env.right_pane=true")
    if ec.get("image_size") is not None:
        raise ValueError("DINO mode requires env.image_size=null; set obs.dino.input_size for preprocessing")
    return {**ec, "left_pane": left, "right_pane": True, "image_size": None,
            "capture_scale": ec.get("capture_scale", 0.5)}


def action_spec_from_config(ac: dict[str, Any]) -> ActionSpec:
    # `click_bounds` supersedes `pane_3d_bounds`: clicks may address either
    # pane now that the simulator matches Chrome on the 2D pane too. The old
    # key is still read so existing run configs stay reproducible.
    x0, y0, x1, y1 = ac.get("click_bounds") or ac["pane_3d_bounds"]
    # action.verbs sizes the policy head (3 = every checkpoint before the
    # double-click verb, 4 = with it, 5 = with the 2D pane's own zoom). Legacy
    # configs carry `pane_3d_bounds` and no `verbs`; they are 3-verb by
    # construction.
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
        **({"xs_zoom_step": float(ac["xs_zoom_step"])} if "xs_zoom_step" in ac else {}),
        zoom_step=ac["zoom_step"],
        split_click_heads=bool(ac.get("split_click_heads", False)),
    )


def build_env(cfg: dict[str, Any], first_episode_limit: int | None = None, frame_callback=None, task_factory=None):
    """Construct `TimeLimit(MultiDiscreteActionWrapper(ngllib.Environment))`."""
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

    cfg, manifest = resolve_task_config(cfg, custom_factory=task_factory is not None)
    ec, ac = cfg["env"], cfg["action"]
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
        ec = dino_layout(ec, oc)

    task = build_task(cfg, manifest, task_factory)

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
    if "start_url" in ec:
        layout_start_url = ec["start_url"]
    else:
        layout_start_url = None

    if ec.get("image_layer") or ec.get("segmentation_layer"):
        from ngllib.dataset import default_start_url, select_viewer_layers, split_state_url, state_to_url

        host, state = split_state_url(layout_start_url or default_start_url())
        state = select_viewer_layers(state, image_layer=ec.get("image_layer"),
                                     segmentation_layer=ec.get("segmentation_layer"))
        layout_start_url = state_to_url(host, state)

    if backend == "simulator":
        # env.pane_mode: the 2D-pane fill policy (atomic | progressive |
        # concurrent | random); ngllib's default is the shipping `atomic`.
        sim_kwargs = dict(cache_dir=ec.get("cv_cache"))
        if "pane_mode" in ec:
            sim_kwargs["pane_mode"] = ec["pane_mode"]
        if cfg.get("action", {}).get("cell_bounded_clicks", False):
            ac = cfg["action"]
            sim_kwargs["click_grid"] = (ac["grid_rows"], ac["grid_cols"])
            sim_kwargs["click_bounds"] = tuple(ac["click_bounds"])
        if layout_start_url:
            sim_kwargs["dataset"] = DatasetSpec.from_start_url(layout_start_url)
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
        # env.viewer: which Neuroglancer build renders -- a path, "packaged"
        # (the default, shipped inside ngllib) or "hosted". env.storage_state:
        # an explicit Playwright credential file, otherwise a middleauth start
        # URL is seeded from the CAVE secret CloudVolume already reads.
        for k in ("browser_restart_every", "retry_on_reset", "recovery_mode",
                  "clear_cache_on_recycle", "extra_launch_args", "state_ready_timeout_s",
                  "viewer", "storage_state", "cave_secret",
                  # PNG capture: parity gates must not measure JPEG artefacts
                  # (2D IoU 0.859 JPEG vs 0.914 PNG on the same frames).
                  "screenshot_format"):
            if k in ec:
                chrome_kwargs[k] = ec[k]
        if layout_start_url:
            chrome_kwargs["start_url"] = layout_start_url
        renderer = ChromeRenderer(**chrome_kwargs)
        # M5 reset-ahead (2026-08): pre-navigate the next episode in a warm
        # context off the critical path; reset swaps pages instead of paying
        # navigate+settle. Off unless the config asks.
        env_kwargs = {}
        for k in ("reset_ahead", "reset_ahead_after_steps"):
            if k in ec:
                env_kwargs[k] = ec[k]

    validate_scene(manifest, renderer.dataset)
    env = Environment(
        backend=renderer,
        orientation=ec.get("orientation", "euler"),
        reset_state_provider=task.reset_state_provider,
        reward_factory=task.reward_factory,
        termination_factory=task.termination_factory,
        task_metadata_provider=task.metadata_provider,
        **env_kwargs,
    )
    if frame_callback is not None:
        import gymnasium as gym

        class FrameTap(gym.ObservationWrapper):
            def observation(self, obs):
                frame_callback(obs["image"])
                return obs

        env = FrameTap(env)
    env = MultiDiscreteActionWrapper(env, action_spec_from_config(ac))
    if cfg.get('diagnostics', {}).get('click_credit', False):
        from .click_credit import ClickOutcomeWrapper
        env = ClickOutcomeWrapper(env, cfg['model']['click_credit_directory'])
    return _wrap_obs_and_limits(env, cfg, first_episode_limit)


def _wrap_obs_and_limits(env, cfg: dict[str, Any], first_episode_limit: int | None):
    """Obs-mode + resilient + TimeLimit (+ stagger) stack shared by both
    backends."""
    import gymnasium as gym

    ec, oc = cfg["env"], cfg.get("obs", {})
    obs_mode = oc.get("mode", "raw")

    # zmax-left diagnostics: BELOW the obs-mode wrappers so it sees the raw
    # ngllib obs (position + segments) and the verb-indexed action vector.
    if cfg.get("reward", {}).get("mode") == "z_free":
        from .wrappers.zmaxleft_diag import ZmaxLeftDiagWrapper

        env = ZmaxLeftDiagWrapper(env)

    # Observation mode (agent_plan.md §10/Round 8). Applied under the resilient
    # wrapper so glitch-truncation returns an already-transformed obs.
    scale = oc.get("pos_state_scale")
    if obs_mode == "dino":
        dc = oc.get("dino", {})
        from .obs import get_dino_encoder  # torch import stays lazy

        encoder = get_dino_encoder(
            model_name=dc.get("model_name", "dinov2_vits14"),
            input_size=dc.get("input_size", 224),
            device=dc.get("device"),
            # Throughput levers (all default off -> bitwise-identical eager path):
            #   cuda_graph = bitwise-safe dispatch collapse (adoptable);
            #   compile/fp16 = numerics-drifting (DYNAMICS-GATED, needs a transfer
            #   check before adoption); noop = R_cap probe (zero features).
            use_cuda_graph=bool(dc.get("cuda_graph", False)),
            use_compile=bool(dc.get("compile", False)),
            use_fp16=bool(dc.get("fp16", False)),
            use_noop=bool(dc.get("noop", False)),
            # patch_tokens: the obs becomes the raw 16x16x384 DINO patch-token
            # grid per pane (obs.patch_features, fp16) instead of the pooled
            # CLS vector; the RLModule's conv trunk owns the first learned layer.
            patch_tokens=bool(dc.get("patch_tokens", False)),
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
    if cfg.get('diagnostics', {}).get('boundary_credit', False):
        from .boundary_credit import BoundaryInfoWrapper
        env = BoundaryInfoWrapper(env)
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
        cfg = yaml.safe_load(f)
    if cfg.get("task", {}).get("manifest"):
        from pathlib import Path

        cfg["task"]["manifest"] = str((Path(path).resolve().parent / cfg["task"]["manifest"]).resolve())
    return cfg
