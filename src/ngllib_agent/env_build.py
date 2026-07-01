"""Build the wrapped ngllib Environment from a config dict.

Shared by the sanity loop and the PPO smoke so both construct the env identically.
Imports `ngllib` (and thus Playwright) lazily — only when actually building an env.
"""

from __future__ import annotations

from typing import Any

from .providers import FlywireSkeletonProvider
from .rewards import ZRewardConfig, make_z_reward_factory, make_z_termination_factory
from .wrappers import ActionSpec, MultiDiscreteActionWrapper


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


def build_env(cfg: dict[str, Any]):
    """Construct `TimeLimit(MultiDiscreteActionWrapper(ngllib.Environment))`."""
    import gymnasium as gym

    from ngllib import Environment

    ec, ac, rc = cfg["env"], cfg["action"], cfg["reward"]

    provider = FlywireSkeletonProvider(ec["parquet_path"])
    rcfg = ZRewardConfig(
        z_tolerance=rc["z_tolerance"],
        success=rc["success"],
        z_shaping_coef=rc["z_shaping_coef"],
        step_penalty=rc["step_penalty"],
    )

    image_size = ec.get("image_size")
    env = Environment(
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

    env = MultiDiscreteActionWrapper(env, action_spec_from_config(ac))
    env = gym.wrappers.TimeLimit(env, max_episode_steps=ec.get("max_episode_steps", 300))
    return env


def load_config(path: str) -> dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)
