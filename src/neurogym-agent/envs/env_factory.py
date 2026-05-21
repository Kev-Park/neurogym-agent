from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _THIS_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from envs.action_translator import ActionSpec
from envs.browser_manager import BrowserManager
from envs.ngl_gym_env import NGLGymEnv
from envs.reward import RewardConfig


def build_env_factory(
    cfg: dict,
    segment_data: dict[str, np.ndarray],
    segment_ids: list[str],
    browser_manager: BrowserManager,
):
    """Return a no-arg callable that creates a monitored NGLGymEnv.

    segment_data and browser_manager are shared across all worker threads in-process —
    no pickling, no shared memory, no semaphores needed.
    """
    env_cfg = cfg["env"]

    action_spec = ActionSpec(
        grid_rows=env_cfg["click_grid_rows"],
        grid_cols=env_cfg["click_grid_cols"],
        pane_x0=env_cfg["pane_3d_bounds"][0],
        pane_y0=env_cfg["pane_3d_bounds"][1],
        pane_x1=env_cfg["pane_3d_bounds"][2],
        pane_y1=env_cfg["pane_3d_bounds"][3],
        rotation_bins_per_axis=env_cfg["rotation_bins_per_axis"],
        rotation_step_rad=env_cfg["rotation_step_rad"],
    )
    reward_cfg = RewardConfig(
        z_tolerance=env_cfg["z_tolerance"],
        success=env_cfg["reward_success"],
        z_shaping_coef=env_cfg.get("z_shaping_coef", 0.001),
        step_penalty=env_cfg.get("step_penalty", 0.0),
    )

    def _make():
        env = NGLGymEnv(
            neurogym_config_path=env_cfg["neurogym_config_path"],
            segment_data=segment_data,
            segment_ids=segment_ids,
            action_spec=action_spec,
            reward_cfg=reward_cfg,
            browser_manager=browser_manager,
            max_episode_steps=env_cfg["max_episode_steps"],
            reset_rotation_perturb_rad=env_cfg["reset_rotation_perturb_rad"],
            reset_zoom_perturb_frac=env_cfg["reset_zoom_perturb_frac"],
            headless=env_cfg.get("headless", True),
        )
        return gym.wrappers.RecordEpisodeStatistics(env)

    return _make
