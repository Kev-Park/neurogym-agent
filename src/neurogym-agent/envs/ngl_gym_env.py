from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_THIS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _THIS_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from envs.action_translator import ActionSpec, decode, sample_reset_perturbation
from envs.reward import RewardConfig, compute as compute_reward
from obs.dino_encoder import DinoEncoder
from ngllib import Environment


@dataclass
class StartLink:
    url: str
    z_max: float


def _read_nonempty_lines(path: str) -> list[str]:
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def load_start_links(urls_path: str, z_max_path: str) -> list[StartLink]:
    """Load start links from two parallel text files.

    `urls_path` has one Neuroglancer URL per line. `z_max_path` has one Z_max float
    per line in the same order. Blank lines and `#`-prefixed lines are ignored in
    both files. Counts must match.
    """
    urls = _read_nonempty_lines(urls_path)
    z_vals_raw = _read_nonempty_lines(z_max_path)
    if len(urls) != len(z_vals_raw):
        raise ValueError(
            f"Line count mismatch: {len(urls)} URLs in {urls_path} vs "
            f"{len(z_vals_raw)} values in {z_max_path}. Files must line up."
        )
    if not urls:
        raise ValueError(f"No start links found in {urls_path}")
    try:
        z_vals = [float(v) for v in z_vals_raw]
    except ValueError as e:
        raise ValueError(f"Non-float value in {z_max_path}: {e}") from e
    return [StartLink(url=u, z_max=z) for u, z in zip(urls, z_vals)]


class NGLGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        neurogym_config_path: str,
        start_links: list[StartLink],
        action_spec: ActionSpec,
        reward_cfg: RewardConfig,
        max_episode_steps: int = 300,
        reset_rotation_perturb_rad: float = 0.5,
        reset_zoom_perturb_frac: float = 0.25,
        headless: bool = False,
        dino_repo: str = "facebookresearch/dinov2",
        dino_model: str = "dinov2_vits14",
        dino_input_size: int = 224,
        dino_device: str | None = None,
    ):
        super().__init__()
        self._start_links = start_links
        self._action_spec = action_spec
        self._reward_cfg = reward_cfg
        self._max_episode_steps = max_episode_steps
        self._reset_rotation_perturb_rad = reset_rotation_perturb_rad
        self._reset_zoom_perturb_frac = reset_zoom_perturb_frac
        self._neuro_env = Environment(
            headless=headless,
            config_path=neurogym_config_path,
        )
        self._neuro_env.options = {"euler_angles": True, "resize": False, "fast": True}

        self._dino = DinoEncoder(
            repo=dino_repo,
            model_name=dino_model,
            input_size=dino_input_size,
            device=dino_device,
        )

        pos_dim = 3 + 1 + 3 + 1
        image_feature_dim = self._dino.feature_dim

        self.observation_space = spaces.Dict(
            {
                "image_features": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(image_feature_dim,), dtype=np.float32
                ),
                "pos_state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(pos_dim,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.MultiDiscrete(action_spec.multidiscrete_nvec())

        self._rng = np.random.default_rng()
        self._step_count = 0
        self._current_z_max: float | None = None
        self._last_image = None

    def _flatten_pos_state(self, pos_state: list) -> np.ndarray:
        position, cs_scale, orientation_euler, proj_scale = pos_state
        return np.asarray(
            list(position) + [cs_scale] + list(orientation_euler) + [proj_scale],
            dtype=np.float32,
        )

    def _encode_image(self, image: np.ndarray) -> np.ndarray:
        feats = self._dino.encode([image])
        return feats.reshape(-1).astype(np.float32)

    def _build_obs(self, state) -> dict[str, np.ndarray]:
        pos_state, image = state
        self._last_image = image
        return {
            "image_features": self._encode_image(image),
            "pos_state": self._flatten_pos_state(pos_state),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0

        idx = int(self._rng.integers(0, len(self._start_links)))
        link = self._start_links[idx]
        self._current_z_max = link.z_max

        self._neuro_env.reset(url=link.url)

        perturb_vec = sample_reset_perturbation(
            self._action_spec,
            self._rng,
            self._reset_rotation_perturb_rad,
            self._reset_zoom_perturb_frac,
        )
        state, _reward, _done, _json = self._neuro_env.step(perturb_vec)

        info = {
            "start_link_idx": idx,
            "z_max": self._current_z_max,
            "z_now": float(state[0][0][2]),
        }
        return self._build_obs(state), info

    def step(self, action):
        prev_state = self._neuro_env.prev_state
        vec, right_click_fired = decode(action, self._action_spec)

        state, _default_reward, _default_done, _json = self._neuro_env.step(vec)
        self._step_count += 1

        reward, terminated, was_noop = compute_reward(
            state,
            prev_state,
            right_click_fired,
            self._current_z_max,
            self._reward_cfg,
        )
        truncated = (not terminated) and (self._step_count >= self._max_episode_steps)

        info = {
            "z_now": float(state[0][0][2]),
            "z_max": self._current_z_max,
            "click_was_noop": was_noop,
            "right_click_fired": right_click_fired,
            "episode_success": terminated,
            "step": self._step_count,
        }
        return self._build_obs(state), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self._last_image

    def close(self):
        try:
            self._neuro_env.end_session()
        except Exception:
            pass
