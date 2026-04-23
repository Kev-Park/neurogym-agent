from __future__ import annotations

import csv
import json
import math
import random
import sys
import urllib.parse
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

csv.field_size_limit(2**31 - 1)

_BASE_STATE = {
    "dimensions": {"x": [4e-9, "m"], "y": [4e-9, "m"], "z": [4e-8, "m"]},
    "position": [0, 0, 0],
    "crossSectionScale": 2.0,
    "projectionOrientation": [0, 0, 0, 1],
    "projectionScale": 14000,
    "layers": [
        {
            "type": "image",
            "source": "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14",
            "tab": "source",
            "name": "Maryland (USA)-image",
        },
        {
            "type": "segmentation",
            "source": "precomputed://gs://flywire_v141_m783",
            "tab": "source",
            "segments": [],
            "name": "flywire_v141_m783",
        },
    ],
    "showDefaultAnnotations": False,
    "selectedLayer": {"size": 350, "layer": "flywire_v141_m783"},
    "layout": "xy-3d",
}


def _load_segment_positions(path: str) -> tuple[dict[str, list[list[float]]], list[str]]:
    """Load segment_positions.csv → ({root_id: [[x,y,z], ...]}, [root_id, ...])."""
    segment_data: dict[str, list[list[float]]] = {}
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            rid = row[0]
            coords: list[list[float]] = []
            for pos in row[1].split("|"):
                x, y, z = pos.split(";")
                coords.append([float(x), float(y), float(z)])
            segment_data[rid] = coords
    return segment_data, list(segment_data.keys())


def _random_quaternion() -> list[float]:
    """Generate a uniformly random unit quaternion [x, y, z, w]."""
    u1, u2, u3 = random.random(), random.random(), random.random()
    return [
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    ]


def _make_url(segment_id: str, segment_data: dict[str, list[list[float]]]) -> str:
    """Build a Neuroglancer URL for a random position along the given segment."""
    pos = random.choice(segment_data[segment_id])
    orientation = _random_quaternion()
    state = json.loads(json.dumps(_BASE_STATE))
    state["layers"][1]["segments"] = [segment_id]
    state["position"] = pos
    state["projectionOrientation"] = orientation
    return "https://neuroglancer-demo.appspot.com/#!" + urllib.parse.quote(json.dumps(state))


class NGLGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        neurogym_config_path: str,
        segment_positions_path: str,
        action_spec: ActionSpec,
        reward_cfg: RewardConfig,
        max_episode_steps: int = 300,
        reset_rotation_perturb_rad: float = 0.5,
        reset_zoom_perturb_frac: float = 0.25,
        headless: bool = True,
        left_pane: bool = False,
        right_pane: bool = True,
        dino_repo: str = "facebookresearch/dinov2",
        dino_model: str = "dinov2_vits14",
        dino_input_size: int = 224,
        dino_device: str | None = None,
    ):
        super().__init__()
        self._action_spec = action_spec
        self._reward_cfg = reward_cfg
        self._max_episode_steps = max_episode_steps
        self._reset_rotation_perturb_rad = reset_rotation_perturb_rad
        self._reset_zoom_perturb_frac = reset_zoom_perturb_frac

        self._segment_data, self._segment_ids = _load_segment_positions(segment_positions_path)

        self._neuro_env = Environment(
            headless=headless,
            config_path=neurogym_config_path,
        )
        self._neuro_env.options = {
            "euler_angles": True,
            "resize": False,
            "add_mouse": False,
            "fast": True,
            "image_path": None,
            "left_pane": left_pane,
            "right_pane": right_pane,
        }

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
        self._last_image = None
        self._z_max: float = float("inf")

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

        seg_id = random.choice(self._segment_ids)
        self._z_max = max(pos[2] for pos in self._segment_data[seg_id])
        url = _make_url(seg_id, self._segment_data)
        self._neuro_env.reset(url=url)

        perturb_vec = sample_reset_perturbation(
            self._action_spec,
            self._rng,
            self._reset_rotation_perturb_rad,
            self._reset_zoom_perturb_frac,
        )
        state, _reward, _done, _json = self._neuro_env.step(perturb_vec)

        info = {
            "segment_id": seg_id,
            "z_max": self._z_max,
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
            self._z_max,
            self._reward_cfg,
        )
        truncated = (not terminated) and (self._step_count >= self._max_episode_steps)

        info = {
            "z_now": float(state[0][0][2]),
            "z_max": self._z_max,
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
