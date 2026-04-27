from __future__ import annotations

import json
import math
import os
import random
import sys
import threading
import urllib.parse
from pathlib import Path
from typing import Any

import psutil

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

_THIS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _THIS_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from envs.action_translator import ActionSpec, decode, sample_reset_perturbation
from envs.reward import RewardConfig, compute as compute_reward
from ngllib import Environment

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
    """Load segment_positions.parquet → ({root_id: [[x,y,z], ...]}, [root_id, ...])."""
    df = pd.read_parquet(path, columns=["root_id", "x", "y", "z"])
    # Keep as float32 numpy arrays (N,3) — ~300 MB total vs ~5 GB for Python lists.
    segment_data: dict[str, np.ndarray] = {
        str(rid): group[["x", "y", "z"]].values.astype(np.float32)
        for rid, group in df.groupby("root_id", sort=False)
    }
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


def _make_url(segment_id: str, segment_data: dict[str, np.ndarray]) -> str:
    """Build a Neuroglancer URL for a random position along the given segment."""
    positions = segment_data[segment_id]
    pos = positions[random.randrange(len(positions))].tolist()
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
        chrome_startup_sem=None,
    ):
        super().__init__()
        self._action_spec = action_spec
        self._reward_cfg = reward_cfg
        self._max_episode_steps = max_episode_steps
        self._reset_rotation_perturb_rad = reset_rotation_perturb_rad
        self._reset_zoom_perturb_frac = reset_zoom_perturb_frac

        self._segment_data, self._segment_ids = _load_segment_positions(segment_positions_path)

        self._neurogym_config_path = neurogym_config_path
        self._headless = headless
        self._neuro_env_options = {
            "euler_angles": True,
            "resize": False,
            "add_mouse": False,
            "fast": True,
            "image_path": None,
            "left_pane": left_pane,
            "right_pane": right_pane,
        }
        self._chrome_startup_sem = chrome_startup_sem
        # Chrome is started lazily on the first reset() to avoid all workers
        # hammering the GPU simultaneously during SubprocVecEnv.__init__.
        self._neuro_env = None

        pos_dim = 3 + 1 + 3 + 1

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=255, shape=(900, 900, 3), dtype=np.uint8
                ),
                "pos_state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(pos_dim,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.MultiDiscrete(action_spec.multidiscrete_nvec())

        self._rng = np.random.default_rng()
        self._step_count = 0
        # Random offset so workers with the same n_envs don't all hit the
        # periodic _restart_browser() at the same time.
        self._episode_count = random.randint(0, 49)
        self._last_image = None
        self._z_max: float = float("inf")
        self._last_seg_id: str = ""

        # Persistent watchdog: fires _kill_chrome_children if _watchdog_deadline is
        # set and the current time exceeds it. Idle (deadline=inf) between calls.
    def _make_neuro_env(self) -> Environment:
        if self._chrome_startup_sem is not None:
            self._chrome_startup_sem.acquire()
        try:
            env = Environment(headless=self._headless, config_path=self._neurogym_config_path)
            env.options = self._neuro_env_options
            return env
        finally:
            if self._chrome_startup_sem is not None:
                self._chrome_startup_sem.release()

    def _chrome_rss_mb(self) -> float:
        """Return total RSS in MB of all Chrome child processes of this worker."""
        try:
            total = 0
            for child in psutil.Process(os.getpid()).children(recursive=True):
                try:
                    if "chrome" in child.name().lower():
                        total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total / (1024 * 1024)
        except Exception:
            return 0.0

    def _kill_chrome_children(self) -> None:
        """Force-SIGKILL any Chrome/Chromium child processes of this worker process."""
        try:
            current = psutil.Process(os.getpid())
            for child in current.children(recursive=True):
                try:
                    if "chrome" in child.name().lower():
                        child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            pass

    def _new_context(self) -> None:
        """Swap to a fresh browser context, discarding the cached tile data.
        Raises if the browser is dead — caller should escalate to _restart_browser()."""
        old_ctx = self._neuro_env.page.context
        new_ctx = self._neuro_env.browser.new_context()
        self._neuro_env.page = new_ctx.new_page()
        try:
            old_ctx.close()
        except Exception:
            pass

    def _neuro_reset(self, url: str, timeout: int = 60):
        """Swap to a fresh context (clears tile cache), then navigate with a watchdog."""
        self._new_context()  # raises if browser dead → reset()'s retry calls _restart_browser()
        watchdog = threading.Timer(timeout, self._kill_chrome_children)
        watchdog.start()
        try:
            self._neuro_env.reset(url=url)
        finally:
            watchdog.cancel()

    def _neuro_step(self, vec, timeout: int = 30):
        """Call neuro_env.step; if Chrome hangs, kill it via a watchdog thread and raise."""
        watchdog = threading.Timer(timeout, self._kill_chrome_children)
        watchdog.start()
        try:
            return self._neuro_env.step(vec)
        finally:
            watchdog.cancel()

    def _restart_browser(self) -> None:
        """Tear down Playwright step-by-step, then force-kill any surviving Chrome, then respawn."""
        try:
            if getattr(self._neuro_env, "page", None):
                try:
                    self._neuro_env.page.close()
                except Exception:
                    pass
            if getattr(self._neuro_env, "browser", None):
                try:
                    self._neuro_env.browser.close()
                except Exception:
                    pass
            if getattr(self._neuro_env, "_playwright", None):
                try:
                    self._neuro_env._playwright.stop()
                except Exception:
                    pass
        except Exception:
            pass
        self._kill_chrome_children()
        self._neuro_env = self._make_neuro_env()

    def _webgl_healthy(self) -> bool:
        """Return False if Chrome's GPU process has crashed and fallen back to SwiftShader."""
        if self._neuro_env is None:
            return True
        try:
            return bool(self._neuro_env.page.evaluate(
                "() => { const c = document.createElement('canvas');"
                " const gl = c.getContext('webgl2') || c.getContext('webgl');"
                " if (!gl) return false;"
                " const dbg = gl.getExtension('WEBGL_debug_renderer_info');"
                " if (!dbg) return true;"
                " const renderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL);"
                " return !renderer.toLowerCase().includes('swiftshader'); }"
            ))
        except Exception:
            return False

    def _flatten_pos_state(self, pos_state: list) -> np.ndarray:
        position, cs_scale, orientation_euler, proj_scale = pos_state
        return np.asarray(
            list(position) + [cs_scale] + list(orientation_euler) + [proj_scale],
            dtype=np.float32,
        )

    def _build_obs(self, state) -> dict[str, np.ndarray]:
        pos_state, image = state
        self._last_image = image
        return {
            "image": image,
            "pos_state": self._flatten_pos_state(pos_state),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._episode_count += 1
        if self._episode_count % 50 == 0:
            rss = self._chrome_rss_mb()
            print(f"[chrome] ep {self._episode_count}: RSS={rss:.0f} MB pre-restart", flush=True)
            self._restart_browser()

        for attempt in range(2):
            try:
                seg_id = random.choice(self._segment_ids)
                self._last_seg_id = seg_id
                self._z_max = float(self._segment_data[seg_id][:, 2].max())
                url = _make_url(seg_id, self._segment_data)
                if not self._webgl_healthy():
                    self._restart_browser()

                self._neuro_reset(url)

                perturb_vec = sample_reset_perturbation(
                    self._action_spec,
                    self._rng,
                    self._reset_rotation_perturb_rad,
                    self._reset_zoom_perturb_frac,
                )
                state, _reward, _done, _json = self._neuro_step(perturb_vec)
                info = {
                    "segment_id": seg_id,
                    "z_max": self._z_max,
                    "z_now": float(state[0][0][2]),
                }
                return self._build_obs(state), info
            except Exception:
                if attempt == 0:
                    self._restart_browser()
                else:
                    raise

    def step(self, action):
        prev_state = self._neuro_env.prev_state
        vec, right_click_fired = decode(action, self._action_spec)

        try:
            state, _default_reward, _default_done, _json = self._neuro_step(vec)
        except Exception:
            self._restart_browser()
            obs, _ = self.reset()
            return obs, 0.0, False, True, {
                "z_now": float("nan"),
                "z_max": self._z_max,
                "click_was_noop": False,
                "right_click_fired": False,
                "episode_success": False,
                "step": self._step_count,
                "browser_crash": True,
            }

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
