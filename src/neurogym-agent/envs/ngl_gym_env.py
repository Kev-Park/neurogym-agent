from __future__ import annotations

import json
import math
import os
import random
import signal
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import psutil
from gymnasium import spaces
from playwright.sync_api import sync_playwright

_THIS_DIR = Path(__file__).resolve().parent
_PKG_DIR = _THIS_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from envs.action_translator import ActionSpec, decode, sample_reset_perturbation
from envs.browser_manager import BrowserManager
from envs.reward import RewardConfig, compute as compute_reward
from ngllib import Environment
from ngllib.utils.MouseActionHandler import MouseActionHandler

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
    "gpuMemoryLimit": 128 * 1024 * 1024,   # 128 MB per-context tile cache cap
    "systemMemoryLimit": 256 * 1024 * 1024, # 256 MB CPU-side prefetch cap
}


def _load_segment_positions(path: str) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load segment_positions.parquet → ({root_id: [[x,y,z], ...]}, [root_id, ...]).

    Call once in the main process; pass the returned dict to all NGLGymEnv workers.
    """
    df = pd.read_parquet(path, columns=["root_id", "x", "y", "z"])
    segment_data: dict[str, np.ndarray] = {
        str(rid): group[["x", "y", "z"]].values.astype(np.float32)
        for rid, group in df.groupby("root_id", sort=False)
    }
    return segment_data, list(segment_data.keys())


def _random_quaternion() -> list[float]:
    u1, u2, u3 = random.random(), random.random(), random.random()
    return [
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    ]


def _make_url(segment_id: str, segment_data: dict[str, np.ndarray]) -> str:
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
        segment_data: dict[str, np.ndarray],
        segment_ids: list[str],
        action_spec: ActionSpec,
        reward_cfg: RewardConfig,
        browser_manager: BrowserManager,
        max_episode_steps: int = 300,
        reset_rotation_perturb_rad: float = 0.5,
        reset_zoom_perturb_frac: float = 0.25,
        headless: bool = True,
        left_pane: bool = False,
        right_pane: bool = True,
        browser_restart_every: int = 90,
    ):
        super().__init__()
        self._action_spec = action_spec
        self._reward_cfg = reward_cfg
        self._max_episode_steps = max_episode_steps
        self._reset_rotation_perturb_rad = reset_rotation_perturb_rad
        self._reset_zoom_perturb_frac = reset_zoom_perturb_frac
        self._segment_data = segment_data
        self._segment_ids = segment_ids
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
        self._browser_manager = browser_manager
        self._browser_restart_every = browser_restart_every

        # Each worker thread owns its own playwright + browser connection.
        # These are created lazily on first use (in the worker thread, not the main thread).
        self._pw = None
        self._browser = None
        self._chrome_pid: int | None = None
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
        self._episode_count = 0
        self._last_image = None
        self._z_max: float = float("inf")
        self._last_seg_id: str = ""
        self._chrome_started: bool = False  # True after first reset completes

    # ------------------------------------------------------------------ browser / context

    def _reconnect(self) -> None:
        """Launch (or relaunch) this worker's own Chrome instance.

        Must be called from the worker thread — sync_playwright is thread-affine.
        Each worker owns its Chrome; no cross-thread browser sharing is attempted.
        """
        if self._pw is not None:
            try:
                self._pw.stop()
            except Exception:
                pass
        self._pw = sync_playwright().start()
        # Mirror ngllib's _build_launch_args for headless Linux GPU rendering.
        # Without these flags Chrome falls back to SwiftShader (CPU software renderer).
        _chrome_args = [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
            "--use-gl=angle",
            "--use-angle=vulkan",
            "--enable-features=Vulkan",
            "--enable-unsafe-swiftshader",
        ] + self._browser_manager.extra_args
        print(f"[chrome] {threading.current_thread().name}: launching Chrome...", flush=True)
        _pre_launch = time.time()
        self._browser = self._pw.chromium.launch(
            headless=self._headless,
            args=_chrome_args,
        )
        print(f"[chrome] {threading.current_thread().name}: Chrome up in {time.time()-_pre_launch:.1f}s", flush=True)
        # browser.process is not exposed in Playwright Python; find the Chrome
        # subprocess via psutil so the watchdog can kill it cross-thread safely.
        self._chrome_pid = None
        try:
            for child in psutil.Process(os.getpid()).children(recursive=True):
                try:
                    name = child.name().lower()
                    if ("chrome" in name or "chromium" in name) and child.create_time() >= _pre_launch - 1.0:
                        self._chrome_pid = child.pid
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception:
            pass
        if self._neuro_env is not None:
            self._neuro_env.browser = self._browser

    def _make_neuro_env(self) -> Environment:
        self._browser_manager.wait_ready()
        self._reconnect()
        env = Environment(headless=self._headless, config_path=self._neurogym_config_path)
        env.options = self._neuro_env_options
        env.browser = self._browser  # inject — skips ngllib's own Chrome launch
        return env

    def _new_context(self) -> None:
        """Open a fresh BrowserContext+Page for this worker, clear HTTP cache via CDP."""
        self._browser_manager.wait_ready()
        # Reconnect if this is the first call or if Chrome restarted since last time.
        if self._browser is None or not self._browser.is_connected():
            self._reconnect()

        new_ctx = self._browser.new_context(
            viewport={"width": self._neuro_env.window_width, "height": self._neuro_env.window_height}
        )
        new_page = new_ctx.new_page()
        try:
            cdp = new_page.context.new_cdp_session(new_page)
            cdp.send("Network.clearBrowserCache")
            cdp.detach()
        except Exception:
            pass
        old_ctx = (
            self._neuro_env.page.context
            if self._neuro_env is not None and self._neuro_env.page is not None
            else None
        )
        self._neuro_env.browser = self._browser  # keep in sync after reconnect
        self._neuro_env.page = new_page
        self._neuro_env.action_handler = MouseActionHandler(new_page)
        if old_ctx is not None:
            try:
                old_ctx.close()
            except Exception:
                pass

    def _restart_context(self) -> None:
        """Close the dead context and open a fresh one.
        If Chrome itself is down, wait_ready() inside _new_context() blocks until
        BrowserManager has relaunched it, then _new_context() reconnects."""
        try:
            if self._neuro_env is not None and self._neuro_env.page is not None:
                self._neuro_env.page.context.close()
        except Exception:
            pass
        # If _neuro_env was never created (e.g. _make_neuro_env failed), skip
        # _new_context — the next _neuro_reset attempt will call _make_neuro_env again.
        if self._neuro_env is not None:
            self._new_context()

    def _watchdog_kill_context(self) -> None:
        """Fired by a threading.Timer when a Playwright call hangs past its timeout.
        Kills the Chrome process directly — safe to call from any thread, unlike
        Playwright's greenlet API which is thread-affine and would deadlock here."""
        pid = self._chrome_pid
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    # ------------------------------------------------------------------ neuro wrappers

    def _neuro_reset(self, url: str, timeout: int = 240) -> None:
        if self._neuro_env is None:
            self._neuro_env = self._make_neuro_env()
        self._new_context()  # fresh context every episode; also clears HTTP cache
        print(f"[ngl] {threading.current_thread().name}: loading URL (timeout={timeout}s)...", flush=True)
        watchdog = threading.Timer(timeout, self._watchdog_kill_context)
        watchdog.start()
        try:
            self._neuro_env.reset(url=url)
            self._neuro_env.page.evaluate("1")  # post-nav health check
            print(f"[ngl] {threading.current_thread().name}: URL loaded OK", flush=True)
        finally:
            watchdog.cancel()

    def _neuro_step(self, vec, timeout: int = 30):
        watchdog = threading.Timer(timeout, self._watchdog_kill_context)
        watchdog.start()
        try:
            return self._neuro_env.step(vec)
        finally:
            watchdog.cancel()

    # ------------------------------------------------------------------ obs helpers

    def _flatten_pos_state(self, pos_state: list) -> np.ndarray:
        position, cs_scale, orientation_euler, proj_scale = pos_state
        return np.asarray(
            list(position) + [cs_scale] + list(orientation_euler) + [proj_scale],
            dtype=np.float32,
        )

    def _build_obs(self, state) -> dict[str, np.ndarray]:
        pos_state, image = state
        expected = self.observation_space["image"].shape
        if image.shape != expected:
            raise RuntimeError(f"image shape {image.shape} != expected {expected}")
        self._last_image = image
        return {
            "image": image,
            "pos_state": self._flatten_pos_state(pos_state),
        }

    # ------------------------------------------------------------------ gym API

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._episode_count += 1

        # Periodically restart the entire Chrome process to flush GPU-subprocess
        # resource pools (Vulkan command buffers, etc.) that accumulate across episodes
        # and aren't released by context-level resets.
        if self._episode_count % self._browser_restart_every == 0:
            self._reconnect()

        # First reset: all workers start Chrome simultaneously → heavy network load.
        # Give extra time. After Chrome is warm, resets are typically < 10s.
        reset_timeout = 240 if not self._chrome_started else 60

        for attempt in range(4):
            try:
                seg_id = random.choice(self._segment_ids)
                self._last_seg_id = seg_id
                self._z_max = float(self._segment_data[seg_id][:, 2].max())
                url = _make_url(seg_id, self._segment_data)
                self._neuro_reset(url, timeout=reset_timeout)
                self._chrome_started = True
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
                if attempt < 3:
                    self._restart_context()
                    time.sleep(random.uniform(1, 5))
                else:
                    raise

    def step(self, action):
        prev_state = self._neuro_env.prev_state
        vec, right_click_fired = decode(action, self._action_spec)

        try:
            state, _default_reward, _default_done, _json = self._neuro_step(vec)
            obs = self._build_obs(state)
        except Exception:
            self._restart_context()
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
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self._last_image

    def close(self):
        # Close this worker's context only — do NOT close the shared Chrome browser.
        try:
            if self._neuro_env is not None and self._neuro_env.page is not None:
                self._neuro_env.page.context.close()
        except Exception:
            pass
        # Stop this worker's playwright instance (disconnects from Chrome).
        try:
            if self._pw is not None:
                self._pw.stop()
        except Exception:
            pass
