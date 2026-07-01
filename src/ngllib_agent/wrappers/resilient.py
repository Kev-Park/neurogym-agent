"""ResilientStepWrapper — truncate the episode on a transient env glitch.

Neuroglancer occasionally returns a viewer state missing an expected field (e.g.
`position`), and a browser can crash mid-run. ngllib surfaces these as raw
exceptions from `step()`. Rather than kill training, end the episode
(`truncated=True`) and let the next `reset()` recover (ngllib re-navigates with
its own retry/browser-restart). Mirrors the legacy `NGLGymEnv.step` recovery.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ResilientStepWrapper:
    """gymnasium `Wrapper` catching step-time env glitches. Imported lazily to
    keep gymnasium/ngllib out of module import for pure-logic tests."""

    def __new__(cls, env):
        import gymnasium as gym

        try:
            from ngllib import NgllibError

            ngllib_errs: tuple = (NgllibError,)
        except Exception:
            ngllib_errs = ()
        # ngllib doesn't yet wrap obs-gathering KeyError/TypeError (its issue #2).
        catch = (KeyError, TypeError) + ngllib_errs

        class _Impl(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                self._last_obs: Any = None

            def reset(self, *, seed=None, options=None):
                obs, info = self.env.reset(seed=seed, options=options)
                self._last_obs = obs
                return obs, info

            def step(self, action):
                try:
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    self._last_obs = obs
                    return obs, reward, terminated, truncated, info
                except catch as e:
                    logger.warning(
                        "resilient: env glitch %s (%s); truncating episode",
                        type(e).__name__,
                        e,
                    )
                    return self._last_obs, 0.0, False, True, {"env_glitch": type(e).__name__}

        return _Impl(env)
