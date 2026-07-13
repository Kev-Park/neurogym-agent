"""ResilientStepWrapper — truncate the episode on a transient env glitch.

Neuroglancer occasionally returns a viewer state missing an expected field (e.g.
`position`), and a browser can crash mid-run. ngllib surfaces these as raw
exceptions from `step()`. Rather than kill training, end the episode
(`truncated=True`) and let the next `reset()` recover (ngllib re-navigates with
its own retry/browser-restart). Mirrors the legacy `NGLGymEnv.step` recovery.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

logger = logging.getLogger(__name__)

# Jittered backoff between reset retries — ported from the legacy
# `NGLGymEnv.reset` (`time.sleep(random.uniform(1, 5))` per attempt). When many
# browsers glitch at once (viewer-state race under load), retrying in lockstep is
# a thundering herd that spikes GPU/CPU and re-fails, cascading into a glitch
# storm. Staggering recovery across the workers lets resources settle so retries
# succeed — this is what kept the legacy 32-browser/node runs clean (2026-07-13
# root-cause: the new stack lost this jitter, so glitches propagated to RLlib).
_RESET_BACKOFF = (1.0, 5.0)


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
        # Raw Playwright errors (e.g. "Page.screenshot: Unable to capture
        # screenshot" under heavy multi-browser load) are NOT wrapped by ngllib,
        # so without this they escape to RLlib and crash the whole EnvRunner
        # (all 16 browsers) instead of truncating one episode (observed in the
        # 2026-07-12 multi-node val run, job 845535 — halved throughput).
        try:
            from playwright.sync_api import Error as PlaywrightError

            pw_errs: tuple = (PlaywrightError,)
        except Exception:
            pw_errs = ()
        # ngllib doesn't yet wrap obs-gathering KeyError/TypeError (its issue #2).
        catch = (KeyError, TypeError) + ngllib_errs + pw_errs

        class _Impl(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                self._last_obs: Any = None

            def reset(self, *, seed=None, options=None):
                # Under heavy multi-browser load a reset can hit a transient
                # viewer/browser glitch (BrowserError etc). Retry the whole reset
                # (each call re-recycles the browser context inside ngllib) with a
                # JITTERED backoff between attempts so the ~16 workers don't retry
                # in lockstep (thundering herd -> resource spike -> re-fail ->
                # storm). Ports the legacy NGLGymEnv.reset recovery. If it still
                # fails after N attempts, propagate (RLlib escalates to an
                # EnvRunner actor restart).
                last_e = None
                for attempt in range(4):
                    try:
                        obs, info = self.env.reset(seed=seed, options=options)
                        self._last_obs = obs
                        return obs, info
                    except catch as e:
                        last_e = e
                        logger.warning(
                            "resilient: reset glitch %s (%s); jittered-backoff "
                            "retry %d/4", type(e).__name__, e, attempt + 1,
                        )
                        if attempt < 3:
                            time.sleep(random.uniform(*_RESET_BACKOFF))
                raise last_e

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
