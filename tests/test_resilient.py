from __future__ import annotations

import numpy as np


def test_resilient_truncates_on_glitch():
    import gymnasium as gym
    from gymnasium import spaces

    from ngllib_agent.wrappers import ResilientStepWrapper

    class _Boom(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            self.action_space = spaces.Discrete(2)
            self.n = 0

        def reset(self, *, seed=None, options=None):
            self.n = 0
            return np.zeros(2, np.float32), {}

        def step(self, a):
            self.n += 1
            if self.n == 1:
                return np.ones(2, np.float32), 1.0, False, False, {}
            raise KeyError("position")  # transient viewer glitch

    w = ResilientStepWrapper(_Boom())
    w.reset()

    obs, r, term, trunc, info = w.step(0)  # first step ok
    assert not trunc and r == 1.0

    obs, r, term, trunc, info = w.step(0)  # glitch -> truncate, return last obs
    assert trunc is True
    assert term is False
    assert info.get("env_glitch") == "KeyError"
    assert np.allclose(obs, np.ones(2))  # last good obs
