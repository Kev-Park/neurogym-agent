from __future__ import annotations

import threading
import time

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv

from ngllib_agent.vector_env import ThreadedVectorEnv


class _ThreadProbeEnv(gym.Env):
    """Records the thread ident of construction/reset/step calls."""

    def __init__(self, idx: int, episode_len: int = 3, step_sleep: float = 0.0):
        self.idx = idx
        self.episode_len = episode_len
        self.step_sleep = step_sleep
        self.threads: set[int] = {threading.get_ident()}
        self.observation_space = spaces.Box(-np.inf, np.inf, (2,), np.float32)
        self.action_space = spaces.Discrete(3)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.threads.add(threading.get_ident())
        self._t = 0
        return np.array([self.idx, 0], np.float32), {"idx": self.idx}

    def step(self, action):
        self.threads.add(threading.get_ident())
        if self.step_sleep:
            time.sleep(self.step_sleep)
        self._t += 1
        obs = np.array([self.idx, self._t], np.float32)
        terminated = self._t >= self.episode_len
        return obs, float(action), terminated, False, {}


def _fns(n, **kw):
    return [(lambda i=i: _ThreadProbeEnv(i, **kw)) for i in range(n)]


def test_sticky_thread_per_env():
    v = ThreadedVectorEnv(_fns(4))
    v.reset(seed=0)
    for _ in range(4):
        v.step(np.zeros(4, dtype=np.int64))
    all_threads = [e.threads for e in v.envs]
    main = threading.get_ident()
    for t in all_threads:
        assert len(t) == 1, "construct/reset/step must share ONE thread per env"
        assert main not in t, "env work must not run on the caller thread"
    assert len(set().union(*all_threads)) == 4, "each env gets a distinct thread"
    v.close()


def test_step_parallelism():
    v = ThreadedVectorEnv(_fns(4, episode_len=100, step_sleep=0.2))
    v.reset(seed=0)
    t0 = time.monotonic()
    v.step(np.zeros(4, dtype=np.int64))
    elapsed = time.monotonic() - t0
    v.close()
    assert elapsed < 0.55, f"4x0.2s steps should overlap, took {elapsed:.2f}s"


def test_semantics_match_sync_vector_env():
    # Same scripted envs + actions through both vectorizers -> identical output,
    # including NEXT_STEP autoreset behavior across episode boundaries.
    ref = SyncVectorEnv(_fns(3, episode_len=2))
    thr = ThreadedVectorEnv(_fns(3, episode_len=2))

    obs_r, _ = ref.reset(seed=7)
    obs_t, _ = thr.reset(seed=7)
    np.testing.assert_array_equal(obs_r, obs_t)

    rng_actions = [np.array([1, 2, 0]), np.array([0, 1, 2]),
                   np.array([2, 2, 2]), np.array([1, 0, 1])]
    for a in rng_actions:
        or_, rr, tr, cr, _ = ref.step(a)
        ot_, rt, tt, ct, _ = thr.step(a)
        np.testing.assert_array_equal(or_, ot_)
        np.testing.assert_array_equal(rr, rt)
        np.testing.assert_array_equal(tr, tt)
        np.testing.assert_array_equal(cr, ct)
    ref.close()
    thr.close()


def test_spaces_and_num_envs():
    v = ThreadedVectorEnv(_fns(5))
    assert v.num_envs == 5
    assert v.observation_space.shape == (5, 2)
    v.close()


def test_creator_threads_mode(monkeypatch):
    import ngllib_agent.env_build as eb

    monkeypatch.setattr(eb, "build_env", lambda cfg: _ThreadProbeEnv(0))
    creator = eb.make_env_creator({"x": 1}, vector_mode="threads")
    v = creator({"num_envs": 3})
    assert isinstance(v, ThreadedVectorEnv) and v.num_envs == 3
    v.close()
    assert creator({"num_envs": 1}).__class__ is _ThreadProbeEnv  # single env passthrough
    with pytest.raises(ValueError):
        eb.make_env_creator({}, vector_mode="bogus")
