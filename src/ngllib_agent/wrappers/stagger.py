"""FirstEpisodeStagger — desynchronize episode boundaries across a vector env.

Envs in a ThreadedVectorEnv step in lockstep, so envs that reset together stay
phase-locked and hit their 300-step TimeLimit truncation together — producing
synchronized reset WAVES (measured 2026-08: 52 resets/10s; one runner's 16
simultaneous resets = ~100s serial stall through the vector join barrier).

Phase lives in step-space, not wall-clock, so the stagger is implemented as a
shorter FIRST episode: truncating env i's first episode at `first_limit` steps
permanently offsets its boundary cycle by (max_steps - first_limit). Even
spacing across a node's envs converts the burst of N simultaneous resets into
~1 reset at a time, forever, at the one-time cost of one short episode per env.

RL-correctness: the early end is a truncation (`truncated=True`, value-
bootstrapped like any TimeLimit truncation) — a shorter on-policy episode, not
a distribution change. All later episodes run the normal TimeLimit.
"""

from __future__ import annotations


class FirstEpisodeStagger:
    """gymnasium `Wrapper` truncating only the FIRST episode at `first_limit`
    steps. No-op passthrough for every episode after (and if the first episode
    ends on its own before the limit, the stagger is never applied). Lazy class
    def keeps gymnasium out of module import for pure-logic tests."""

    def __new__(cls, env, first_limit: int):
        import gymnasium as gym

        limit = max(1, int(first_limit))

        class _Impl(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                self._steps = 0
                self._first_done = False

            def reset(self, *, seed=None, options=None):
                self._steps = 0
                return self.env.reset(seed=seed, options=options)

            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                if not self._first_done:
                    self._steps += 1
                    if terminated or truncated:
                        self._first_done = True
                    elif self._steps >= limit:
                        self._first_done = True
                        truncated = True
                        info = {**info, "stagger_truncated": True}
                return obs, reward, terminated, truncated, info

        return _Impl(env)
