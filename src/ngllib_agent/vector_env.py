"""ThreadedVectorEnv — single-process, M-browser-threads vectorization (R4).

Replaces spawn-subprocess-per-env for M>1: one process per env-runner holds all
M envs, so there is ONE CUDA context and ONE process-singleton DINO encoder
(the legacy `ThreadedVecEnv` topology, rebuilt on gymnasium's vector API).

Thread-affinity contract: sync Playwright objects must only be touched from
the thread that created them. Each sub-env therefore gets a dedicated
single-thread executor that runs its construction, reset, step, and close;
`reset()`/`step()` dispatch all envs concurrently (browser stepping is
I/O-bound, so threads parallelize) and reuse SyncVectorEnv's batching state.

NEXT_STEP autoreset only (gymnasium default; what RLlib's env runner expects).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from copy import deepcopy
from typing import Any, Callable, Sequence

import numpy as np
from gymnasium import Env
from gymnasium.vector import AutoresetMode
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.vector.utils import concatenate, iterate


class ThreadedVectorEnv(SyncVectorEnv):
    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env]],
        copy: bool = True,
        observation_mode: str = "same",
        result_timeout_s: float = 300.0,
    ):
        env_fns = list(env_fns)
        # Abandon-and-rebuild backstop (2026-08-17): playwright-python can fail
        # to reject an in-flight sync call on connection loss (py-spy verified:
        # caller greenlet waits forever while the loop idles in select()) — a
        # thread wedge NO process-kill can clear. The vector layer therefore
        # never trusts an env thread: a step/reset result that exceeds
        # result_timeout_s abandons the wedged thread (it idles harmlessly; its
        # browser/driver are already dead) and rebuilds that env on a fresh
        # executor. Bounded cost, invisible to RLlib.
        self._env_fns = list(env_fns)
        self._result_timeout_s = result_timeout_s
        self._executors = [
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"venv-{i}")
            for i in range(len(env_fns))
        ]
        # Route each construction onto its sticky thread. The parent's
        # sequential list-comprehension keeps constructions serialized, which
        # also makes the process-singleton encoder init race-free.
        sticky_fns = [
            (lambda fn=fn, ex=ex: ex.submit(fn).result())
            for fn, ex in zip(env_fns, self._executors)
        ]
        super().__init__(
            sticky_fns,
            copy=copy,
            observation_mode=observation_mode,
            autoreset_mode=AutoresetMode.NEXT_STEP,
        )

    # ------------------------------------------------------------------ reset

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        if options is not None and "reset_mask" in options:
            options = dict(options)
            reset_mask = options.pop("reset_mask")
            assert reset_mask.shape == (self.num_envs,)
            mask = list(np.asarray(reset_mask, dtype=bool))
        else:
            mask = [True] * self.num_envs
            self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
            self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)
            self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

        for i, m in enumerate(mask):
            if m:
                self._terminations[i] = False
                self._truncations[i] = False
                self._autoreset_envs[i] = False

        futures = {
            i: self._executors[i].submit(self.envs[i].reset, seed=seed[i], options=options)
            for i, m in enumerate(mask)
            if m
        }
        infos: dict[str, Any] = {}
        for i in sorted(futures):
            try:
                self._env_obs[i], env_info = futures[i].result(
                    timeout=self._result_timeout_s
                )
            except FuturesTimeout:
                self._env_obs[i], env_info = self._abandon_and_rebuild(i)
            infos = self._add_info(infos, env_info, i)

        self._observations = concatenate(
            self.single_observation_space, self._env_obs, self._observations
        )
        return (deepcopy(self._observations) if self.copy else self._observations), infos

    # ------------------------------------------------------------------- step

    def step(self, actions):
        assert self.autoreset_mode == AutoresetMode.NEXT_STEP
        actions = list(iterate(self.action_space, actions))

        def _one(i: int, action):
            if self._autoreset_envs[i]:
                obs, info = self.envs[i].reset()
                return obs, 0.0, False, False, info
            return self.envs[i].step(action)

        futures = [
            self._executors[i].submit(_one, i, action)
            for i, action in enumerate(actions)
        ]
        infos: dict[str, Any] = {}
        for i, fut in enumerate(futures):
            try:
                (
                    self._env_obs[i],
                    self._rewards[i],
                    self._terminations[i],
                    self._truncations[i],
                    env_info,
                ) = fut.result(timeout=self._result_timeout_s)
            except FuturesTimeout:
                obs, env_info = self._abandon_and_rebuild(i)
                self._env_obs[i] = obs
                self._rewards[i] = 0.0
                self._terminations[i] = False
                self._truncations[i] = True
                env_info = {**env_info, "env_rebuilt": True}
            infos = self._add_info(infos, env_info, i)

        self._observations = concatenate(
            self.single_observation_space, self._env_obs, self._observations
        )
        self._autoreset_envs = np.logical_or(self._terminations, self._truncations)

        return (
            deepcopy(self._observations) if self.copy else self._observations,
            np.copy(self._rewards),
            np.copy(self._terminations),
            np.copy(self._truncations),
            infos,
        )

    # -------------------------------------------------------- abandon/rebuild

    def _abandon_and_rebuild(self, i: int):
        """Replace a wedged env with a fresh one on a fresh sticky thread.

        The old executor/thread is abandoned (never joined — it may be wedged
        inside playwright forever); its processes are dead or killed by the
        env-level watchdogs, so the leak is one idle thread. Build+reset run on
        the NEW thread with generous timeouts; if even that fails, raise — the
        runner dies and RLlib's actor restart is the outermost net.
        """
        logging.getLogger(__name__).error(
            "vector env %d wedged past %.0fs: abandoning thread and rebuilding",
            i, self._result_timeout_s,
        )
        try:
            self._executors[i].shutdown(wait=False)
        except Exception:
            pass
        ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"venv-{i}r")
        self._executors[i] = ex
        env = ex.submit(self._env_fns[i]).result(timeout=240.0)
        self.envs[i] = env
        obs, info = ex.submit(env.reset).result(timeout=300.0)
        return obs, info

    # ------------------------------------------------------------------ close

    def close_extras(self, **kwargs):
        for i, env in enumerate(self.envs):
            try:
                self._executors[i].submit(env.close).result(timeout=60)
            except Exception:
                pass
        for ex in self._executors:
            ex.shutdown(wait=False)
