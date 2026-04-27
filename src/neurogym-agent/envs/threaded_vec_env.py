from __future__ import annotations

import queue
import threading
from typing import Any, Callable

import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn


def _worker(env_fn: Callable, cmd_q: queue.Queue, res_q: queue.Queue) -> None:
    env = env_fn()
    while True:
        cmd, data = cmd_q.get()
        try:
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    info["terminal_observation"] = obs
                    obs, _ = env.reset()
                res_q.put(("ok", (obs, reward, done, info)))

            elif cmd == "reset":
                obs, reset_info = env.reset(**data)
                res_q.put(("ok", (obs, reset_info)))

            elif cmd == "get_attr":
                res_q.put(("ok", getattr(env, data)))

            elif cmd == "set_attr":
                name, val = data
                setattr(env, name, val)
                res_q.put(("ok", None))

            elif cmd == "env_method":
                name, args, kwargs = data
                res_q.put(("ok", getattr(env, name)(*args, **kwargs)))

            elif cmd == "is_wrapped":
                from stable_baselines3.common import env_util
                res_q.put(("ok", env_util.is_wrapped(env, data)))

            elif cmd == "close":
                env.close()
                res_q.put(("ok", None))
                return

        except Exception as exc:
            res_q.put(("err", exc))


class ThreadedVecEnv(VecEnv):
    """VecEnv backed by daemon threads instead of subprocesses.

    Suitable when envs are I/O-bound (e.g. waiting on Playwright/Chrome):
    the GIL is released during I/O waits, giving true concurrency without
    the subprocess overhead and cross-process serialisation of SubprocVecEnv.

    The step_wait() return matches SB3's expected 4-tuple
    (obs, rewards, dones, infos) with terminated/truncated merged into dones
    and TimeLimit.truncated stored in infos, exactly as SubprocVecEnv does.
    """

    def __init__(self, env_fns: list[Callable]):
        # Instantiate one env just to read spaces, then discard it.
        probe = env_fns[0]()
        super().__init__(len(env_fns), probe.observation_space, probe.action_space)
        probe.close()

        self._cmd_qs: list[queue.Queue] = [queue.Queue() for _ in env_fns]
        self._res_qs: list[queue.Queue] = [queue.Queue() for _ in env_fns]
        self._threads: list[threading.Thread] = []
        for i, fn in enumerate(env_fns):
            t = threading.Thread(
                target=_worker,
                args=(fn, self._cmd_qs[i], self._res_qs[i]),
                daemon=True,
                name=f"vecenv-worker-{i}",
            )
            t.start()
            self._threads.append(t)

    # ------------------------------------------------------------------ helpers

    def _send_all(self, cmd: str, data_list: list) -> None:
        for q, d in zip(self._cmd_qs, data_list):
            q.put((cmd, d))

    def _recv_all(self) -> list:
        results = []
        for i, q in enumerate(self._res_qs):
            status, val = q.get()
            if status == "err":
                raise RuntimeError(f"Worker {i} raised: {val}") from val
            results.append(val)
        return results

    def _get_indices(self, indices) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    @staticmethod
    def _stack_obs(obs_list: list) -> Any:
        if isinstance(obs_list[0], dict):
            return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
        return np.stack(obs_list)

    # ------------------------------------------------------------------ VecEnv API

    def step_async(self, actions: np.ndarray) -> None:
        self._send_all("step", list(actions))

    def step_wait(self) -> VecEnvStepReturn:
        results = self._recv_all()
        obs_list, rews, dones, infos = zip(*results)
        return (
            self._stack_obs(list(obs_list)),
            np.array(rews, dtype=np.float32),
            np.array(dones),
            list(infos),
        )

    def reset(self) -> VecEnvObs:
        self._send_all("reset", [{} for _ in self._cmd_qs])
        results = self._recv_all()
        obs_list = [r[0] for r in results]
        return self._stack_obs(obs_list)

    def close(self) -> None:
        self._send_all("close", [None] * self.num_envs)
        self._recv_all()
        for t in self._threads:
            t.join(timeout=30)

    def get_attr(self, attr_name: str, indices=None) -> list:
        idxs = self._get_indices(indices)
        for i in idxs:
            self._cmd_qs[i].put(("get_attr", attr_name))
        return [self._res_qs[i].get()[1] for i in idxs]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        idxs = self._get_indices(indices)
        for i in idxs:
            self._cmd_qs[i].put(("set_attr", (attr_name, value)))
        for i in idxs:
            self._res_qs[i].get()

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list:
        idxs = self._get_indices(indices)
        for i in idxs:
            self._cmd_qs[i].put(("env_method", (method_name, method_args, method_kwargs)))
        return [self._res_qs[i].get()[1] for i in idxs]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        idxs = self._get_indices(indices)
        for i in idxs:
            self._cmd_qs[i].put(("is_wrapped", wrapper_class))
        return [self._res_qs[i].get()[1] for i in idxs]

    def seed(self, seed=None) -> list:
        return [None] * self.num_envs
