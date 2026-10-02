"""Where does the RLlib learner's GPU memory go for the token-obs policy?

Runs ONE real PPO training iteration (RLlib new API, GPU learner, local
sampling from a fake env with the v7 observation/action spaces) at the real
batch size, and logs, every time the module's compute_values / forward_train
is entered: the obs dtype/shape/device and torch.cuda.memory_allocated /
max_memory_allocated. Needs one GPU; no simulator.

    uv run --no-sync python native/probe_learner_rllib_mem.py 16000
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ.setdefault("RAY_ADDRESS", "local")

import gymnasium as gym
import numpy as np
import ray
import torch
import yaml
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from ngllib_agent.policies import HierarchicalPPOModule
from ngllib_agent.wrappers.action import ActionSpec

CFG = yaml.safe_load(open("configs/native_zmaxleft_v7.yaml"))
AC = CFG["action"]
SPEC = ActionSpec(verbs=AC["verbs"], grid_rows=AC["grid_rows"], grid_cols=AC["grid_cols"],
                  rotation_bins_per_axis=AC["rotation_bins_per_axis"], zoom_bins=AC["zoom_bins"])
OBS = spaces.Dict({
    "pos_state": spaces.Box(-np.inf, np.inf, (8,), np.float32),
    "patch_features": spaces.Box(-np.inf, np.inf, (2, 16, 16, 384), np.float16),
})
ACT = spaces.MultiDiscrete(SPEC.nvec())
_TOK = np.random.randn(64, 2, 16, 16, 384).astype(np.float16)


class FakeEnv(gym.Env):
    def __init__(self, config=None):
        self.observation_space, self.action_space, self._t = OBS, ACT, 0

    def _obs(self):
        return {"pos_state": np.random.randn(8).astype(np.float32),
                "patch_features": _TOK[np.random.randint(64)]}

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, a):
        self._t += 1
        return self._obs(), float(np.random.randn()), False, self._t >= 500, {}


def _gb(x):
    return x / 2**30


class ProbeModule(HierarchicalPPOModule):
    def _log(self, tag, batch):
        pf = batch[Columns.OBS]["patch_features"]
        print(f"[probe] {tag}: patch {tuple(pf.shape)} {pf.dtype} {pf.device} | "
              f"alloc {_gb(torch.cuda.memory_allocated()):.2f} GB "
              f"peak {_gb(torch.cuda.max_memory_allocated()):.2f} GB", flush=True)

    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            self._log("compute_values", batch)
        return super().compute_values(batch, embeddings)

    def _forward_train(self, batch, **kw):
        if not getattr(self, "_logged_train", False):
            self._log("forward_train(first)", batch)
            self._logged_train = True
        return super()._forward_train(batch, **kw)


def main() -> int:
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
    ray.init(include_dashboard=False, num_cpus=4, num_gpus=1)
    config = (
        PPOConfig()
        .environment(FakeEnv)
        .framework("torch")
        .env_runners(num_env_runners=0, num_envs_per_env_runner=1,
                     rollout_fragment_length="auto")
        .learners(num_learners=0, num_gpus_per_learner=1)
        .training(train_batch_size=batch, minibatch_size=512, num_epochs=1)
        .rl_module(rl_module_spec=RLModuleSpec(module_class=ProbeModule,
                                               model_config=CFG["model"]))
    )
    algo = config.build_algo()
    algo.train()
    print(f"[probe] after iteration: alloc {_gb(torch.cuda.memory_allocated()):.2f} GB "
          f"peak {_gb(torch.cuda.max_memory_allocated()):.2f} GB", flush=True)
    print("PROBE-OK", flush=True)
    algo.stop()
    ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
