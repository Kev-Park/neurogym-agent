from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ray = pytest.importorskip("ray")

import gymnasium as gym
from gymnasium import spaces

from ngllib_agent.policies import HierarchicalPPOModule

NVEC = [3, 1024, 9, 9, 9, 9]
IMG_DIM, POS_DIM = 768, 8

OBS_SPACE = spaces.Dict(
    {
        "image_features": spaces.Box(-np.inf, np.inf, (IMG_DIM,), np.float32),
        "pos_state": spaces.Box(-np.inf, np.inf, (POS_DIM,), np.float32),
    }
)
ACT_SPACE = spaces.MultiDiscrete(NVEC)


def _module(**kw):
    return HierarchicalPPOModule(
        observation_space=OBS_SPACE,
        action_space=ACT_SPACE,
        model_config={"pos_hidden_dim": 32, "trunk_hiddens": [64]},
        **kw,
    )


def _batch(b=5):
    from ray.rllib.core.columns import Columns

    return {
        Columns.OBS: {
            "image_features": torch.randn(b, IMG_DIM),
            "pos_state": torch.randn(b, POS_DIM),
        }
    }


def test_forward_shapes():
    from ray.rllib.core.columns import Columns

    m = _module()
    out = m.forward_inference(_batch())
    assert out[Columns.ACTION_DIST_INPUTS].shape == (5, sum(NVEC))
    out = m.forward_exploration(_batch())
    assert out[Columns.ACTION_DIST_INPUTS].shape == (5, sum(NVEC))
    out = m.forward_train(_batch())
    assert Columns.EMBEDDINGS in out
    assert out[Columns.ACTION_DIST_INPUTS].shape == (5, sum(NVEC))


def test_compute_values():
    m = _module()
    v = m.compute_values(_batch())
    assert v.shape == (5,)
    # and via precomputed embeddings
    from ray.rllib.core.columns import Columns

    out = m.forward_train(_batch())
    v2 = m.compute_values(_batch(), embeddings=out[Columns.EMBEDDINGS])
    assert v2.shape == (5,)


def test_inference_only_has_no_vf():
    m = _module(inference_only=True)
    assert not hasattr(m, "_vf_head")
    m.forward_inference(_batch())  # still computes actions


def test_dist_cls_bound_to_nvec():
    m = _module()
    dist_cls = m.get_inference_action_dist_cls()
    logits = torch.randn(2, sum(NVEC))
    dist = dist_cls.from_logits(logits)
    assert dist.sample().shape == (2, 6)


class _FakeZNavEnv(gym.Env):
    """Same spaces as the DINO-wrapped env; reward loosely favors action_type=2."""

    def __init__(self, config=None):
        self.observation_space = OBS_SPACE
        self.action_space = ACT_SPACE
        self._t = 0

    def _obs(self):
        return {
            "image_features": np.random.randn(IMG_DIM).astype(np.float32),
            "pos_state": np.random.randn(POS_DIM).astype(np.float32),
        }

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        reward = 1.0 if int(action[0]) == 2 else 0.0
        return self._obs(), reward, False, self._t >= 20, {}


def test_ppo_one_iter_end_to_end():
    """Full PPO train iteration through the custom module + gated distribution."""
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    ray.init(include_dashboard=False, num_cpus=2, ignore_reinit_error=True)
    try:
        config = (
            PPOConfig()
            .environment(_FakeZNavEnv)
            .framework("torch")
            .env_runners(num_env_runners=0, rollout_fragment_length="auto")
            .learners(num_learners=0)
            .training(train_batch_size=64, minibatch_size=32, num_epochs=1)
            .rl_module(
                rl_module_spec=RLModuleSpec(
                    module_class=HierarchicalPPOModule,
                    model_config={"pos_hidden_dim": 16, "trunk_hiddens": [32]},
                )
            )
        )
        algo = config.build_algo() if hasattr(config, "build_algo") else config.build()
        result = algo.train()
        pol = (result.get("learners", {}) or {}).get("default_policy", {}) or {}
        assert pol.get("total_loss") is not None
        assert np.isfinite(float(pol["total_loss"]))
        algo.stop()
    finally:
        ray.shutdown()
