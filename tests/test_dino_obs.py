from __future__ import annotations

import numpy as np
import pytest

from ngllib_agent.wrappers import (
    DEFAULT_POS_STATE_SCALE,
    DinoObservationWrapper,
    PosStateWrapper,
    pos_state_from_obs,
    split_panes,
)


class _StubEncoder:
    feature_dim = 16

    def __init__(self):
        self.calls = []

    def encode(self, images):
        self.calls.append([im.shape for im in images])
        # deterministic: mean pixel value per image, tiled
        return np.stack(
            [np.full(self.feature_dim, im.mean(), np.float32) for im in images]
        )


def _raw_obs(w=64, h=32):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 2] = 10   # left pane
    img[:, w // 2:] = 200   # right pane
    return {
        "image": img,
        "position": np.array([128000.0, 52000.0, 3000.0], np.float32),
        "xs_scale": np.array([2.0], np.float32),
        "orientation": np.array([0.1, -0.2, 0.3], np.float32),
        "proj_scale": np.array([14000.0], np.float32),
    }


def test_split_panes():
    left, right = split_panes(_raw_obs()["image"])
    assert left.shape == right.shape == (32, 32, 3)
    assert left.mean() == 10 and right.mean() == 200


def test_pos_state_scaling():
    v = pos_state_from_obs(_raw_obs(), DEFAULT_POS_STATE_SCALE)
    assert v.shape == (8,)
    assert np.allclose(v[:3], [1.28, 0.52, 0.03])
    assert np.isclose(v[3], 2.0) and np.isclose(v[7], 1.4)
    assert np.allclose(v[4:7], [0.1, -0.2, 0.3])


class _StubEnv:
    """Minimal gym.Env with ngllib-shaped Dict obs."""

    def __new__(cls, left_pane=True, right_pane=True, orientation="euler"):
        import gymnasium as gym
        from gymnasium import spaces

        class _Impl(gym.Env):
            def __init__(self):
                self.left_pane = left_pane
                self.right_pane = right_pane
                self.orientation = orientation
                self.observation_space = spaces.Dict(
                    {
                        "image": spaces.Box(0, 255, (32, 64, 3), np.uint8),
                        "position": spaces.Box(-np.inf, np.inf, (3,), np.float32),
                        "xs_scale": spaces.Box(0, np.inf, (1,), np.float32),
                        "orientation": spaces.Box(-np.inf, np.inf, (3,), np.float32),
                        "proj_scale": spaces.Box(0, np.inf, (1,), np.float32),
                    }
                )
                self.action_space = spaces.Discrete(2)

            def reset(self, *, seed=None, options=None):
                return _raw_obs(), {}

            def step(self, action):
                return _raw_obs(), 0.0, False, False, {}

        return _Impl()


def test_dino_wrapper_transforms_obs():
    enc = _StubEncoder()
    w = DinoObservationWrapper(_StubEnv(), enc)
    obs, _ = w.reset()
    assert set(obs) == {"image_features", "pos_state"}
    assert obs["image_features"].shape == (2 * enc.feature_dim,)
    # left-pane features then right-pane features
    assert np.allclose(obs["image_features"][: enc.feature_dim], 10.0)
    assert np.allclose(obs["image_features"][enc.feature_dim:], 200.0)
    assert enc.calls[-1] == [(32, 32, 3), (32, 32, 3)]
    assert w.observation_space.contains(obs)

    obs2, *_ = w.step(0)
    assert obs2["pos_state"].shape == (8,)


def test_dino_wrapper_rejects_single_pane_and_quaternion():
    enc = _StubEncoder()
    with pytest.raises(ValueError):
        DinoObservationWrapper(_StubEnv(left_pane=False), enc)
    with pytest.raises(ValueError):
        DinoObservationWrapper(_StubEnv(orientation="quaternion"), enc)


def test_pos_state_wrapper():
    w = PosStateWrapper(_StubEnv())
    obs, _ = w.reset()
    assert obs.shape == (8,)
    assert w.observation_space.contains(obs)
