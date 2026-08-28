"""Env-side observation wrappers: ngllib Dict obs -> policy-facing features.

`DinoObservationWrapper` implements agent_plan.md §10/Round 8: the full two-pane
image is split into left (EM) / right (3D) panes, each encoded by a frozen
env-side DINO into a feature vector; scalar viewer state is flattened into
`pos_state`. Output: `Dict(image_features: Box(2*D,), pos_state: Box(8,))`.

`PosStateWrapper` is the pos-only reduction used by the infra smokes (no image).

Both take an injectable `encoder` (anything with `.encode(list[img]) -> (B, D)`
and `.feature_dim`) so tests run with a stub — the real `DinoEncoder` is only
constructed inside an env-runner via the config hook in `env_build.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Raw viewer coordinates are ~1e5 (position) / ~1e4 (projectionScale); feeding
# them into an MLP unscaled swamps the other features. Legacy passed them raw —
# these static divisors are the one deliberate deviation (configurable).
DEFAULT_POS_STATE_SCALE = np.array(
    [1e5, 1e5, 1e5, 1.0, 1.0, 1.0, 1.0, 1e4], dtype=np.float32
)


def split_panes(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a full two-pane screenshot (H, W, 3) into (left, right) halves."""
    mid = image.shape[1] // 2
    return image[:, :mid], image[:, mid:]


def pos_state_from_obs(obs: dict[str, Any], scale: np.ndarray) -> np.ndarray:
    """Flatten position/xs_scale/orientation/proj_scale into a scaled (8,) vector."""
    vec = np.concatenate(
        [
            np.asarray(obs["position"], np.float32).ravel(),
            np.asarray(obs["xs_scale"], np.float32).ravel(),
            np.asarray(obs["orientation"], np.float32).ravel(),
            np.asarray(obs["proj_scale"], np.float32).ravel(),
        ]
    ).astype(np.float32)
    return vec / scale


class DinoObservationWrapper:
    """gymnasium `ObservationWrapper`: two-pane image -> DINO features + pos_state.

    Requires an euler-orientation, both-panes env (pos_state dim 8; the pane
    split assumes the full window). Lazy class def keeps gymnasium out of
    module import for pure-logic tests.
    """

    def __new__(cls, env, encoder, pos_state_scale: np.ndarray | None = None):
        import gymnasium as gym
        from gymnasium import spaces

        base = env.unwrapped
        if not (getattr(base, "left_pane", True) and getattr(base, "right_pane", True)):
            raise ValueError(
                "DinoObservationWrapper needs both panes rendered "
                "(env left_pane=True, right_pane=True) to split EM|3D."
            )
        if getattr(base, "orientation", "euler") != "euler":
            raise ValueError("DinoObservationWrapper requires orientation='euler' (pos_state dim 8).")

        scale = (
            np.asarray(pos_state_scale, np.float32)
            if pos_state_scale is not None
            else DEFAULT_POS_STATE_SCALE
        )
        feat_dim = 2 * int(encoder.feature_dim)

        class _Impl(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                self._encoder = encoder
                self._scale = scale
                self.observation_space = spaces.Dict(
                    {
                        "image_features": spaces.Box(
                            -np.inf, np.inf, shape=(feat_dim,), dtype=np.float32
                        ),
                        "pos_state": spaces.Box(
                            -np.inf, np.inf, shape=(8,), dtype=np.float32
                        ),
                    }
                )

            def observation(self, obs):
                left, right = split_panes(obs["image"])
                feats = self._encoder.encode([left, right])  # (2, D)
                return {
                    "image_features": feats.reshape(-1).astype(np.float32),
                    "pos_state": pos_state_from_obs(obs, self._scale),
                }

        return _Impl(env)


class ServiceFeaturesWrapper:
    """gymnasium `ObservationWrapper` for service-mode native envs: the env
    already returns `image_features` (encoded by the per-node render
    service); this just assembles the same policy-facing Dict as
    `DinoObservationWrapper` — no torch in the client process."""

    def __new__(cls, env, feature_dim: int,
                pos_state_scale: np.ndarray | None = None):
        import gymnasium as gym
        from gymnasium import spaces

        scale = (
            np.asarray(pos_state_scale, np.float32)
            if pos_state_scale is not None
            else DEFAULT_POS_STATE_SCALE
        )

        class _Impl(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                self._scale = scale
                self.observation_space = spaces.Dict(
                    {
                        "image_features": spaces.Box(
                            -np.inf, np.inf, shape=(2 * feature_dim,),
                            dtype=np.float32
                        ),
                        "pos_state": spaces.Box(
                            -np.inf, np.inf, shape=(8,), dtype=np.float32
                        ),
                    }
                )

            def observation(self, obs):
                return {
                    "image_features": np.asarray(
                        obs["image_features"], np.float32),
                    "pos_state": pos_state_from_obs(obs, self._scale),
                }

        return _Impl(env)


class PosStateWrapper:
    """gymnasium `ObservationWrapper`: Dict obs -> flat scaled pos_state Box(8,)."""

    def __new__(cls, env, pos_state_scale: np.ndarray | None = None):
        import gymnasium as gym
        from gymnasium import spaces

        scale = (
            np.asarray(pos_state_scale, np.float32)
            if pos_state_scale is not None
            else DEFAULT_POS_STATE_SCALE
        )

        class _Impl(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                self._scale = scale
                self.observation_space = spaces.Box(
                    -np.inf, np.inf, shape=(8,), dtype=np.float32
                )

            def observation(self, obs):
                return pos_state_from_obs(obs, self._scale)

        return _Impl(env)
