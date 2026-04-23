from dataclasses import dataclass
from typing import Callable

import numpy as np

RIGHT_CLICK_IDX = 1


@dataclass
class RewardConfig:
    z_tolerance: float = 10.0
    success: float = 1.0
    noop_penalty: float = -0.01
    noop_position_eps: float = 0.5
    z_shaping_coef: float = 0.001


def compute(
    state,
    prev_state,
    right_click_fired: bool,
    z_max: float,
    cfg: RewardConfig,
) -> tuple[float, bool, bool]:
    z_now = float(state[0][0][2])
    z_prev = float(prev_state[0][0][2])

    if abs(z_now - z_max) <= cfg.z_tolerance:
        return cfg.success, True, False

    if right_click_fired:
        pos_now = np.asarray(state[0][0], dtype=np.float64)
        pos_prev = np.asarray(prev_state[0][0], dtype=np.float64)
        if np.linalg.norm(pos_now - pos_prev) < cfg.noop_position_eps:
            return cfg.noop_penalty, False, True

    # Dense shaping: reward progress toward z_max, penalise moving away.
    # Scaled small so it doesn't overwhelm the sparse success signal.
    shaping = cfg.z_shaping_coef * (z_now - z_prev) * np.sign(z_max - z_prev)
    return float(shaping), False, False


def make_env_reward_fn(
    z_max: float,
    cfg: RewardConfig,
) -> Callable[[list, list, list], tuple[float, bool]]:
    """
    Return a closure with the exact signature `ngllib.Environment` expects for
    its `reward_function` constructor argument: `(state, action, prev_state) -> (reward, done)`.

    This lets the same reward live inside `Environment` directly — useful for
    evaluation / manual rollouts that bypass the Gym wrapper.
    """

    def reward_fn(state, action, prev_state):
        right_click_fired = bool(action[RIGHT_CLICK_IDX]) if action is not None else False
        reward, done, _was_noop = compute(state, prev_state, right_click_fired, z_max, cfg)
        return reward, done

    return reward_fn


def load_z_max_table(path: str) -> dict[str, float]:
    import yaml

    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return {str(k): float(v) for k, v in raw.items()}
