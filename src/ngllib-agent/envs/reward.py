from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class RewardConfig:
    z_tolerance: float = 10.0
    success: float = 1.0
    z_shaping_coef: float = 0.001
    step_penalty: float = 0.0


def compute(
    state,
    prev_state,
    z_max: float,
    cfg: RewardConfig,
) -> tuple[float, bool]:
    z_now = float(state[0][0][2])
    z_prev = float(prev_state[0][0][2])

    if abs(z_now - z_max) <= cfg.z_tolerance:
        return cfg.success, True

    shaping = cfg.z_shaping_coef * (z_now - z_prev) * np.sign(z_max - z_prev)
    return float(shaping) + cfg.step_penalty, False


def make_env_reward_fn(
    z_max: float,
    cfg: RewardConfig,
) -> Callable[[list, list, list], tuple[float, bool]]:
    """
    Return a closure with the exact signature `ngllib.Environment` expects for
    its `reward_function` constructor argument: `(state, action, prev_state) -> (reward, done)`.
    """

    def reward_fn(state, action, prev_state):
        reward, done = compute(state, prev_state, z_max, cfg)
        return reward, done

    return reward_fn


def load_z_max_table(path: str) -> dict[str, float]:
    import yaml

    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return {str(k): float(v) for k, v in raw.items()}
