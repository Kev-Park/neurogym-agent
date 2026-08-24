"""Z-navigate reward + termination factories for ngllib 0.2.

Task: reach the target segment's max-z point. Termination fires when the viewer
z is within `z_tolerance` of `task_info["z_max"]`; reward is a sparse success
bonus on termination, otherwise Z-shaping toward the target plus a step penalty.
Ported from the legacy `neurogym-agent/envs/reward.py::compute`, split to fit
ngllib 0.2's separate reward/termination factory hooks.

`task_info` contract (produced by the StateProvider): `{"z_max": float, ...}`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:  # avoid a runtime ngllib/browser import for pure-logic use + tests
    from ngllib import RewardFactory, TerminationFactory


@dataclass(frozen=True)
class ZRewardConfig:
    z_tolerance: float = 10.0
    success: float = 1.0
    z_shaping_coef: float = 0.001
    step_penalty: float = 0.0
    # Proportional band (2026-08-24): tol = max(z_tolerance, frac * z-extent),
    # extent = task_info z_max - z_min. The absolute z_tolerance becomes the
    # floor (guards degenerate flat segments). None = legacy absolute band
    # (v7 and earlier). Requires z_min in task_info when set.
    z_tolerance_frac: float | None = None


def _z(obs: dict[str, Any]) -> float:
    return float(np.asarray(obs["position"])[2])


def effective_z_tolerance(cfg: ZRewardConfig, task_info: dict[str, Any]) -> float:
    """The success band half-width for one episode under `cfg`."""
    if cfg.z_tolerance_frac is None:
        return cfg.z_tolerance
    extent = float(task_info["z_max"]) - float(task_info["z_min"])
    return max(cfg.z_tolerance, cfg.z_tolerance_frac * extent)


def make_z_termination_factory(
    cfg: ZRewardConfig = ZRewardConfig(),
) -> "TerminationFactory":
    """Factory: `task_info -> (obs, action, prev_obs) -> bool`."""

    def factory(task_info: dict[str, Any]) -> Callable[..., bool]:
        z_max = float(task_info["z_max"])
        tol = effective_z_tolerance(cfg, task_info)

        def terminated_fn(obs, action, prev_obs) -> bool:
            return abs(_z(obs) - z_max) <= tol

        return terminated_fn

    return factory


def make_z_reward_factory(cfg: ZRewardConfig = ZRewardConfig()) -> "RewardFactory":
    """Factory: `task_info -> (obs, action, prev_obs, terminated) -> float`.

    ngllib runs termination before reward and passes `terminated` in, so the
    success bonus reuses the termination decision rather than recomputing it.
    """

    def factory(task_info: dict[str, Any]) -> Callable[..., float]:
        z_max = float(task_info["z_max"])

        def reward_fn(obs, action, prev_obs, terminated) -> float:
            if terminated:
                return cfg.success
            z_now, z_prev = _z(obs), _z(prev_obs)
            shaping = cfg.z_shaping_coef * (z_now - z_prev) * np.sign(z_max - z_prev)
            return float(shaping) + cfg.step_penalty

        return reward_fn

    return factory
