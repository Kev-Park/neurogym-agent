from __future__ import annotations

import numpy as np

from ngllib_agent.rewards import (
    ZRewardConfig,
    make_z_reward_factory,
    make_z_termination_factory,
)


def _obs(z):
    return {"position": np.array([0.0, 0.0, float(z)], dtype=np.float32)}


def test_termination_within_tolerance():
    term = make_z_termination_factory(ZRewardConfig(z_tolerance=10.0))({"z_max": 100.0})
    assert term(_obs(95.0), None, _obs(0.0)) is True
    assert term(_obs(100.0), None, _obs(0.0)) is True
    assert term(_obs(80.0), None, _obs(0.0)) is False


def test_reward_success_on_terminated():
    rew = make_z_reward_factory(ZRewardConfig(success=1.0))({"z_max": 100.0})
    assert rew(_obs(100.0), None, _obs(0.0), True) == 1.0


def test_reward_shaping_sign_and_step_penalty():
    cfg = ZRewardConfig(z_shaping_coef=0.001, step_penalty=-0.01)
    rew = make_z_reward_factory(cfg)({"z_max": 100.0})
    # moved +10 toward z_max (which is above): positive shaping, minus step penalty
    r_toward = rew(_obs(10.0), None, _obs(0.0), False)
    assert r_toward == 0.001 * 10 * 1.0 - 0.01
    # moved away from z_max: negative shaping
    r_away = rew(_obs(0.0), None, _obs(10.0), False)
    assert r_away == 0.001 * (-10) * 1.0 - 0.01


def test_reward_shaping_direction_when_target_below():
    rew = make_z_reward_factory(ZRewardConfig(step_penalty=0.0))({"z_max": -100.0})
    # target below start; moving down (negative) should be positive shaping
    assert rew(_obs(-10.0), None, _obs(0.0), False) > 0


def test_frac_tolerance_termination():
    from ngllib_agent.rewards import (
        ZRewardConfig,
        effective_z_tolerance,
        make_z_termination_factory,
    )

    cfg = ZRewardConfig(z_tolerance=10.0, z_tolerance_frac=0.05)
    ti = {"z_max": 1000.0, "z_min": 0.0}  # extent 1000 -> tol 50
    assert effective_z_tolerance(cfg, ti) == 50.0
    fn = make_z_termination_factory(cfg)(ti)
    assert fn({"position": [0, 0, 951.0]}, None, None) is True
    assert fn({"position": [0, 0, 949.0]}, None, None) is False

    # abs floor guards tiny extents: 5% of 100 = 5 < 10 -> 10 wins
    ti_small = {"z_max": 100.0, "z_min": 0.0}
    assert effective_z_tolerance(cfg, ti_small) == 10.0

    # frac unset -> legacy absolute band, z_min not required
    legacy = ZRewardConfig(z_tolerance=10.0)
    assert effective_z_tolerance(legacy, {"z_max": 1000.0}) == 10.0
