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


def test_v3_budget_termination_samples_within_range():
    from ngllib_agent.rewards import make_v3_termination_factory

    factory = make_v3_termination_factory(budget_min=5, budget_max=8,
                                          stall_limit=None, seed=0)
    for _ in range(20):
        term = factory({})
        steps = 0
        # keep improving z so only the budget can fire
        while not term(_obs(steps + 1.0), None, _obs(float(steps))):
            steps += 1
            assert steps < 9
        assert 5 <= steps + 1 <= 8


def test_v3_stall_termination_resets_on_new_best():
    from ngllib_agent.rewards import make_v3_termination_factory

    factory = make_v3_termination_factory(budget_min=1000, budget_max=1000,
                                          stall_limit=3, seed=0)
    term = factory({})
    # two stalled steps, then a new best resets the counter
    assert term(_obs(0.0), None, _obs(0.0)) is False
    assert term(_obs(-1.0), None, _obs(0.0)) is False
    assert term(_obs(5.0), None, _obs(0.0)) is False
    # three consecutive non-improving steps fire the stall terminal
    assert term(_obs(5.0), None, _obs(0.0)) is False
    assert term(_obs(4.0), None, _obs(0.0)) is False
    assert term(_obs(3.0), None, _obs(0.0)) is True


def test_v4_delta_and_select_cost():
    from ngllib_agent.rewards import ZFreeRewardConfig, make_zfree_reward_factory

    cfg = ZFreeRewardConfig(shaping_coef=0.001, select_novelty_bonus=0.0,
                            delta_coef=0.0003, select_cost=0.003)
    rew = make_zfree_reward_factory(cfg)({})

    def _o(z, segs=()):
        return {"position": np.array([0.0, 0.0, float(z)], dtype=np.float32),
                "segments": tuple(segs)}

    # ascent: best-so-far + delta both pay
    r = rew(_o(100.0), None, _o(0.0), False)
    assert abs(r - (0.001 * 100 + 0.0003 * 100)) < 1e-9
    # descent from the peak: only the delta term (negative), no potential
    r = rew(_o(60.0), None, _o(100.0), False)
    assert abs(r - (0.0003 * -40)) < 1e-9
    # two NEW segments selected: charged select_cost each (bonus disabled)
    r = rew(_o(60.0, ("a", "b")), None, _o(60.0), False)
    assert abs(r - (-0.003 * 2)) < 1e-9
    # reselect of an already-seen segment: free
    r = rew(_o(60.0, ("a",)), None, _o(60.0, ()), False)
    assert abs(r) < 1e-9
