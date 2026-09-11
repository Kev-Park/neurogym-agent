from __future__ import annotations

import numpy as np
import pytest

from ngllib_agent.wrappers import ActionSpec, cell_to_pixel, decode

SPEC = ActionSpec()


def test_nvec():
    assert SPEC.nvec() == [3, 1024, 9, 9, 9, 9]


def test_cell_to_pixel_within_pane():
    for cell in (0, 1023, 500):
        x, y = cell_to_pixel(cell, SPEC)
        assert SPEC.pane_x0 <= x <= SPEC.pane_x1
        assert SPEC.pane_y0 <= y <= SPEC.pane_y1


def test_right_click_decode():
    act = decode([0, 0, 0, 0, 0, 0], SPEC)
    assert act["action_type"] == 1  # ngllib right_click
    x, y = cell_to_pixel(0, SPEC)
    assert np.allclose(act["mouse_xy"], [x, y])
    assert np.all(act["delta_orient"] == 0)
    assert act["delta_proj_scale"][0] == 0


def test_rotate_decode_center_bin_is_zero():
    # center bin (4) on all axes -> no rotation
    act = decode([1, 0, 4, 4, 4, 0], SPEC)
    assert act["action_type"] == 3  # edit_state
    assert np.allclose(act["delta_orient"], [0, 0, 0])


def test_rotate_decode_signed_magnitude():
    act = decode([1, 0, 8, 0, 4, 0], SPEC)  # +4 step, -4 step, 0
    assert act["action_type"] == 3
    assert np.allclose(
        act["delta_orient"],
        [4 * SPEC.rotation_step_rad, -4 * SPEC.rotation_step_rad, 0.0],
    )
    assert act["mouse_xy"][0] == 0  # no click


def test_zoom_decode():
    act = decode([2, 0, 4, 4, 4, 8], SPEC)  # +4 zoom steps
    assert act["action_type"] == 3
    assert act["delta_proj_scale"][0] == 4 * SPEC.zoom_step
    assert np.all(act["delta_orient"] == 0)


def test_verbs_mutually_exclusive():
    # rotate action leaves click + zoom neutral
    act = decode([1, 500, 8, 8, 8, 8], SPEC)
    assert np.all(act["mouse_xy"] == 0)
    assert act["delta_proj_scale"][0] == 0


def test_dtypes():
    act = decode([0, 0, 0, 0, 0, 0], SPEC)
    assert act["mouse_xy"].dtype == np.float32
    assert act["delta_orient"].dtype == np.float32
    assert act["modifiers"].dtype == np.int8


def test_wrapper_instantiation_and_decode():
    # browser-free: wrap a stub euler env and check space + action() translation.
    import gymnasium as gym
    from gymnasium import spaces

    from ngllib_agent.wrappers import MultiDiscreteActionWrapper

    class _Stub(gym.Env):
        orientation = "euler"

        def __init__(self):
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(1,))
            self.action_space = spaces.Dict({})

        def reset(self, *a, **k):
            return None, {}

        def step(self, a):
            return None, 0.0, False, False, {}

    w = MultiDiscreteActionWrapper(_Stub(), SPEC)
    assert list(w.action_space.nvec) == [3, 1024, 9, 9, 9, 9]
    act = w.action([1, 0, 8, 0, 4, 0])  # rotate
    assert act["action_type"] == 3
    assert np.isclose(act["delta_orient"][0], 4 * SPEC.rotation_step_rad)


def test_wrapper_rejects_quaternion():
    import gymnasium as gym
    from gymnasium import spaces

    from ngllib_agent.wrappers import MultiDiscreteActionWrapper

    class _QStub(gym.Env):
        orientation = "quaternion"

        def __init__(self):
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(1,))
            self.action_space = spaces.Dict({})

    with pytest.raises(ValueError):
        MultiDiscreteActionWrapper(_QStub(), SPEC)


def test_three_verb_spec_matches_legacy_checkpoints():
    """Every checkpoint before 2026-09-10 has a 3-verb head over the 3D-pane
    grid; the spec must still be able to describe it, and a verb-3 sample
    must never decode to a double-click there."""
    legacy = ActionSpec(verbs=3, grid_cols=32, pane_x0=900.0)
    assert legacy.nvec() == [3, 1024, 9, 9, 9, 9]
    act = decode([0, 0, 4, 4, 4, 4], legacy)
    assert act["action_type"] == 1

    with pytest.raises(ValueError):
        ActionSpec(verbs=5)


def test_hierarchical_head_accepts_both_verb_counts():
    torch = pytest.importorskip("torch")
    from ngllib_agent.policies.hierarchical import HierarchicalMultiCategorical

    for verbs in (3, 4):
        cls = HierarchicalMultiCategorical.for_nvec([verbs, 8, 3, 3, 3, 3])
        logits = torch.zeros(2, verbs + 8 + 3 * 3 + 3)
        d = cls.from_logits(logits)
        a = torch.zeros(2, 6, dtype=torch.long)
        a[1, 0] = verbs - 1                       # last verb (zoom for 3, select for 4)
        assert d.logp(a).shape == (2,)
        assert torch.isfinite(d.entropy()).all()
        assert torch.isfinite(d.kl(cls.from_logits(logits + 0.1))).all()
    with pytest.raises(ValueError):
        HierarchicalMultiCategorical.for_nvec([5, 8, 3, 3, 3, 3])
