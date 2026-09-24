from __future__ import annotations

import numpy as np
import pytest

from ngllib_agent.wrappers import ActionSpec, cell_to_pixel, decode

SPEC = ActionSpec()


def test_nvec():
    # 4 verbs (right-click, rotate, zoom, select) over a 32x64 grid spanning both panes
    assert SPEC.nvec() == [4, 2048, 9, 9, 9, 9]


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
    assert list(w.action_space.nvec) == [4, 2048, 9, 9, 9, 9]
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

    for verbs in (3, 4, 5):
        cls = HierarchicalMultiCategorical.for_nvec([verbs, 8, 3, 3, 3, 3])
        logits = torch.zeros(2, verbs + 8 + 3 * 3 + 3)
        d = cls.from_logits(logits)
        a = torch.zeros(2, 6, dtype=torch.long)
        a[1, 0] = verbs - 1        # last verb: zoom (3), select (4), xs_zoom (5)
        assert d.logp(a).shape == (2,)
        assert torch.isfinite(d.entropy()).all()
        assert torch.isfinite(d.kl(cls.from_logits(logits + 0.1))).all()
    with pytest.raises(ValueError):
        HierarchicalMultiCategorical.for_nvec([6, 8, 3, 3, 3, 3])


def test_xs_zoom_verb_edits_the_2d_pane_only():
    """Verb 4 moves crossSectionScale, the 2D pane's zoom, and nothing else.

    The 2D pane's zoom was unreachable by any policy until 2026-09-24: the zoom
    verb wrote only delta_proj_scale (the 3D camera), so crossSectionScale was
    fixed for a whole episode at whatever the reset state carried.
    """
    spec = ActionSpec(verbs=5, xs_zoom_step=0.25, zoom_bins=9)
    assert spec.nvec() == [5, 2048, 9, 9, 9, 9]

    zoom_in = decode([4, 0, 4, 4, 4, 6], spec)          # 2 bins above centre
    assert zoom_in["action_type"] == 3                          # edit_state
    assert zoom_in["delta_xs_scale"][0] == pytest.approx(0.5)
    assert zoom_in["delta_proj_scale"][0] == 0.0                # 3D camera untouched
    assert zoom_in["delta_pos"].tolist() == [0.0, 0.0, 0.0]
    assert zoom_in["delta_orient"].tolist() == [0.0, 0.0, 0.0]

    zoom_out = decode([4, 0, 4, 4, 4, 2], spec)
    assert zoom_out["delta_xs_scale"][0] == pytest.approx(-0.5)
    assert decode([4, 0, 4, 4, 4, 4], spec)["delta_xs_scale"][0] == 0.0   # centre bin


def test_the_zoom_verbs_are_independent():
    """Verb 2 must not touch the 2D zoom, and verb 4 must not touch the 3D."""
    spec = ActionSpec(verbs=5)
    three_d = decode([2, 0, 4, 4, 4, 6], spec)
    assert three_d["delta_proj_scale"][0] != 0.0
    assert three_d["delta_xs_scale"][0] == 0.0


def test_a_four_verb_spec_cannot_emit_xs_zoom():
    spec = ActionSpec(verbs=4)
    assert spec.nvec()[0] == 4
    with pytest.raises(ValueError):
        decode([4, 0, 4, 4, 4, 6], spec)


def test_verbs_must_be_3_4_or_5():
    with pytest.raises(ValueError):
        ActionSpec(verbs=6)
