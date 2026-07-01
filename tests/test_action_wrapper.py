from __future__ import annotations

import numpy as np

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
