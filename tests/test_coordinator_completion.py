from __future__ import annotations

import json

from ngllib_agent.distributed.coordinator import (
    Coordinator,
    build_argparser,
    read_progress_iteration,
)


def _coord(tmp_path, extra=()):
    args = build_argparser().parse_args(
        [
            "--state-file", str(tmp_path / "state.json"),
            "--learner-cmd", "true",
            "--renderer-cmd", "true",
            *extra,
        ]
    )
    return Coordinator(args)


def test_read_progress_iteration(tmp_path):
    p = tmp_path / "meta.json"
    assert read_progress_iteration(p) is None                    # missing
    p.write_text("{not json")
    assert read_progress_iteration(p) is None                    # corrupt
    p.write_text(json.dumps({"wandb_id": "x"}))
    assert read_progress_iteration(p) is None                    # no key
    p.write_text(json.dumps({"iteration": 42}))
    assert read_progress_iteration(p) == 42


def test_target_reached_logic(tmp_path):
    meta = tmp_path / "meta.json"
    c = _coord(
        tmp_path,
        ["--target-iterations", "100", "--progress-file", str(meta)],
    )
    assert c._target_reached() is False          # no file yet -> keep running
    meta.write_text(json.dumps({"iteration": 99}))
    assert c._target_reached() is False
    assert c._last_progress == 99
    meta.write_text(json.dumps({"iteration": 100}))
    assert c._target_reached() is True
    meta.write_text("{corrupt mid-write")        # transient bad read -> no data
    assert c._target_reached() is False
    assert c._last_progress == 100               # last good value retained


def test_target_disabled_by_default(tmp_path):
    c = _coord(tmp_path)
    assert c._target_reached() is False
    c2 = _coord(tmp_path, ["--target-iterations", "10"])  # no progress file
    assert c2._target_reached() is False


def test_force_promotion_once_routes_to_promotion(tmp_path, monkeypatch):
    c = _coord(tmp_path, ["--force-promotion-once"])
    promoted, launched = [], []
    monkeypatch.setattr(c, "_promote_renderer_to_learner", lambda cyc: promoted.append(cyc))
    monkeypatch.setattr(c, "_launch", lambda *a, **k: launched.append(a) or object())

    c._maybe_respawn_learner(1)          # first death -> forced promotion
    assert promoted == [1] and launched == []
    assert c._promotion_forced is True

    c._maybe_respawn_learner(2)          # second death -> normal respawn path
    assert promoted == [1] and len(launched) == 1
