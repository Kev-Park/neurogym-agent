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


class _DeadPopen:
    returncode = 1

    def poll(self):
        return 1


def _dead_learner(log_path=None):
    import time as _t

    from ngllib_agent.distributed.coordinator import ManagedProcess

    return ManagedProcess(
        role="learner", cmd="x", node_hint="", popen=_DeadPopen(),
        started_at="t", started_at_mono=_t.monotonic(), log_path=log_path,
    )


def test_workload_ran_discriminator(tmp_path):
    log = tmp_path / "learner-001.log"
    mp = _dead_learner(log)
    assert mp.workload_ran() is False                      # log absent
    log.write_text("srun: error: Unable to allocate resources\n")
    assert mp.workload_ran() is False                      # srun noise only
    log.write_text("srun: launching\n[node] learner pid=1 START\n")
    assert mp.workload_ran() is True                       # real output
    assert _dead_learner(None).workload_ran() is None      # no log configured


def test_fast_crash_respawns_instead_of_promoting(tmp_path, monkeypatch):
    # Quick death WITH workload output = crash -> respawn; empty log = srun
    # denial -> promotion. (The R5 dummy test showed the old heuristic serially
    # sacrificing renderers for a fast-crashing learner.)
    c = _coord(tmp_path)
    promoted, launched = [], []
    monkeypatch.setattr(c, "_promote_renderer_to_learner", lambda cyc: promoted.append(cyc))
    monkeypatch.setattr(c, "_launch", lambda *a, **k: launched.append(a) or object())

    crash_log = tmp_path / "crash.log"
    crash_log.write_text("[node] learner pid=9 START\nTraceback ...\n")
    c.learner = _dead_learner(crash_log)
    c._maybe_respawn_learner(1)
    assert launched and not promoted                       # crash -> respawn

    denial_log = tmp_path / "denial.log"                   # never created
    c.learner = _dead_learner(denial_log)
    c._maybe_respawn_learner(2)
    assert promoted == [2]                                 # denial -> promote


class _AlivePopen:
    returncode = None

    def __init__(self):
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        if not self.terminated:
            raise AssertionError("wait before terminate")


def _alive_learner():
    import time as _t

    from ngllib_agent.distributed.coordinator import ManagedProcess

    return ManagedProcess(
        role="learner", cmd="x", node_hint="", popen=_AlivePopen(),
        started_at="t", started_at_mono=_t.monotonic(),
    )


def test_progress_stall_disabled_by_default(tmp_path):
    import time as _t

    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"iteration": 5}))
    c = _coord(tmp_path, ["--progress-file", str(meta)])
    c.learner = _alive_learner()
    c._progress_changed_mono = _t.monotonic() - 99999
    c._check_progress_stall(1)
    assert c.learner.popen.terminated is False


def test_progress_stall_terminates_wedged_learner(tmp_path):
    import time as _t

    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"iteration": 5}))
    c = _coord(
        tmp_path,
        ["--progress-file", str(meta), "--progress-stall-timeout-s", "60"],
    )
    c.learner = _alive_learner()

    c._check_progress_stall(1)                    # first sighting: arms clock
    assert c._progress_last_val == 5
    assert c.learner.popen.terminated is False

    c._progress_changed_mono = _t.monotonic() - 61   # iteration frozen past timeout
    c._check_progress_stall(2)
    assert c.learner.popen.terminated is True

    # Clock was reset on the kill: the (still-alive popen wrapper) isn't
    # re-terminated every subsequent cycle.
    c.learner = _alive_learner()
    c._check_progress_stall(3)
    assert c.learner.popen.terminated is False


def test_progress_advance_resets_stall_clock(tmp_path):
    import time as _t

    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"iteration": 5}))
    c = _coord(
        tmp_path,
        ["--progress-file", str(meta), "--progress-stall-timeout-s", "60"],
    )
    c.learner = _alive_learner()
    c._check_progress_stall(1)
    c._progress_changed_mono = _t.monotonic() - 61
    meta.write_text(json.dumps({"iteration": 6}))    # progress! no kill
    c._check_progress_stall(2)
    assert c.learner.popen.terminated is False
    assert c._progress_last_val == 6


def test_missing_progress_file_counts_as_stall(tmp_path):
    # A learner that never writes meta.json (wedged before iter 1, beyond
    # startup grace) must still be caught — None reads don't reset the clock.
    import time as _t

    c = _coord(
        tmp_path,
        ["--progress-file", str(tmp_path / "absent.json"),
         "--progress-stall-timeout-s", "60"],
    )
    c.learner = _alive_learner()
    c._progress_changed_mono = _t.monotonic() - 61
    c._check_progress_stall(1)
    assert c.learner.popen.terminated is True


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


def test_transient_cycle_errors_do_not_kill_loop(tmp_path, monkeypatch):
    # The 2026-08-25 suspect: one NFS blip inside a monitor cycle must not
    # end the coordinator — record and continue; max_cycles still honored.
    c = _coord(tmp_path, ["--max-cycles", "3", "--ping-interval", "0.01"])
    c.jobid = "1"
    monkeypatch.setattr(c, "_salloc_lost", lambda: False)
    calls = {"n": 0}

    def boom(cycle):
        calls["n"] += 1
        raise OSError("nfs blip")

    monkeypatch.setattr(c, "_log_status", boom)
    c._monitor_loop()
    assert calls["n"] == 3            # every cycle failed, loop survived
    assert c._exit_reason == "max-cycles"


def test_flight_recorder_best_effort(tmp_path):
    from pathlib import Path

    c = _coord(tmp_path)
    c._flight_paths = [tmp_path / "fr.jsonl", Path("/nonexistent-zz/x.jsonl")]
    c._flight("test-event", foo=1)    # second path fails silently
    rec = json.loads((tmp_path / "fr.jsonl").read_text().strip())
    assert rec["event"] == "test-event" and rec["foo"] == 1


def test_exit_reason_on_signal(tmp_path):
    c = _coord(tmp_path)
    c._flight_paths = [tmp_path / "fr.jsonl"]
    c._sig_stop(15, None)
    assert c.stopped is True
    assert c._exit_reason == "signal-15"
