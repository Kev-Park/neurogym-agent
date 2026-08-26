"""ngllib-agent coordinator — login-node process manager.

Runs on a login node (in `tmux`/`screen`). Holds a SLURM allocation via
`salloc --no-shell`, launches learner + N renderers via `srun --overlap`,
monitors liveness, writes state to NFS, and cleans up on SIGTERM. See
`agent_plan.md` §4-7 for the full multi-milestone design.

**Milestone 4a scope (this file):** MVP — happy-path launch + monitor + clean
teardown. Later milestones add:
- 4b: process respawn on death
- 4c: renderer→learner promotion when learner srun is denied
- 4d: `salloc` preemption detection + re-request

Usage (typical, from login node inside tmux):

    uv run --no-sync python -m ngllib_agent.distributed.coordinator \
        --state-file /scratch/kp0374/coord-state/run-<id>.json \
        --learner-cmd "uv run --no-sync python scripts/dummy_worker.py --role learner" \
        --renderer-cmd "uv run --no-sync python scripts/dummy_worker.py --role renderer" \
        --renderers 1 \
        --salloc-time 00:15:00 \
        --nodelist sarekl15-3,sarekl15-6

For local self-tests bounded by wall clock, pass `--max-cycles N` so the
coordinator exits cleanly after N monitor cycles.
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import faulthandler
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ngllib_agent.coordinator")


# ============================================================================
# Managed process abstraction
# ============================================================================


@dataclasses.dataclass
class ManagedProcess:
    """A learner or renderer, launched as a backgrounded `srun` subprocess."""

    role: str  # "learner" or "renderer"
    cmd: str  # user-supplied shell command line
    node_hint: str  # "" if unpinned
    popen: subprocess.Popen  # the srun subprocess (parent of the actual work)
    started_at: str  # ISO timestamp, UTC
    started_at_mono: float  # time.monotonic() at launch — used for quick-death detection
    log_path: Optional[Path] = None  # srun --output target, if worker_log_dir set

    def alive(self) -> bool:
        return self.popen.poll() is None

    @property
    def returncode(self) -> Optional[int]:
        return self.popen.returncode

    def died_quickly(self, threshold_s: float) -> bool:
        """True if the process is dead AND died within `threshold_s` of launch.

        A common signature of `srun --immediate=N` failure: the srun subprocess
        exits within roughly N seconds without ever running the workload.
        """
        return (
            not self.alive()
            and (time.monotonic() - self.started_at_mono) <= threshold_s
        )

    def workload_ran(self) -> Optional[bool]:
        """Did the actual workload produce output (vs srun never placing it)?

        Disambiguates quick deaths: srun-denial leaves the log absent/empty or
        containing only `srun:` lines; a fast-crashing workload writes real
        output. None if unknown (no log configured).
        """
        if self.log_path is None:
            return None
        try:
            with open(self.log_path) as f:
                return any(line.strip() and not line.startswith("srun:") for line in f)
        except OSError:
            return False

    def terminate(self, timeout: float = 5.0) -> None:
        """Best-effort SIGTERM + reap; escalate to SIGKILL after timeout."""
        if not self.alive():
            return
        try:
            self.popen.terminate()
        except Exception:
            pass
        try:
            self.popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                self.popen.kill()
                self.popen.wait(timeout=2.0)
            except Exception:
                pass


# ============================================================================
# Coordinator
# ============================================================================


class Coordinator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.state_path = Path(args.state_file)
        self.jobid: Optional[str] = None
        self.learner: Optional[ManagedProcess] = None
        self.renderers: list[ManagedProcess] = []
        self.stopped = False
        self.respawns = {"learner": 0, "renderer": 0, "promoted_renderer_to_learner": 0}
        # Capture the working directory as a STRING and pass it to every
        # subprocess (2026-08-24): a long-lived coordinator's inherited CWD
        # handle goes stale when the login node's automount cycles (observed
        # after a 3h50m salloc wait: every srun died client-side with
        # "getcwd failed: No such file or directory", 112x). An explicit cwd=
        # re-resolves the path at each exec instead.
        self._workdir = os.getcwd()
        # Flight recorder (2026-08-26): the coord-v8 extension coordinator
        # died at 22:05 with no recorded cause, and the relaunch truncated
        # the only log. Record heartbeats/exceptions/exit reasons to an
        # append-only JSONL on BOTH local /tmp (immune to the NFS blips that
        # are the leading suspect) and the NFS state dir (survives login-node
        # reboots). Best-effort per path — never raises.
        self._exit_reason = "unknown"
        self._flight_paths = [
            Path(f"/tmp/coordflight-{args.run_id}.jsonl"),
            self.state_path.parent / f"flight-{args.run_id}.jsonl",
        ]
        self.salloc_resubmissions = 0
        self._teardown_done = False
        self._launch_counter = 0  # monotonic per-worker id for log naming
        self._last_progress: Optional[int] = None  # last iteration seen in progress file
        # R10 progress-stall watchdog: when the iteration counter last CHANGED
        # (monotonic). Reset on every learner launch so startup (browsers +
        # DINO + ray join, ~10 min) never counts against the timeout.
        self._progress_last_val: Optional[int] = None
        self._progress_changed_mono = time.monotonic()
        self._promotion_forced = False  # --force-promotion-once fired yet?
        # Sliding-window log of respawn timestamps (monotonic seconds) for the
        # per-hour circuit breaker. Any older entries are pruned each cycle.
        self._respawn_times: list[float] = []

    # ------------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------------

    def _flight(self, event: str, **kw) -> None:
        """Append one JSON record to every flight-recorder path; never raises."""
        rec = json.dumps({"ts": _utcnow(), "pid": os.getpid(),
                          "event": event, **kw})
        for p in self._flight_paths:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "a") as f:
                    f.write(rec + "\n")
            except OSError:
                pass

    def start(self) -> int:
        signal.signal(signal.SIGTERM, self._sig_stop)
        signal.signal(signal.SIGINT, self._sig_stop)
        atexit.register(self._teardown)
        self._flight("start", host=socket.gethostname(),
                     argv=" ".join(sys.argv)[:500])

        try:
            self._salloc()
        except Exception as e:
            logger.error("salloc failed: %s", e)
            self._exit_reason = f"salloc-failed: {e}"
            return 1

        try:
            self._launch_all()
            self._monitor_loop()
        except Exception:
            logger.exception("coordinator loop crashed")
            self._exit_reason = "loop-crash"
            self._flight("loop-crash", tb=traceback.format_exc()[-3000:])
            return 2

        return 0

    # ------------------------------------------------------------------------
    # SLURM: salloc + srun
    # ------------------------------------------------------------------------

    def _salloc(self) -> None:
        cmd = [
            "salloc", "--no-shell",
            "-N", str(self.args.renderers + 1),
            "-t", self.args.salloc_time,
            "-J", self.args.job_name,
            "-A", self.args.account,
            "-p", self.args.partition,
        ]
        if self.args.salloc_mem:
            cmd += ["--mem", self.args.salloc_mem]
        if self.args.salloc_cpus_per_node:
            cmd += ["--cpus-per-task", str(self.args.salloc_cpus_per_node)]
        if self.args.salloc_gres:
            cmd += ["--gres", self.args.salloc_gres]
        if self.args.nodelist:
            cmd += ["--nodelist", self.args.nodelist]
        if self.args.exclude:
            cmd += ["--exclude", self.args.exclude]

        logger.info("salloc: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self._workdir)
        # salloc prints "salloc: Granted job allocation N" on stderr in most SLURM setups.
        blob = (result.stderr or "") + (result.stdout or "")
        jobid = None
        for line in blob.splitlines():
            if "Granted job allocation" in line:
                jobid = line.strip().split()[-1]
                break
        if jobid is None:
            raise RuntimeError(
                f"could not parse JOBID from salloc output (exit={result.returncode!r}); "
                f"raw:\n{blob!r}"
            )
        self.jobid = jobid
        logger.info("salloc granted JOBID=%s", self.jobid)

    def _launch(
        self, role: str, cmd: str, node_hint: str = ""
    ) -> ManagedProcess:
        self._launch_counter += 1
        srun = [
            "srun",
            f"--jobid={self.jobid}",
            "--overlap",
            "--nodes=1",
            "--ntasks=1",
            f"--immediate={self.args.srun_immediate_timeout}",
        ]
        if node_hint:
            srun += ["-w", node_hint]
        # If a log dir is configured, route srun's task stdout+stderr to a
        # unique file per launch attempt so we can inspect them post-hoc.
        # Otherwise the coordinator inherits stdout (interleaved but visible).
        stdout_target = None
        stderr_target = None
        log_path: Optional[Path] = None
        if self.args.worker_log_dir:
            log_dir = Path(self.args.worker_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{role}-{self._launch_counter:03d}.log"
            srun += [f"--output={log_path}", f"--error={log_path}"]
        # Wrap user cmd in a login shell so `uv run` etc. resolve on remote.
        full = srun + ["bash", "-lc", cmd]
        logger.info("srun[%s #%d]: %s", role, self._launch_counter, " ".join(full))
        p = subprocess.Popen(
            full,
            stdout=stdout_target,   # None -> inherit; or DEVNULL if --output pinned
            stderr=stderr_target,
            stdin=subprocess.DEVNULL,
            cwd=self._workdir,
        )
        if role == "learner":
            self._progress_changed_mono = time.monotonic()
        return ManagedProcess(
            role=role,
            cmd=cmd,
            node_hint=node_hint,
            popen=p,
            started_at=_utcnow(),
            started_at_mono=time.monotonic(),
            log_path=log_path,
        )

    def _launch_all(self) -> None:
        self.learner = self._launch("learner", self.args.learner_cmd)
        for _ in range(self.args.renderers):
            self.renderers.append(self._launch("renderer", self.args.renderer_cmd))
        logger.info(
            "launched learner + %d renderers (pids: learner=%s, renderers=%s)",
            len(self.renderers),
            self.learner.popen.pid if self.learner else "?",
            [r.popen.pid for r in self.renderers],
        )

    # ------------------------------------------------------------------------
    # Monitor loop
    # ------------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        cycle = 0
        consec_errors = 0
        while not self.stopped:
            cycle += 1
            if self.args.max_cycles and cycle > self.args.max_cycles:
                logger.info("reached max_cycles=%d; exiting", self.args.max_cycles)
                self._exit_reason = "max-cycles"
                break

            # A transient failure in ONE cycle (NFS blip on squeue/state
            # writes — the leading suspect for the unexplained 2026-08-25
            # coordinator death) must not kill the coordinator: record it and
            # keep monitoring. Only sustained failure (~10 min of consecutive
            # errors) gives up, with the reason on record.
            try:
                # M4d: check the salloc allocation is still alive; if not,
                # re-request and relaunch all processes. Happens BEFORE
                # _handle_deaths because a preempted allocation will have
                # killed every managed process simultaneously and any respawn
                # attempt would fail.
                if self._salloc_lost():
                    self._resalloc_and_relaunch(cycle)
                    self._write_state(cycle)
                    consec_errors = 0
                    time.sleep(self.args.ping_interval)
                    continue

                # R1: iteration-based completion — training progress is the
                # RL-native "done" measure. train.py heartbeats meta.json
                # each iter.
                if self._target_reached():
                    logger.info(
                        "target iterations reached (%d >= %d); tearing down",
                        self._last_progress or -1, self.args.target_iterations,
                    )
                    self._exit_reason = "target-reached"
                    break

                self._log_status(cycle)
                self._check_progress_stall(cycle)  # R10
                self._handle_deaths(cycle)   # M4b, M4c
                self._write_state(cycle)
                if cycle % 60 == 0:  # ~5 min at the 5s default interval
                    self._flight("heartbeat", cycle=cycle, jobid=self.jobid)
                consec_errors = 0
            except Exception:
                consec_errors += 1
                logger.exception(
                    "cycle %d body failed (consecutive=%d); continuing",
                    cycle, consec_errors,
                )
                self._flight("cycle-error", cycle=cycle, consec=consec_errors,
                             tb=traceback.format_exc()[-2000:])
                if consec_errors >= 120:
                    self._exit_reason = "persistent-cycle-errors"
                    logger.error("120 consecutive cycle failures; giving up")
                    break

            time.sleep(self.args.ping_interval)

    # ------------------------------------------------------------------------
    # M4b: respawn dead processes (learner + renderers) with a circuit breaker.
    # M4c (renderer->learner promotion on srun-immediate failure) is separate.
    # ------------------------------------------------------------------------

    def _handle_deaths(self, cycle: int) -> None:
        # Prune respawn history to just the last hour.
        now = time.monotonic()
        self._respawn_times = [t for t in self._respawn_times if now - t < 3600]

        if self.learner is not None and not self.learner.alive():
            self._maybe_respawn_learner(cycle)

        for i, r in enumerate(self.renderers):
            if not r.alive():
                self._maybe_respawn_renderer(cycle, i)

    def _circuit_ok(self) -> bool:
        """Return True if we're allowed another respawn under the per-hour cap."""
        cap = self.args.max_respawn_per_hour
        if cap <= 0:
            return True
        if len(self._respawn_times) >= cap:
            logger.error(
                "circuit breaker OPEN: %d respawns in the last hour "
                "(cap=%d); refusing further respawns",
                len(self._respawn_times), cap,
            )
            return False
        return True

    def _maybe_respawn_learner(self, cycle: int) -> None:
        dead = self.learner
        # TEST HOOK (R5): force the promotion branch once, deterministically —
        # genuine srun --immediate denial is hard to stage under --overlap.
        if self.args.force_promotion_once and not self._promotion_forced:
            self._promotion_forced = True
            logger.warning(
                "cycle %d: TEST HOOK force-promotion-once: skipping respawn, "
                "exercising renderer->learner promotion", cycle,
            )
            self._promote_renderer_to_learner(cycle)
            return
        # If the *previous* learner (which is now dead) died within the
        # srun-immediate window AND its workload never produced output, that's
        # the signature of srun --immediate=N denial: no slot in the allocation.
        # A fast-CRASHING workload also dies quickly but leaves real log output
        # — that must go down the normal respawn path, or we'd serially
        # sacrifice healthy renderers for a learner that can't run (observed
        # in the R5 dummy test, 2026-07-08).
        immediate_denial = (
            dead is not None
            and dead.died_quickly(self.args.srun_immediate_timeout + 10)
            and dead.workload_ran() is not True
        )
        if immediate_denial:
            logger.warning(
                "cycle %d: learner DEAD within srun-immediate window "
                "(likely allocation full); attempting renderer promotion",
                cycle,
            )
            self._promote_renderer_to_learner(cycle)
            return

        logger.warning(
            "cycle %d: learner DEAD (exit=%s); respawning",
            cycle, dead.returncode if dead else "?",
        )
        if not self._circuit_ok():
            return
        try:
            self.learner = self._launch("learner", self.args.learner_cmd)
        except OSError as e:
            logger.error("srun failed while respawning learner: %s", e)
            return
        self.respawns["learner"] += 1
        self._respawn_times.append(time.monotonic())

    # ------------------------------------------------------------------------
    # M4d: detect salloc preemption/expiry and re-request the allocation.
    # ------------------------------------------------------------------------

    def _salloc_lost(self) -> bool:
        """Ask SLURM whether our allocation is still LIVE (JOBID present AND
        RUNNING). Returns True if we've lost the slot in either sense:

          - JOBID gone from squeue: job was scanceled/completed/killed for good.
          - JOBID present but state != RUNNING (PENDING, REQUEUED, SUSPENDED,
            CONFIGURING, ...): common preempt/requeue case on `preempt` partition,
            where SLURM sets `PreemptMode=REQUEUE` — the sruns still die but the
            JOBID sticks around waiting for another slot.

        Both mean we can't launch new steps against $JOBID and must respond.
        """
        if self.jobid is None:
            return True
        try:
            r = subprocess.run(
                ["squeue", "-j", self.jobid, "-h", "-o", "%i %T"],
                capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            logger.warning("squeue timed out; assuming allocation still up for now")
            return False
        out = r.stdout.strip()
        if not out:
            logger.warning("salloc lost: JOBID=%s absent from squeue", self.jobid)
            return True
        # Format: "JOBID STATE"
        parts = out.split()
        state = parts[1] if len(parts) >= 2 else ""
        if state and state != "RUNNING":
            logger.warning("salloc lost: JOBID=%s state=%s (not RUNNING)", self.jobid, state)
            return True
        return False

    def _resalloc_and_relaunch(self, cycle: int) -> None:
        logger.warning(
            "cycle %d: re-requesting salloc (old JOBID=%s)", cycle, self.jobid,
        )
        # Best-effort: terminate any lingering popen wrappers. Their
        # underlying srun/task should already be dead but be defensive.
        if self.learner:
            self.learner.terminate()
        for r in self.renderers:
            r.terminate()

        old_jobid = self.jobid
        # If the OLD JOBID is still around (e.g. REQUEUED / PENDING after a
        # preempt), scancel it before re-requesting so it doesn't come back
        # later, hold nodes we don't want, and compete with the fresh salloc.
        if old_jobid:
            subprocess.run(
                ["scancel", old_jobid], check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        self.jobid = None
        try:
            self._salloc()
        except Exception as e:
            logger.error("re-salloc failed: %s (old JOBID=%s)", e, old_jobid)
            # Nothing we can do; the next cycle will retry.
            return

        self.salloc_resubmissions += 1
        # Rebuild the pool from scratch on the new allocation.
        self.learner = None
        self.renderers = []
        try:
            self._launch_all()
        except OSError as e:
            logger.error("relaunch after re-salloc failed: %s", e)

    def _promote_renderer_to_learner(self, cycle: int) -> None:
        """Kill the oldest live renderer, freeing its slot for a learner srun.

        Precondition: previous learner death was inside srun --immediate window,
        suggesting the allocation had no free node to accept a fresh learner.
        Sacrifice a renderer to make room.
        """
        if not self._circuit_ok():
            return
        # Pick oldest ALIVE renderer (dead ones don't hold slots).
        candidates = [(i, r) for i, r in enumerate(self.renderers) if r.alive()]
        if not candidates:
            logger.error(
                "cycle %d: no live renderer to promote; learner stays dead this cycle",
                cycle,
            )
            return
        candidates.sort(key=lambda t: t[1].started_at_mono)  # oldest first
        idx, victim = candidates[0]
        logger.warning(
            "cycle %d: promoting: killing renderer[%d] (started %s) to free slot for learner",
            cycle, idx, victim.started_at,
        )
        victim.terminate()
        # Give SLURM a moment to release the node before the learner srun tries
        # to claim it.
        time.sleep(self.args.promotion_slot_wait_s)

        try:
            self.learner = self._launch("learner", self.args.learner_cmd)
        except OSError as e:
            logger.error("srun failed while promoting to learner: %s", e)
            return
        self.respawns["learner"] += 1
        self.respawns["promoted_renderer_to_learner"] += 1
        self._respawn_times.append(time.monotonic())

        # The renderer we killed leaves a dead slot in self.renderers[]; on the
        # next cycle _handle_deaths will try to respawn it. That's the intended
        # steady-state: pool tries to recover to N renderers whenever possible.

    def _maybe_respawn_renderer(self, cycle: int, idx: int) -> None:
        dead = self.renderers[idx]
        logger.warning(
            "cycle %d: renderer[%d] DEAD (exit=%s); respawning",
            cycle, idx, dead.returncode,
        )
        if not self._circuit_ok():
            return
        try:
            self.renderers[idx] = self._launch("renderer", self.args.renderer_cmd)
        except OSError as e:
            logger.error("srun failed while respawning renderer[%d]: %s", idx, e)
            return
        self.respawns["renderer"] += 1
        self._respawn_times.append(time.monotonic())

    def _check_progress_stall(self, cycle: int) -> None:
        """R10 outermost net: learner ALIVE but the iteration counter frozen.

        The train.py degraded-exit detector (R10 rung 1) can't fire if the
        training loop itself is wedged (e.g. an NFS checkpoint write hangs —
        sample_timeout_s bounds sampling, not everything). If meta.json's
        iteration hasn't advanced within the timeout, SIGTERM the learner;
        _handle_deaths respawns it next cycle (counts toward the circuit
        breaker like any respawn).
        """
        if self.args.progress_stall_timeout_s <= 0 or not self.args.progress_file:
            return
        now = time.monotonic()
        it = read_progress_iteration(self.args.progress_file)
        if it is not None and it != self._progress_last_val:
            self._progress_last_val = it
            self._progress_changed_mono = now
            return
        stalled_for = now - self._progress_changed_mono
        if (
            self.learner is not None
            and self.learner.alive()
            and stalled_for > self.args.progress_stall_timeout_s
        ):
            logger.warning(
                "cycle %d: progress STALLED (iteration=%s unchanged for %.0fs "
                "> %.0fs) with learner nominally ALIVE; terminating learner "
                "for respawn",
                cycle, self._progress_last_val, stalled_for,
                self.args.progress_stall_timeout_s,
            )
            self.learner.terminate()
            self._progress_changed_mono = now

    def _target_reached(self) -> bool:
        if self.args.target_iterations <= 0 or not self.args.progress_file:
            return False
        it = read_progress_iteration(self.args.progress_file)
        if it is not None:
            self._last_progress = it
        return it is not None and it >= self.args.target_iterations

    def _log_status(self, cycle: int) -> None:
        alive_learner = self.learner is not None and self.learner.alive()
        alive_r = sum(1 for r in self.renderers if r.alive())
        logger.info(
            "cycle %d: learner=%s renderers=%d/%d alive",
            cycle,
            "ALIVE" if alive_learner else f"DEAD(exit={self.learner.returncode if self.learner else '?'})",
            alive_r,
            len(self.renderers),
        )
        # Individual death reports so we can see the transition.
        for i, r in enumerate(self.renderers):
            if not r.alive():
                logger.warning("  renderer[%d] DEAD  exit=%s  cmd=%r",
                               i, r.returncode, r.cmd[:80])

    # ------------------------------------------------------------------------
    # State persistence (NFS-safe atomic write)
    # ------------------------------------------------------------------------

    def _write_state(self, cycle: int) -> None:
        state = {
            "run_id": self.args.run_id,
            "cycle": cycle,
            "coordinator_host": socket.gethostname(),
            "jobid": self.jobid,
            "learner": self._pinfo(self.learner),
            "renderers": [self._pinfo(r) for r in self.renderers],
            "respawns": dict(self.respawns),
            "salloc_resubmissions": self.salloc_resubmissions,
            "updated_at": _utcnow(),
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, self.state_path)  # atomic on same fs

    @staticmethod
    def _pinfo(p: Optional[ManagedProcess]) -> Optional[dict]:
        if p is None:
            return None
        return {
            "role": p.role,
            "cmd": p.cmd,
            "node_hint": p.node_hint,
            "srun_pid": p.popen.pid,
            "alive": p.alive(),
            "returncode": p.returncode,
            "started_at": p.started_at,
        }

    # ------------------------------------------------------------------------
    # Signals + teardown
    # ------------------------------------------------------------------------

    def _sig_stop(self, signum, frame):  # noqa: ARG002
        logger.info("received signal %d; stopping cleanly", signum)
        self._exit_reason = f"signal-{signum}"
        self._flight("signal", signum=signum)
        self.stopped = True

    def _teardown(self) -> None:
        if self._teardown_done:
            return
        self._teardown_done = True
        exc = sys.exc_info()
        self._flight(
            "teardown",
            reason=self._exit_reason,
            jobid=self.jobid,
            respawns=dict(self.respawns),
            in_flight_exc=(
                "".join(traceback.format_exception(*exc))[-2000:]
                if exc[0] else None
            ),
        )
        logger.info("teardown: terminating processes (reason=%s)", self._exit_reason)
        if self.learner:
            self.learner.terminate()
        for r in self.renderers:
            r.terminate()
        if self.jobid:
            logger.info("teardown: scancel %s", self.jobid)
            subprocess.run(["scancel", self.jobid], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ============================================================================
# Helpers
# ============================================================================


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_progress_iteration(path: str | Path) -> Optional[int]:
    """Best-effort `iteration` from a train.py meta.json. None if unreadable
    (missing file, partial write, no key) — callers treat that as 'no data'."""
    try:
        with open(path) as f:
            return int(json.load(f)["iteration"])
    except Exception:
        return None


# ============================================================================
# CLI
# ============================================================================


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ngllib_agent.distributed.coordinator",
        description="ngllib-agent login-node coordinator (M4a MVP).",
    )
    ap.add_argument("--run-id", default=f"run-{int(time.time())}")
    ap.add_argument("--state-file", required=True,
                    help="Path to NFS-visible state JSON (atomic tmp+rename).")
    ap.add_argument("--renderers", type=int, default=1)
    ap.add_argument("--learner-cmd", required=True,
                    help="Shell command to run as the learner (via bash -lc).")
    ap.add_argument("--renderer-cmd", required=True,
                    help="Shell command to run as each renderer (via bash -lc).")

    ap.add_argument("--salloc-time", default="00:30:00")
    ap.add_argument("--salloc-mem", default="16G",
                    help="Passed to salloc --mem. Empty string disables.")
    ap.add_argument("--salloc-cpus-per-node", type=int, default=4,
                    help="Passed to salloc --cpus-per-task. 0 disables.")
    ap.add_argument("--salloc-gres", default="",
                    help='e.g. "gpu:3090:1". Empty disables --gres.')

    ap.add_argument("--partition", default="preempt")
    ap.add_argument("--account", default="pni")
    ap.add_argument("--job-name", default="ngllib-agent-coord")
    ap.add_argument("--nodelist", default="",
                    help="Comma-separated whitelist (SLURM --nodelist).")
    ap.add_argument("--exclude", default="",
                    help="Comma-separated blacklist (SLURM --exclude).")

    ap.add_argument("--srun-immediate-timeout", type=int, default=60,
                    help="Passed to each srun as --immediate=N.")
    ap.add_argument("--ping-interval", type=float, default=5.0,
                    help="Seconds between monitor cycles.")
    ap.add_argument("--max-cycles", type=int, default=0,
                    help="If >0, exit cleanly after this many cycles (for tests).")
    ap.add_argument("--target-iterations", type=int, default=0,
                    help="If >0, tear down once --progress-file shows "
                         "iteration >= this (RL-native completion; R1).")
    ap.add_argument("--progress-file", default="",
                    help="Path to train.py's meta.json (heartbeated every iter).")
    ap.add_argument("--progress-stall-timeout-s", type=float, default=0.0,
                    help="If >0 and --progress-file is set: SIGTERM a nominally "
                         "ALIVE learner whose iteration counter hasn't advanced "
                         "in this many seconds (R10 wedge net; respawn follows). "
                         "0 disables. Must comfortably exceed worst-case startup "
                         "+ one slow iteration (~1800 for the 2x16 topology).")
    ap.add_argument("--force-promotion-once", action="store_true",
                    help="TEST ONLY: on the first learner death, skip respawn "
                         "and exercise the renderer->learner promotion branch.")
    ap.add_argument("--worker-log-dir", default="",
                    help="If set, route each srun's --output/--error to a "
                         "per-launch file under this directory. Otherwise "
                         "workers inherit the coordinator's stdout/stderr.")
    ap.add_argument("--max-respawn-per-hour", type=int, default=20,
                    help="Circuit breaker: refuse further respawns once this "
                         "many have fired in a sliding 1-hour window. "
                         "0 disables the cap.")
    ap.add_argument("--promotion-slot-wait-s", type=float, default=3.0,
                    help="Seconds to wait after killing a renderer before "
                         "srun-ing the learner into the freed slot. Gives "
                         "SLURM time to reap the previous step allocation.")

    return ap


_FAULT_LOG = None  # module ref: faulthandler requires the file to stay open


def main(argv: Optional[list[str]] = None) -> int:
    global _FAULT_LOG
    ap = build_argparser()
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Hard-crash forensics (segfault/fatal signal → C-level traceback) on
    # LOCAL disk, since NFS may be the thing that's failing.
    try:
        _FAULT_LOG = open(f"/tmp/coordfault-{args.run_id}.log", "a")
        faulthandler.enable(_FAULT_LOG)
    except OSError:
        pass

    coord = Coordinator(args)
    return coord.start()


if __name__ == "__main__":
    sys.exit(main())
