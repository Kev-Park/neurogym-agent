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
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
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

    def alive(self) -> bool:
        return self.popen.poll() is None

    @property
    def returncode(self) -> Optional[int]:
        return self.popen.returncode

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
        self.respawns = {"learner": 0, "renderer": 0}
        self._teardown_done = False
        self._launch_counter = 0  # monotonic per-worker id for log naming
        # Sliding-window log of respawn timestamps (monotonic seconds) for the
        # per-hour circuit breaker. Any older entries are pruned each cycle.
        self._respawn_times: list[float] = []

    # ------------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------------

    def start(self) -> int:
        signal.signal(signal.SIGTERM, self._sig_stop)
        signal.signal(signal.SIGINT, self._sig_stop)
        atexit.register(self._teardown)

        try:
            self._salloc()
        except Exception as e:
            logger.error("salloc failed: %s", e)
            return 1

        try:
            self._launch_all()
            self._monitor_loop()
        except Exception:
            logger.exception("coordinator loop crashed")
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
        result = subprocess.run(cmd, capture_output=True, text=True)
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
        )
        return ManagedProcess(
            role=role,
            cmd=cmd,
            node_hint=node_hint,
            popen=p,
            started_at=_utcnow(),
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
        while not self.stopped:
            cycle += 1
            if self.args.max_cycles and cycle > self.args.max_cycles:
                logger.info("reached max_cycles=%d; exiting", self.args.max_cycles)
                break

            self._log_status(cycle)
            self._handle_deaths(cycle)   # M4b
            self._write_state(cycle)

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
        logger.warning(
            "cycle %d: learner DEAD (exit=%s); respawning",
            cycle, dead.returncode if dead else "?",
        )
        if not self._circuit_ok():
            return
        try:
            self.learner = self._launch("learner", self.args.learner_cmd)
        except OSError as e:
            # Popen itself failed (rare — usually srun succeeds even if the
            # target work later dies). Leave `self.learner` at the dead
            # process for reporting and try again next cycle.
            logger.error("srun failed while respawning learner: %s", e)
            return
        self.respawns["learner"] += 1
        self._respawn_times.append(time.monotonic())

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
        self.stopped = True

    def _teardown(self) -> None:
        if self._teardown_done:
            return
        self._teardown_done = True
        logger.info("teardown: terminating processes")
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
    ap.add_argument("--worker-log-dir", default="",
                    help="If set, route each srun's --output/--error to a "
                         "per-launch file under this directory. Otherwise "
                         "workers inherit the coordinator's stdout/stderr.")
    ap.add_argument("--max-respawn-per-hour", type=int, default=20,
                    help="Circuit breaker: refuse further respawns once this "
                         "many have fired in a sliding 1-hour window. "
                         "0 disables the cap.")

    return ap


def main(argv: Optional[list[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    coord = Coordinator(args)
    return coord.start()


if __name__ == "__main__":
    sys.exit(main())
