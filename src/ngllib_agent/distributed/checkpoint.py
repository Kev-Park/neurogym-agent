"""Async checkpointing (agent_plan.md §14c).

A single Ray `CheckpointWriter` actor serializes writes (one outstanding at a
time); writes are atomic via tmp + `os.replace`, so an interrupted write can
never corrupt the previous good checkpoint. `AsyncCheckpointer` is the learner-
side helper: training continues while the actor writes; it blocks only if a
write is still pending when the next one is due (rare).
"""

from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Optional


def atomic_pickle(state: Any, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)  # atomic on same fs
    return str(path)


def atomic_json(obj: Any, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)
    return str(path)


_CKPT_RE = re.compile(r"ckpt_(\d+)\.pkl$")


def latest_checkpoint(ckpt_dir: str | Path) -> Optional[Path]:
    """Highest-iteration `ckpt_NNNNNN.pkl` in the dir, or None."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return None
    best, best_it = None, -1
    for p in ckpt_dir.glob("ckpt_*.pkl"):
        m = _CKPT_RE.search(p.name)
        if m and int(m.group(1)) > best_it:
            best, best_it = p, int(m.group(1))
    return best


class AsyncCheckpointer:
    """Every-K-iterations checkpointing through a CheckpointWriter actor.

    With `use_actor=False` writes happen synchronously in-process (tests /
    driver-only runs — same file format, no Ray dependency).
    """

    def __init__(self, ckpt_dir: str | Path, every: int, use_actor: bool = True):
        self.ckpt_dir = Path(ckpt_dir)
        self.every = int(every)
        self._pending = None
        self._writer = None
        if use_actor:
            import ray

            @ray.remote(num_cpus=0.1)
            class CheckpointWriter:
                def write(self, state, path):
                    return atomic_pickle(state, path)

            self._writer = CheckpointWriter.remote()

    def maybe_save(self, algo, iteration: int, meta: dict | None = None) -> Optional[str]:
        """Checkpoint if `iteration` is on the cadence. Returns the path if saved."""
        if self.every <= 0 or iteration % self.every != 0:
            return None
        path = self.ckpt_dir / f"ckpt_{iteration:06d}.pkl"
        state = algo.get_state()  # quick in-memory snapshot
        if meta is not None:
            atomic_json({**meta, "iteration": iteration}, self.ckpt_dir / "meta.json")
        if self._writer is None:
            return atomic_pickle(state, path)
        import ray

        if self._pending is not None:
            ray.get(self._pending)  # rare: previous write still in flight
        self._pending = self._writer.write.remote(state, str(path))
        return str(path)

    def finalize(self) -> None:
        """Block until any in-flight write lands."""
        if self._pending is not None:
            import ray

            ray.get(self._pending)
            self._pending = None


def load_checkpoint(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
