"""FlywireSkeletonProvider — ngllib 0.2 StateProvider over a skeleton parquet.

Samples a start state from a parquet of L2 skeleton nodes with schema
`(root_id: str, x: float, y: float, z: float)`. Each episode: pick a `root_id`,
start at a random node of that segment with a random orientation, and target the
segment's max-z point (`task_info["z_max"]`). Ported from the legacy
`neurogym-agent/envs/ngl_gym_env.py` sampling, backed by DuckDB over the parquet
(shared via the OS page cache — see `agent_plan.md` §9).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np

if TYPE_CHECKING:
    from ngllib import NglState


def _random_quaternion(rng: np.random.Generator) -> list[float]:
    """Uniform random unit quaternion (Shoemake). Uses `rng` for reproducibility."""
    u1, u2, u3 = (float(x) for x in rng.random(3))
    return [
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    ]


class FlywireSkeletonProvider:
    """StateProvider sampling FlyWire skeleton start states from a parquet.

    Implements the `ngllib.StateProvider` Protocol structurally (no import).

    Note: `WHERE root_id = ?` on the raw parquet is a full column scan per
    episode — fine for the single-process smoke; a partitioned/indexed layout is
    a later optimization (agent_plan §9).
    """

    def __init__(
        self,
        parquet_path: str,
        *,
        projection_scale: float = 14000.0,
        cross_section_scale: float = 2.0,
        root_ids: list[str] | None = None,
        exclude_root_ids: list[str] | None = None,
    ):
        self.parquet_path = str(parquet_path)
        self._con = duckdb.connect()
        # CREATE VIEW can't bind a prepared parameter, so inline the path
        # (single-quote-escaped). Per-episode SELECTs below still use `?` binds.
        _escaped = self.parquet_path.replace("'", "''")
        self._con.execute(
            f"CREATE VIEW skeletons AS SELECT * FROM read_parquet('{_escaped}')"
        )
        self._projection_scale = float(projection_scale)
        self._cross_section_scale = float(cross_section_scale)

        if root_ids is not None:
            self._root_ids = [str(r) for r in root_ids]
        else:
            rows = self._con.execute(
                "SELECT DISTINCT root_id FROM skeletons"
            ).fetchall()
            self._root_ids = [str(r[0]) for r in rows]
        # Eval-holdout separation: drop the frozen eval pool's segments from
        # the training distribution (explicit `segment_id` reset options — the
        # eval CLI's path — bypass this list on purpose).
        if exclude_root_ids:
            excl = {str(r) for r in exclude_root_ids}
            self._root_ids = [r for r in self._root_ids if r not in excl]
        if not self._root_ids:
            raise ValueError(f"no root_ids found in {self.parquet_path}")

    # -- StateProvider Protocol ------------------------------------------------

    def __call__(
        self, rng: np.random.Generator, options: dict[str, Any] | None
    ) -> tuple["NglState", dict[str, Any]]:
        options = options or {}
        root_id = str(options.get("segment_id") or rng.choice(self._root_ids))
        nodes = self._nodes(root_id)
        start = nodes[int(rng.integers(len(nodes)))]

        state: "NglState" = {
            "position": [float(start[0]), float(start[1]), float(start[2])],
            "projectionOrientation": _random_quaternion(rng),
            "projectionScale": self._projection_scale,
            "crossSectionScale": self._cross_section_scale,
            "segments": [root_id],
        }
        task_info = {"segment_id": root_id, "z_max": float(nodes[:, 2].max())}
        return state, task_info

    def task_info_from_state(self, state: "NglState | str") -> dict[str, Any]:
        if isinstance(state, str):
            raise NotImplementedError(
                "cannot derive task_info from a raw URL; supply an NglState with `segments`"
            )
        segments = state.get("segments") or []
        if not segments:
            raise NotImplementedError("state has no `segments` to derive task_info from")
        root_id = str(segments[0])
        z_max = self._con.execute(
            "SELECT max(z) FROM skeletons WHERE root_id = ?", [root_id]
        ).fetchone()[0]
        if z_max is None:
            raise ValueError(f"root_id {root_id!r} not found in {self.parquet_path}")
        return {"segment_id": root_id, "z_max": float(z_max)}

    # -- internals -------------------------------------------------------------

    def _nodes(self, root_id: str) -> np.ndarray:
        res = self._con.execute(
            "SELECT x, y, z FROM skeletons WHERE root_id = ?", [root_id]
        ).fetchnumpy()
        if len(res["x"]) == 0:
            raise ValueError(f"root_id {root_id!r} not found in {self.parquet_path}")
        return np.stack([res["x"], res["y"], res["z"]], axis=1).astype(np.float64)

    @property
    def root_ids(self) -> list[str]:
        return self._root_ids
