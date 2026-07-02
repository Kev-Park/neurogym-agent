"""IndexedStateProvider — FlywireSkeletonProvider filtered by cell_stats predicate.

Restricts the root_id sampling pool to segments whose per-neuron features
(length_nm, area_nm, size_nm) satisfy a caller-supplied SQL predicate.
Use for curriculum runs (train easy-first) and for eval-set filtering.

Both inputs can be `.parquet` or `.csv` — DuckDB reads either.

Example:
    provider = IndexedStateProvider(
        skeleton_source="/scratch/kp0374/neurogym-agent/segment_positions.parquet",
        cell_stats_source="/scratch/kp0374/neurogym-agent/cell_stats.parquet",
        predicate="length_nm > 100000",   # only ~long neurons
    )

The intersection (a) root_ids present in skeleton_source and (b) root_ids
matching predicate in cell_stats_source is computed once at construction;
`__call__` then delegates to a `FlywireSkeletonProvider` restricted to that
pool. See `agent_plan.md` §9 for the design rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np

from .flywire_skeleton import FlywireSkeletonProvider

if TYPE_CHECKING:
    from ngllib import NglState


def _read_table_expr(path: str) -> str:
    """Return a DuckDB expression to read from `path` (CSV or Parquet)."""
    esc = str(path).replace("'", "''")
    lower = str(path).lower()
    if lower.endswith(".parquet"):
        return f"read_parquet('{esc}')"
    if lower.endswith(".csv") or lower.endswith(".csv.gz"):
        return f"read_csv_auto('{esc}')"
    # Fall back to parquet — matches the plan's expected default and gives a
    # sensible error message if the extension is unrecognized.
    return f"read_parquet('{esc}')"


class IndexedStateProvider:
    """StateProvider sampling a predicate-filtered subset of a skeleton parquet.

    Structurally implements the `ngllib.StateProvider` Protocol via delegation.

    Args:
        skeleton_source: path to skeleton table (root_id, x, y, z).
        cell_stats_source: path to per-neuron feature table (root_id, length_nm, ...).
        predicate: SQL WHERE fragment applied against cell_stats. Empty or
            "1=1" means no filter. Referenced columns must exist in cell_stats.
        **provider_kwargs: forwarded verbatim to `FlywireSkeletonProvider`
            (projection_scale, cross_section_scale).
    """

    def __init__(
        self,
        skeleton_source: str,
        cell_stats_source: str,
        *,
        predicate: str = "1=1",
        **provider_kwargs: Any,
    ):
        self.skeleton_source = str(skeleton_source)
        self.cell_stats_source = str(cell_stats_source)
        self.predicate = predicate or "1=1"

        con = duckdb.connect()
        skel_expr = _read_table_expr(self.skeleton_source)
        stats_expr = _read_table_expr(self.cell_stats_source)

        # Intersect on root_id-as-string so mixed schemas (int in cell_stats,
        # str in skeletons — or vice versa) join correctly.
        rows = con.execute(
            f"""
            WITH skel AS (
                SELECT DISTINCT CAST(root_id AS VARCHAR) AS root_id
                FROM {skel_expr}
            ),
            stats AS (
                SELECT CAST(root_id AS VARCHAR) AS root_id
                FROM {stats_expr}
                WHERE {self.predicate}
            )
            SELECT skel.root_id
            FROM skel INNER JOIN stats USING (root_id)
            ORDER BY root_id
            """
        ).fetchall()
        con.close()

        allowed = [r[0] for r in rows]
        if not allowed:
            raise ValueError(
                f"empty pool: skeleton={self.skeleton_source} "
                f"cell_stats={self.cell_stats_source} predicate={self.predicate!r}"
            )
        self._inner = FlywireSkeletonProvider(
            self.skeleton_source, root_ids=allowed, **provider_kwargs,
        )

    # -- StateProvider Protocol ------------------------------------------------

    def __call__(
        self, rng: np.random.Generator, options: dict[str, Any] | None
    ) -> tuple["NglState", dict[str, Any]]:
        return self._inner(rng, options)

    def task_info_from_state(self, state: "NglState | str") -> dict[str, Any]:
        return self._inner.task_info_from_state(state)

    @property
    def root_ids(self) -> list[str]:
        return self._inner.root_ids
