"""Generate a frozen eval d0 — N seeded (root_id, node_index) pairs.

Sampling flow:
  1. Intersect skeleton_source.root_id with cell_stats_source.root_id filtered
     by the predicate. Get (root_id, n_nodes, length_nm) per eligible segment.
  2. Uniformly sample N (root_id, node_index) pairs using a seeded numpy RNG:
     - pick a root_id from the pool
     - pick a node_index in [0, n_nodes)
  3. Write to parquet: (pair_idx, root_id, node_index, length_nm).

Uniform-over-root-ids matches the "neuron is the unit of generalization" default
from `agent_plan.md`. The `length_nm` column is preserved so the eval CLI can
report success rate per length quartile.

Reproducibility: same seed + same input files + same predicate + same n_pairs
=> byte-identical parquet.
"""

from __future__ import annotations

import argparse
import sys

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _read_expr(path: str) -> str:
    esc = str(path).replace("'", "''")
    lower = str(path).lower()
    if lower.endswith(".csv") or lower.endswith(".csv.gz"):
        return f"read_csv_auto('{esc}')"
    return f"read_parquet('{esc}')"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate frozen eval d0 pairs.")
    ap.add_argument("--skeleton", required=True,
                    help="Skeleton parquet/csv (root_id, x, y, z).")
    ap.add_argument("--cell-stats", required=True,
                    help="Cell-stats parquet/csv (root_id, length_nm, ...).")
    ap.add_argument("--predicate", default="1=1",
                    help='SQL WHERE fragment applied to cell_stats. '
                         'Default: "1=1" (all rows).')
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True,
                    help="Output parquet path.")
    args = ap.parse_args()

    con = duckdb.connect()
    skel_expr = _read_expr(args.skeleton)
    stats_expr = _read_expr(args.cell_stats)

    print(f"[eval_d0] intersecting {args.skeleton} ∩ {args.cell_stats} "
          f"WHERE {args.predicate}", flush=True)
    rows = con.execute(
        f"""
        WITH skel_counts AS (
            SELECT CAST(root_id AS VARCHAR) AS root_id,
                   count(*) AS n_nodes
            FROM {skel_expr}
            GROUP BY root_id
        ),
        stats AS (
            SELECT CAST(root_id AS VARCHAR) AS root_id,
                   length_nm
            FROM {stats_expr}
            WHERE {args.predicate}
        )
        SELECT s.root_id, s.n_nodes, st.length_nm
        FROM skel_counts s INNER JOIN stats st USING (root_id)
        ORDER BY s.root_id
        """
    ).fetchall()
    con.close()

    if not rows:
        print(f"FAIL: empty eligible pool for predicate {args.predicate!r}",
              file=sys.stderr, flush=True)
        return 1

    print(f"[eval_d0] eligible pool: {len(rows)} root_ids", flush=True)

    rng = np.random.default_rng(args.seed)
    # Sample N (root_id, node_idx) pairs. Uniform over root_ids first, then
    # uniform node within that root_id.
    idxs = rng.integers(0, len(rows), size=args.n_pairs)
    pair_records: list[tuple[int, str, int, int]] = []
    for i, row_i in enumerate(idxs):
        root_id, n_nodes, length_nm = rows[int(row_i)]
        node_index = int(rng.integers(int(n_nodes)))
        pair_records.append((i, str(root_id), node_index, int(length_nm)))

    table = pa.table({
        "pair_idx": pa.array([p[0] for p in pair_records], type=pa.int32()),
        "root_id": pa.array([p[1] for p in pair_records], type=pa.string()),
        "node_index": pa.array([p[2] for p in pair_records], type=pa.int32()),
        "length_nm": pa.array([p[3] for p in pair_records], type=pa.int64()),
    })
    pq.write_table(table, args.output)

    # Distribution summary (per-quartile of length_nm) — the eval CLI groups
    # success rate along these same quartile buckets.
    lengths = np.asarray([p[3] for p in pair_records])
    q = np.quantile(lengths, [0.25, 0.5, 0.75])
    print(f"[eval_d0] wrote {args.n_pairs} pairs -> {args.output}", flush=True)
    print(f"[eval_d0] length_nm: min={lengths.min()}, "
          f"q25={q[0]:.0f}, q50={q[1]:.0f}, q75={q[2]:.0f}, "
          f"max={lengths.max()}", flush=True)
    unique_ids = len(set(p[1] for p in pair_records))
    print(f"[eval_d0] unique root_ids among {args.n_pairs} pairs: {unique_ids}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
