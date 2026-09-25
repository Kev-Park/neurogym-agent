"""Generate the frozen zmax-left eval holdout — N complete random START STATES.

Unlike eval_d0 (target-based (root_id, node_index) pairs), zmax-left has no
per-start target, so the holdout freezes the full start state: position,
orientation quaternion, projectionScale, crossSectionScale, and the start
segment. Evaluation is paired per-start delta-z across policies (signed-rank),
which cancels the unknown per-start achievable maximum.

Schema: (idx, root_id, x, y, z, qx, qy, qz, qw, projection_scale,
         cross_section_scale, seg_z_max, seg_z_min)
seg_z_max/min are the START segment's skeleton extent — descriptive columns for
stratified reporting ("did it beat its own neuron's ceiling" = a hop happened),
never used by training. `root_id` doubles as the training-exclusion column
(env_build reads it from env.holdout_parquet).

Reproducibility: same seed + same skeleton parquet => byte-identical output.
Sampling mirrors FlywireSkeletonProvider exactly (uniform root, uniform node,
Shoemake quaternion, log-uniform projectionScale in [3500, 14000]).

  uv run --no-sync python scripts/generate_zmaxleft_holdout.py \
      --skeleton /scratch/kp0374/neurogym-agent/segment_positions.parquet \
      --output   /scratch/kp0374/neurogym-agent/eval_zmaxleft_v1.parquet
"""

from __future__ import annotations

import argparse
import math
import sys

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _random_quaternion(rng: np.random.Generator) -> list[float]:
    """Uniform random unit quaternion (Shoemake) — mirrors the provider."""
    u1, u2, u3 = (float(x) for x in rng.random(3))
    return [
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate frozen zmax-left start states.")
    ap.add_argument("--skeleton", required=True,
                    help="Skeleton parquet (root_id, x, y, z).")
    ap.add_argument("--n-states", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ps-range", type=float, nargs=2, default=(3500.0, 14000.0))
    ap.add_argument("--cross-section-scale", type=float, default=2.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    con = duckdb.connect()
    esc = str(args.skeleton).replace("'", "''")
    con.execute(f"CREATE VIEW skel AS SELECT * FROM read_parquet('{esc}')")
    roots = [str(r[0]) for r in
             con.execute("SELECT DISTINCT root_id FROM skel ORDER BY root_id").fetchall()]
    print(f"[zmaxleft-holdout] {len(roots)} roots in {args.skeleton}", flush=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for idx in range(args.n_states):
        root = roots[int(rng.integers(len(roots)))]
        res = con.execute(
            "SELECT x, y, z FROM skel WHERE root_id = ? ORDER BY x, y, z", [root]
        ).fetchnumpy()
        nodes = np.stack([res["x"], res["y"], res["z"]], axis=1).astype(np.float64)
        start = nodes[int(rng.integers(len(nodes)))]
        q = _random_quaternion(rng)
        lo, hi = args.ps_range
        ps = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        rows.append({
            "idx": idx, "root_id": root,
            "x": float(start[0]), "y": float(start[1]), "z": float(start[2]),
            "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3],
            "projection_scale": ps,
            "cross_section_scale": float(args.cross_section_scale),
            "seg_z_max": float(nodes[:, 2].max()),
            "seg_z_min": float(nodes[:, 2].min()),
        })

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, args.output)
    zs = np.array([r["z"] for r in rows])
    head = np.array([r["seg_z_max"] - r["z"] for r in rows])
    print(f"[zmaxleft-holdout] wrote {len(rows)} states -> {args.output}", flush=True)
    print(f"[zmaxleft-holdout] start-z quartiles: "
          f"{np.percentile(zs, [25, 50, 75]).round(1).tolist()} | "
          f"own-ceiling headroom quartiles: "
          f"{np.percentile(head, [25, 50, 75]).round(1).tolist()}", flush=True)
    print(f"[zmaxleft-holdout] distinct roots: {len({r['root_id'] for r in rows})}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
