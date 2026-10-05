"""Merge partial zmax-left eval outputs (eval_zmaxleft.py --offset shards).

A killed eval keeps its saved prefix (`partial: true`, flushed every 10
states); rerunning the rest with `--offset N` and merging gives the same
per-state records as one uninterrupted run. Rows are deduplicated by (idx,
rep), first file wins, and the summary is recomputed exactly as
eval_zmaxleft.py does.

    python scripts/merge_zmaxleft_evals.py out.json part1.json part2.json
"""

from __future__ import annotations

import json
import sys

import numpy as np


def main() -> int:
    out, *parts = sys.argv[1:]
    seen, rows = set(), []
    for p in parts:
        for r in json.load(open(p))["per_state"]:
            key = (int(r["idx"]), int(r.get("rep", 0)))
            if key not in seen:
                seen.add(key)
                rows.append(r)
    budgets = sorted({int(k.split("@")[1]) for k in rows[0] if k.startswith("dz@")})
    summary = {"n": len(rows), "budgets": budgets, "partial": False,
               "merged_from": parts}
    for b in budgets:
        dzs = np.asarray([r[f"dz@{b}"] for r in rows], dtype=float)
        summary[f"dz@{b}"] = {
            "median": round(float(np.median(dzs)), 1),
            "q25": round(float(np.percentile(dzs, 25)), 1),
            "q75": round(float(np.percentile(dzs, 75)), 1),
            "mean": round(float(dzs.mean()), 1),
            "frac_positive": round(float((dzs > 0).mean()), 3),
            "frac_beat_ceiling": round(float(np.mean([r[f"beat_ceiling@{b}"] for r in rows])), 3),
        }
    diags = [r["diag"] for r in rows if r.get("diag")]
    if diags:
        summary["diag"] = {
            "mean_new_segs": round(float(np.mean([d["new_segs"] for d in diags])), 2),
            "frac_hop_climb": round(float(np.mean([d["hop_climb"] for d in diags])), 3),
            "n_diag": len(diags),
        }
    with open(out, "w") as f:
        json.dump({"summary": summary, "per_state": rows}, f)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
