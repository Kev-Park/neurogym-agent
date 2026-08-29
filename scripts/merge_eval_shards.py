"""Merge sharded eval_d0 outputs into one result set + report.

Sharded evals (native/r_eval_browser_shards.slurm) write one JSON per
shard; each shard's own summary is over its slice only. This recomputes
the real summary over the union — and reports MISSING pair_idx explicitly,
so a wedged/preempted shard shows up as a gap instead of silently biasing
the score.

    uv run --no-sync python scripts/merge_eval_shards.py \
        --prefix /scratch/.../eval_svc_browser_740 --expect 200
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True,
                    help="shard files are <prefix>_shard*.json")
    ap.add_argument("--expect", type=int, default=200)
    ap.add_argument("--config", default="configs/ppo_zmax_navigate.yaml")
    ap.add_argument("--report-budgets", default="300")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    import yaml

    from eval_thresholds import analyze
    from ngllib_agent.rewards import ZRewardConfig, effective_z_tolerance

    files = sorted(glob.glob(f"{args.prefix}_shard*.json"))
    if not files:
        print(f"no shards matched {args.prefix}_shard*.json")
        return 1
    results, seen = [], set()
    for f in files:
        d = json.load(open(f))
        n = 0
        for r in d.get("per_pair", []):
            if r["pair_idx"] in seen:
                continue
            seen.add(r["pair_idx"])
            results.append(r)
            n += 1
        print(f"[merge] {f}: {n} pairs "
              f"({'PARTIAL' if d.get('summary', {}).get('partial') else 'complete'})")
    results.sort(key=lambda r: r["pair_idx"])
    missing = sorted(set(range(args.expect)) - seen)
    print(f"[merge] {len(results)}/{args.expect} pairs; "
          f"missing pair_idx: {missing if missing else 'none'}")

    cfg = yaml.safe_load(open(args.config))
    rcfg = ZRewardConfig(
        z_tolerance=cfg["reward"]["z_tolerance"],
        z_tolerance_frac=cfg["reward"].get("z_tolerance_frac"))

    n = len(results)
    n_success = sum(1 for r in results if r["terminated"])
    lengths = np.asarray([r["length_nm"] for r in results])
    q1, q2, q3 = (float(x) for x in np.quantile(lengths, [0.25, 0.5, 0.75]))
    buckets = [("q1  (min - q25)", -np.inf, q1), ("q2  (q25 - q50)", q1, q2),
               ("q3  (q50 - q75)", q2, q3), ("q4  (q75 - max)", q3, np.inf)]
    per_quartile = []
    for label, lo, hi in buckets:
        b = [r for r in results if lo <= r["length_nm"] < hi]
        s = sum(1 for r in b if r["terminated"])
        per_quartile.append({"label": label, "n": len(b), "n_success": s,
                             "success_rate": s / len(b) if b else 0.0})

    print("\n============ merged eval summary ============")
    print(f"n_pairs             : {n}")
    print(f"overall success rate: {n_success / n:.2%}" if n else "no pairs")
    for q in per_quartile:
        print(f"  {q['label']:20s} n={q['n']:3d} success={q['n_success']:3d} "
              f"rate={q['success_rate']:.2%}")

    budgets = sorted({int(b) for b in args.report_budgets.split(",") if b}
                     | {args.max_steps})
    bq1, bq2, bq3 = np.quantile(lengths, [0.25, 0.5, 0.75])
    print("\n== Success by step budget (first band entry) ==")
    print(f'{"budget":<14}{"overall":>9}{"q1":>7}{"q2":>7}{"q3":>7}{"q4":>7}')
    for b in budgets:
        wins = []
        for r in results:
            tol = effective_z_tolerance(
                rcfg, {"z_max": r["z_max"], "z_min": r["z_min"]})
            fi = next((i for i, z in enumerate(r["z_series"])
                       if abs(z - r["z_max"]) <= tol), None)
            wins.append(fi is not None and fi <= b)
        cells = []
        for lo, hi in [(-1, bq1), (bq1, bq2), (bq2, bq3), (bq3, float("inf"))]:
            sel = [w for w, r in zip(wins, results) if lo < r["length_nm"] <= hi]
            cells.append(f"{100 * sum(sel) / len(sel):>6.0f}%" if sel else "    --")
        print(f"@{b:<13}{100 * sum(wins) / len(wins):>8.1f}%" + "".join(cells))

    text, table = analyze(results, abs_tol=float(cfg["reward"]["z_tolerance"]),
                          run_frac=cfg["reward"].get("z_tolerance_frac"))
    print("\n" + text)

    out = args.output or f"{args.prefix}_merged.json"
    with open(out, "w") as f:
        json.dump({"summary": {"n_pairs": n,
                               "overall_success_rate": n_success / n if n else 0.0,
                               "quartiles": per_quartile,
                               "missing_pair_idx": missing,
                               "shards": files},
                   "per_pair": results}, f, indent=2)
    with open(out + ".thresholds.json", "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
