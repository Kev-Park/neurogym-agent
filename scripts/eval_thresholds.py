"""Success rates under alternative z-tolerances, post-hoc from recorded z_series.

Reads an eval_d0 results JSON whose per_pair entries carry z_min/z_max/z_series
(recorded since 2026-08-23). For a candidate tolerance, an episode counts as a
success if its trajectory EVER enters the band |z - z_max| <= tol — exact for
"would the run under this tolerance have terminated", because the trajectory
prefix up to first entry is unaffected by the tolerance choice.

Reports the shipped absolute band (sanity: must reproduce the run's own
success count) plus percentage-of-z-extent bands, overall and per length
quartile, and the distribution of what the absolute band means in % terms.

    uv run --no-sync python scripts/eval_thresholds.py \
        --results eval_results/v7_ckpt370_stoch_v2.json \
        --abs-tol 10 --fracs 0.05,0.10,0.15
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def crossed(zs, z_max, tol) -> bool:
    return any(abs(z - z_max) <= tol for z in zs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--abs-tol", type=float, default=10.0)
    ap.add_argument("--fracs", default="0.05,0.10,0.15")
    args = ap.parse_args()

    with open(args.results) as f:
        data = json.load(f)
    rows = [r for r in data["per_pair"] if "z_series" in r]
    if not rows:
        print("FAIL: no z_series in per_pair — re-run eval_d0 with recording",
              file=sys.stderr)
        return 1
    if data.get("summary", {}).get("partial"):
        print(f"[thresholds] NOTE: partial results file ({len(rows)} pairs)")

    lengths = np.asarray([r["length_nm"] for r in rows])
    q1, q2, q3 = np.quantile(lengths, [0.25, 0.5, 0.75])
    buckets = [("q1", -np.inf, q1), ("q2", q1, q2), ("q3", q2, q3), ("q4", q3, np.inf)]
    extents = np.asarray([r["z_max"] - r["z_min"] for r in rows])
    fracs = [float(x) for x in args.fracs.split(",")]

    print(f"[thresholds] n={len(rows)}  z-extent: median {np.median(extents):.0f} vox "
          f"(q25 {np.quantile(extents, .25):.0f} / q75 {np.quantile(extents, .75):.0f}); "
          f"abs +/-{args.abs_tol:g} vox = {np.median(2 * args.abs_tol / extents * 100):.2f}% "
          f"of median extent (as +/-% of extent: median "
          f"{np.median(args.abs_tol / extents * 100):.2f}%)")

    crit = [("abs +/-%g vox" % args.abs_tol, lambda r: args.abs_tol)]
    crit += [("+/-%g%% extent" % (f * 100),
              lambda r, f=f: f * (r["z_max"] - r["z_min"])) for f in fracs]

    hdr = f'{"criterion":<18}{"overall":>9}' + "".join(f"{b[0]:>8}" for b in buckets)
    print(hdr)
    sanity = None
    for name, tol_fn in crit:
        wins = [crossed(r["z_series"], r["z_max"], tol_fn(r)) for r in rows]
        cells = []
        for _, lo, hi in buckets:
            sel = [w for w, r in zip(wins, rows) if lo <= r["length_nm"] < hi]
            cells.append(f"{100 * sum(sel) / len(sel):>7.1f}%" if sel else "     --")
        print(f"{name:<18}{100 * sum(wins) / len(wins):>8.1f}%" + "".join(cells))
        if sanity is None:
            sanity = sum(wins)
    reported = sum(1 for r in rows if r["terminated"])
    if sanity != reported:
        print(f"[thresholds] WARNING: abs-band crossings ({sanity}) != run's own "
              f"terminations ({reported}) — check abs-tol matches the run config")
    return 0


if __name__ == "__main__":
    sys.exit(main())
