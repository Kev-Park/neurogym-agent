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


def analyze(rows, abs_tol: float = 10.0, fracs=(0.05, 0.10, 0.15),
            run_frac: float | None = None):
    """Threshold table over per_pair rows carrying z_min/z_max/z_series.

    Returns (text, table_dict): printable report + the JSON shape
    eval_report_html --thresholds-json consumes. Overall rates are computed
    per-episode across ALL rows (sample-weighted); quartile buckets are
    equal-count by construction (quantiles of the rows' own lengths).

    run_frac: the run's OWN termination fraction (None = the abs band was the
    run criterion) — used only for the crossings-vs-terminations sanity check.
    Caveat for the non-criterion rows: episodes END at the run's band, so
    rows tighter than the criterion are lower bounds (the episode stops
    before it could reach the tighter band).
    """
    lengths = np.asarray([r["length_nm"] for r in rows])
    q1, q2, q3 = np.quantile(lengths, [0.25, 0.5, 0.75])
    buckets = [("q1", -np.inf, q1), ("q2", q1, q2), ("q3", q2, q3), ("q4", q3, np.inf)]
    extents = np.asarray([r["z_max"] - r["z_min"] for r in rows])

    lines = [
        f"[thresholds] n={len(rows)}  z-extent: median {np.median(extents):.0f} vox "
        f"(q25 {np.quantile(extents, .25):.0f} / q75 {np.quantile(extents, .75):.0f}); "
        f"abs +/-{abs_tol:g} vox as +/-% of extent: median "
        f"{np.median(abs_tol / extents * 100):.2f}%"
    ]
    crit = [("abs +/-%g vox" % abs_tol, lambda r: abs_tol)]
    crit += [("+/-%g%% extent" % (f * 100),
              lambda r, f=f: f * (r["z_max"] - r["z_min"])) for f in fracs]

    lines.append(f'{"criterion":<18}{"overall":>9}'
                 + "".join(f"{b[0]:>8}" for b in buckets))
    table = []
    for name, tol_fn in crit:
        wins = [crossed(r["z_series"], r["z_max"], tol_fn(r)) for r in rows]
        cells = []
        row = {"criterion": name,
               "overall": round(100 * sum(wins) / len(wins), 1)}
        for blab, lo, hi in buckets:
            sel = [w for w, r in zip(wins, rows) if lo <= r["length_nm"] < hi]
            row[blab] = round(100 * sum(sel) / len(sel), 1) if sel else None
            cells.append(f"{row[blab]:>7.1f}%" if sel else "     --")
        table.append(row)
        lines.append(f"{name:<18}{row['overall']:>8.1f}%" + "".join(cells))

    def _run_tol(r):
        if run_frac is None:
            return abs_tol
        return max(abs_tol, run_frac * (r["z_max"] - r["z_min"]))

    run_crossings = sum(crossed(r["z_series"], r["z_max"], _run_tol(r)) for r in rows)
    reported = sum(1 for r in rows if r["terminated"])
    if run_crossings != reported:
        lines.append(
            f"[thresholds] WARNING: run-band crossings ({run_crossings}) != run's "
            f"own terminations ({reported}) — check abs-tol/run-frac vs the run config")
    return "\n".join(lines), {
        "n": len(rows),
        "extent_median_vox": round(float(np.median(extents))),
        "rows": table,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--abs-tol", type=float, default=10.0)
    ap.add_argument("--fracs", default="0.05,0.10,0.15")
    ap.add_argument("--json", default="",
                    help="Also write the table as JSON (for eval_report_html).")
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

    text, table = analyze(rows, args.abs_tol,
                          [float(x) for x in args.fracs.split(",")])
    print(text)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(table, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
