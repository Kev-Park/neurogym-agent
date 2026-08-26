"""Failure-correlation analysis over instrumented eval runs (post-hoc).

Approved scope (2026-08-26): trajectory-shape taxonomy, failure margin,
spawn geometry, pair-level hardness across runs, and eval-neuron
characterization from the skeleton parquet. All computed from existing
eval JSONs (z_series/z_min/z_max per pair) — no new rollouts.

    uv run --no-sync python scripts/eval_failure_analysis.py \
        --primary v8@740=eval_results/v8_ckpt740.json \
        --also v7=eval_results/v7_ckpt370_protocol_v2.json \
        --also v8@370=eval_results/v8_ckpt370.json \
        --skeleton segment_positions.parquet

Taxonomy (failures of the PRIMARY run; precedence top-down):
  glitch       ended early without termination (env glitch / wedge)
  band-skip    a single step jumped clear across the success band
  near-stall   got within 2x band, then hovered outside it
  slow-approach still progressing toward the target at truncation
  oscillator   >=4 crossings of the (z_max - 2*band) level
  lost         never within 25% of extent of the target
  partial      everything else (made ground, stopped far out)

Neuron 'complexity': the skeleton parquet has points, NO edges, so true
branch counts are unavailable; proxies = node count, node density
(nodes per Mnm of path length), z-extent, xy spread, and node density
near the target (within 2x band of z_max).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import duckdb
import numpy as np


def band_of(r, abs_tol=10.0, frac=0.05):
    return max(abs_tol, frac * (r["z_max"] - r["z_min"]))


def classify(r, max_steps=300):
    zs = np.asarray(r["z_series"], float)
    zmax = r["z_max"]
    span = zmax - r["z_min"]
    band = band_of(r)
    d = np.abs(zs - zmax)
    closest = float(d.min())
    if r.get("wedged") or (r["steps"] < max_steps and not r["terminated"]):
        return "glitch", closest, band
    # band-skip: consecutive samples straddle the band, neither inside
    lo, hi = zmax - band, zmax + band
    skip = np.any((zs[:-1] < lo) & (zs[1:] > hi)) or np.any(
        (zs[:-1] > hi) & (zs[1:] < lo))
    if skip:
        return "band-skip", closest, band
    if closest > 0.25 * span:
        return "lost", closest, band
    tail = zs[-min(50, len(zs)):]
    tail_progress = (tail[-1] - tail[0]) * np.sign(zmax - tail[0])
    if closest <= 2 * band:
        if tail_progress < band / 2:
            return "near-stall", closest, band
        return "slow-approach", closest, band
    crossings = int(np.sum(np.diff(np.sign(zs - (zmax - 2 * band))) != 0))
    if crossings >= 4:
        return "oscillator", closest, band
    if tail_progress > band / 2:
        return "slow-approach", closest, band
    return "partial", closest, band


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, help="label=path of the run to dissect")
    ap.add_argument("--also", action="append", default=[],
                    help="label=path of additional runs (pair-level hardness)")
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--json", default="", help="also write findings as JSON")
    args = ap.parse_args()

    def load(spec):
        label, path = spec.split("=", 1)
        with open(path) as f:
            return label, {r["pair_idx"]: r for r in json.load(f)["per_pair"]}

    plabel, prim = load(args.primary)
    others = [load(s) for s in args.also]
    rows = list(prim.values())
    out = {"primary": plabel, "n": len(rows)}
    print(f"[failure-analysis] primary={plabel} n={len(rows)}; "
          f"also: {', '.join(l for l, _ in others) or 'none'}\n")

    # ---- taxonomy + margin ----
    fails, succ = [], []
    for r in rows:
        if r["terminated"]:
            succ.append(r)
        else:
            cls, closest, band = classify(r)
            fails.append({**r, "cls": cls, "closest": closest, "band": band,
                          "margin": closest / band})
    print(f"== Failure taxonomy ({len(fails)} failures / {len(rows)}) ==")
    by = Counter(f["cls"] for f in fails)
    tax = {}
    for cls, n in by.most_common():
        ms = [f["margin"] for f in fails if f["cls"] == cls]
        tax[cls] = {"n": n, "margin_median": round(float(np.median(ms)), 2)}
        print(f"  {cls:<14} n={n:<3} margin(closest/band): "
              f"median {np.median(ms):.2f}x  range {min(ms):.2f}-{max(ms):.2f}x")
    within2 = sum(1 for f in fails if f["margin"] <= 2.0)
    print(f"  -> {within2}/{len(fails)} failures ended within 2x band "
          f"(near-miss mass); converting them = "
          f"+{100 * within2 / len(rows):.1f} pts overall\n")
    out["taxonomy"] = tax

    # ---- spawn geometry ----
    def spawn_frac(r):
        span = r["z_max"] - r["z_min"] or 1e-6
        return (r["z_series"][0] - r["z_min"]) / span

    print("== Spawn geometry ==")
    sf_s = [spawn_frac(r) for r in succ]
    sf_f = [spawn_frac(r) for r in fails]
    print(f"  spawn height fraction: success median {np.median(sf_s):.2f} | "
          f"failure median {np.median(sf_f):.2f}")
    climb_s = [r["z_max"] - r["z_series"][0] for r in succ]
    climb_f = [r["z_max"] - r["z_series"][0] for r in fails]
    print(f"  climb distance (vox):  success median {np.median(climb_s):.0f} | "
          f"failure median {np.median(climb_f):.0f}")
    allr = [(spawn_frac(r), r["terminated"]) for r in rows]
    terc = np.quantile([a for a, _ in allr], [1 / 3, 2 / 3])
    for lab, lo, hi in [("low spawn", -1, terc[0]), ("mid", terc[0], terc[1]),
                        ("high spawn", terc[1], 2)]:
        sel = [t for a, t in allr if lo < a <= hi]
        print(f"  success rate, {lab:<11} tercile: "
              f"{100 * sum(sel) / len(sel):.0f}%  (n={len(sel)})")
    if any("projection_scale" in r for r in rows):
        ps_s = [r["projection_scale"] for r in succ if "projection_scale" in r]
        ps_f = [r["projection_scale"] for r in fails if "projection_scale" in r]
        print(f"  spawn zoom (projScale): success median {np.median(ps_s):.0f} | "
              f"failure median {np.median(ps_f):.0f}")
    else:
        print("  spawn zoom: not recorded in this run (B7 lands next eval)")
    print()

    # ---- pair-level hardness across runs ----
    print("== Pair-level hardness across runs ==")
    labels = [plabel] + [l for l, _ in others]
    outcomes = {}
    for pid in prim:
        o = [prim[pid]["terminated"]]
        for _, d in others:
            o.append(d[pid]["terminated"] if pid in d else None)
        outcomes[pid] = o
    counts = Counter(sum(1 for x in o if x) for o in outcomes.values())
    for k in sorted(counts, reverse=True):
        print(f"  succeeded in {k}/{len(labels)} runs: {counts[k]} pairs")
    always_fail = [pid for pid, o in outcomes.items()
                   if all(x is False for x in o)]
    print(f"  ALWAYS-FAIL pairs (0/{len(labels)}): {len(always_fail)} -> "
          f"{[prim[p]['root_id'] for p in always_fail[:8]]}"
          f"{'...' if len(always_fail) > 8 else ''}\n")
    out["always_fail_root_ids"] = [prim[p]["root_id"] for p in always_fail]

    # ---- neuron characterization (proxies; no edge data => no true branches) --
    print("== Eval-neuron characterization (skeleton proxies) ==")
    con = duckdb.connect()
    esc = args.skeleton.replace("'", "''")
    ids = sorted({r["root_id"] for r in rows})
    con.execute("CREATE TEMP TABLE pool(root_id VARCHAR)")
    con.executemany("INSERT INTO pool VALUES (?)", [[i] for i in ids])
    stats = con.execute(f"""
        SELECT CAST(s.root_id AS VARCHAR) rid, count(*) n_nodes,
               max(s.z)-min(s.z) ext_z,
               sqrt(pow(max(s.x)-min(s.x),2)+pow(max(s.y)-min(s.y),2)) xy_spread
        FROM read_parquet('{esc}') s JOIN pool p
          ON CAST(s.root_id AS VARCHAR)=p.root_id
        GROUP BY 1""").fetchall()
    morph = {r[0]: {"n_nodes": r[1], "ext_z": r[2], "xy_spread": r[3]}
             for r in stats}
    # node density near the target (within 2x band of z_max), per pair
    for r in rows:
        m = morph.get(r["root_id"])
        if m is None:
            continue
        band = band_of(r)
        near = con.execute(
            f"SELECT count(*) FROM read_parquet('{esc}') "
            "WHERE CAST(root_id AS VARCHAR)=? AND z >= ?",
            [r["root_id"], r["z_max"] - 2 * band]).fetchone()[0]
        r["_near_target_nodes"] = near
        r["_m"] = m

    def med(sel, key):
        vals = [key(r) for r in sel if "_m" in r]
        return np.median(vals) if vals else float("nan")

    # `fails` holds pre-annotation copies; use the annotated originals here.
    fail_rows = [r for r in rows if not r["terminated"]]
    print(f"  {'':<22}{'success':>10}{'failure':>10}")
    for name, key in [
        ("n_nodes", lambda r: r["_m"]["n_nodes"]),
        ("z-extent (vox)", lambda r: r["_m"]["ext_z"]),
        ("xy spread (vox)", lambda r: r["_m"]["xy_spread"]),
        ("length_nm (Mnm)", lambda r: r["length_nm"] / 1e6),
        ("node density /Mnm", lambda r: r["_m"]["n_nodes"] / (r["length_nm"] / 1e6)),
        ("nodes near target", lambda r: r["_near_target_nodes"]),
    ]:
        print(f"  {name:<22}{med(succ, key):>10.1f}{med(fail_rows, key):>10.1f}")
    af = [r for r in rows if r["root_id"] in set(out["always_fail_root_ids"])]
    if af:
        print(f"  always-fail neurons ({len(af)}): "
              f"n_nodes med {med(af, lambda r: r['_m']['n_nodes']):.0f}, "
              f"near-target nodes med "
              f"{med(af, lambda r: r['_near_target_nodes']):.0f}")
    print("\n  (branch counts need skeleton EDGES — parquet has points only; "
        "the above are proxies)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
