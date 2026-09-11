"""Gate 6: throughput + return curves of short training runs, pre vs post refactor.

Reads the `iter N: return_mean=... steps=... t=...s sps=...` lines train.py
prints and compares arms. Also accepts the ORIGINAL production log of the run
whose recipe the arms reproduce (native-v9-pace740, 740 iterations), so the
question "did the seam cost throughput" is answered against the run that
actually trained the checkpoint, not only against a 50-iteration replay.

    uv run --no-sync python scripts/gate6_compare.py \
        --arm pre=/scratch/kp0374/native_spike/slurm-gate6-883960.out \
        --arm post=/scratch/kp0374/native_spike/slurm-gate6-883964.out \
        --arm pace740=/scratch/kp0374/native_spike/slurm-nativetrain-873607.out \
        --warmup 5
"""

from __future__ import annotations

import argparse
import re

import numpy as np

_ITER = re.compile(r"^iter (\d+): return_mean=([-\d.e+]+) steps=(\d+) t=([\d.]+)s sps=([\d.]+)")


def parse(path):
    rows = []
    for line in open(path, errors="replace"):
        m = _ITER.match(line.strip())
        if m:
            rows.append((int(m[1]), float(m[2]), int(m[3]), float(m[4]), float(m[5])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="label=path/to/slurm.out (repeatable)")
    ap.add_argument("--warmup", type=int, default=5,
                    help="iterations to drop from the front (cold caches, first resets)")
    ap.add_argument("--first", type=int, default=0,
                    help="if >0, use only the first N iterations of each arm (so a 740-iter "
                         "production log is compared over the same window as a 50-iter arm)")
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        label, path = spec.split("=", 1)
        rows = parse(path)
        if args.first:
            rows = rows[:args.first]
        arms[label] = rows

    print(f"{'arm':10s} {'iters':>5s} {'sps med':>8s} {'sps mean':>9s} {'sps p10':>8s} "
          f"{'sps p90':>8s} {'t/iter':>7s} {'ret first5':>10s} {'ret last5':>9s}")
    stats = {}
    for label, rows in arms.items():
        if not rows:
            print(f"{label:10s} no iterations parsed")
            continue
        body = rows[args.warmup:] or rows
        sps = np.array([r[4] for r in body])
        t = np.array([r[3] for r in body])
        ret = np.array([r[1] for r in rows])
        stats[label] = sps
        print(f"{label:10s} {len(rows):5d} {np.median(sps):8.1f} {sps.mean():9.1f} "
              f"{np.percentile(sps, 10):8.1f} {np.percentile(sps, 90):8.1f} {np.median(t):7.1f} "
              f"{ret[:5].mean():10.3f} {ret[-5:].mean():9.3f}")

    labels = list(stats)
    if len(labels) >= 2:
        print()
        base = labels[0]
        for other in labels[1:]:
            a, b = stats[base], stats[other]
            ratio = np.median(b) / np.median(a)
            # Welch t on per-iteration sps; iterations are near-independent samples of
            # a stationary rate once warm.
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            z = (b.mean() - a.mean()) / se if se > 0 else float("nan")
            print(f"{other} / {base}: median sps ratio {ratio:.3f}  "
                  f"mean diff {b.mean() - a.mean():+.1f} sps  (z={z:+.2f}, n={len(a)}/{len(b)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
