"""Paired comparison of two eval_zmaxleft result sets.

zmax-left outcomes are continuous per-state z gains (dz@budget), not success
bits, so the comparison is: per-state paired differences d_i = dz_A(i) - dz_B(i)
on the SAME frozen starts, summarized by the median difference and tested with
a sign-flip permutation test on the mean of signed ranks (numpy-only, the same
machinery family as eval_compare.py's Tier 2). The unknown per-start achievable
maximum is common to both arms and cancels in d_i.

With --repeats>1 result files, each state's dz is first averaged over its
repeats (within-state mean), keeping one paired sample per state.

  uv run --no-sync python scripts/compare_zmaxleft.py A.json B.json \
      --labels ckptA ckptB [--budget 500]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np


def load(path: str, key: str) -> dict[int, float]:
    rows = json.load(open(path))["per_state"]
    by: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("wedged"):
            continue
        by[int(r["idx"])].append(float(r[key]))
    return {i: float(np.mean(v)) for i, v in by.items()}


def signflip_p(d: np.ndarray, n_iter: int = 100_000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation p for mean(signed ranks of d)."""
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    w_obs = float(np.sum(np.sign(d) * ranks))
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_iter, len(d)))
    w_null = flips @ ranks
    return float((np.abs(w_null) >= abs(w_obs)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--budget", type=int, default=500)
    args = ap.parse_args()

    key = f"dz@{args.budget}"
    A, B = load(args.a, key), load(args.b, key)
    common = sorted(set(A) & set(B))
    if not common:
        print("no common states", file=sys.stderr)
        return 1
    a = np.asarray([A[i] for i in common])
    b = np.asarray([B[i] for i in common])
    d = a - b
    la, lb = args.labels
    print(f"[compare-zl] {len(common)} paired states, outcome {key}")
    print(f"[compare-zl] {la}: median={np.median(a):.1f} mean={a.mean():.1f} | "
          f"{lb}: median={np.median(b):.1f} mean={b.mean():.1f}")
    print(f"[compare-zl] paired diff ({la}-{lb}): median={np.median(d):.1f} "
          f"mean={d.mean():.1f} | {la} better on {(d > 0).sum()}, "
          f"{lb} better on {(d < 0).sum()}, ties {(d == 0).sum()}")
    p = signflip_p(d)
    print(f"[compare-zl] sign-flip signed-rank p = {p:.4g} "
          f"({'significant' if p < 0.05 else 'not significant'} at 0.05)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
