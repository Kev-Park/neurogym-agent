"""Paired comparison of two eval_d0 result sets, at the tier the data supports.

Tier 1 -- repeats=1 (the default eval): each pair carries a single Bernoulli
outcome, so the only honest test is McNemar's exact test on the discordant
pairs. Cheap screen; run it on every eval.

Tier 2 -- repeats>1 (eval_d0 --repeats K): each pair carries a success RATE
out of K rollouts from the SAME initial state. Rates are paired continuous
samples, so a signed-rank / permutation test uses within-pair information
that McNemar discards, and reaches the same power with far fewer episodes.

Tier 2 also measures what Tier 1 cannot: the ROLLOUT NOISE FLOOR. A stochastic
policy disagrees with itself across repeats, and that disagreement inflates
McNemar's discordant count -- so a Tier-1 p near 1 may mean "no effect" or
merely "the measurement was mostly coin-flips". The floor separates them.

Both tiers compare THESE TWO CHECKPOINTS on this holdout. Neither licenses a
claim about the training condition that produced them: with one seed per arm
the training run is itself an unreplicated draw. Only seeds fix that.

    uv run --no-sync python scripts/eval_compare.py A.json B.json --labels a b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict

import numpy as np


def load(path):
    """pair_idx -> list of per-repeat successes, ordered by rep index."""
    rows = json.load(open(path))["per_pair"]
    by = defaultdict(dict)
    for r in rows:
        by[int(r["pair_idx"])][int(r.get("rep", 0))] = bool(r["terminated"])
    return {p: [v[k] for k in sorted(v)] for p, v in by.items()}


def mcnemar(a_bin, b_bin):
    b01 = sum(1 for x, y in zip(a_bin, b_bin) if not x and y)
    b10 = sum(1 for x, y in zip(a_bin, b_bin) if x and not y)
    n = b01 + b10
    if n == 0:
        return 0, 0, 1.0
    tail = sum(math.comb(n, i) for i in range(min(b01, b10) + 1))
    return b10, b01, min(1.0, 2.0 * tail / 2.0 ** n)


def signflip_p(w, rng, n_iter):
    """Two-sided sign-flip test on a paired statistic written as sum(w).

    Under the null of within-pair exchangeability, flipping the sign of any
    pair's contribution is a valid relabelling. Assumption-free -- no
    normality, no minimum n -- which matters because per-pair rates out of
    K=5 take six distinct values and are heavily tied.

    Both statistics we use are linear in the flipped signs (the mean of the
    differences, and the signed-rank sum -- ranks depend only on |d|, which
    sign flips leave unchanged), so the null is one matrix product.
    """
    w = np.asarray(w, dtype=float)
    n = w.size
    if n == 0:
        return 1.0
    obs = float(w.sum())
    if n <= 18:  # exact enumeration
        signs = (((np.arange(2 ** n, dtype=np.int64)[:, None]
                   >> np.arange(n)) & 1) * 2 - 1).astype(np.int8)
    else:
        signs = (rng.integers(0, 2, size=(n_iter, n)) * 2 - 1).astype(np.int8)
    null = signs @ w
    return float(min(1.0, (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1)
                     / (null.size + 1)))


def signed_ranks(d):
    """Signed ranks of |d| with mean-rank tie handling (Wilcoxon weights)."""
    a = np.abs(d)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(d.size, dtype=float)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return np.sign(d) * ranks


def noise_floor(runs, k):
    """Unbiased within-pair Bernoulli variance E[p(1-p)], and the Tier-1
    self-discordance 2*E[p(1-p)] it implies -- i.e. how many pairs this
    policy would flip against a rerun of ITSELF."""
    xs = np.array([sum(v) for v in runs], dtype=float)
    var = float(np.mean(xs * (k - xs) / (k * (k - 1))))
    return var, 2.0 * var


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--labels", nargs=2, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=50000)
    args = ap.parse_args()

    la, lb = args.labels or (args.a.split("/")[-1], args.b.split("/")[-1])
    A, B = load(args.a), load(args.b)
    common = sorted(set(A) & set(B))
    if not common:
        print("no shared pair_idx between the two files")
        return 1
    ka = min(len(A[p]) for p in common)
    kb = min(len(B[p]) for p in common)
    k = min(ka, kb)
    ra = np.array([sum(A[p][:k]) / k for p in common])
    rb = np.array([sum(B[p][:k]) / k for p in common])
    rng = np.random.default_rng(args.seed)

    print("pairs compared      : %d   (repeats: %s=%d, %s=%d -> comparing at k=%d)"
          % (len(common), la, ka, lb, kb, k))
    print("%-20s: %6.2f%%   (%d/%d episodes)"
          % (la, 100 * ra.mean(), round(ra.sum() * k), len(common) * k))
    print("%-20s: %6.2f%%   (%d/%d episodes)"
          % (lb, 100 * rb.mean(), round(rb.sum() * k), len(common) * k))
    print("difference          : %+.2f pp   (%s minus %s)"
          % (100 * (ra.mean() - rb.mean()), la, lb))

    b10, b01, p_mc = mcnemar([A[p][0] for p in common], [B[p][0] for p in common])
    print()
    print("-- Tier 1: McNemar exact (first rollout of each pair) --")
    print("discordant=%d   (%s-only %d / %s-only %d)   p=%.4f   %s"
          % (b10 + b01, la, b10, lb, b01, p_mc,
             "SIGNIFICANT" if p_mc < 0.05 else "not significant"))

    if k < 2:
        print()
        print("Tier 2 needs repeats>1 in BOTH files (rerun with --repeats 5).")
        return 0

    d = ra - rb
    nz = d[d != 0.0]
    print()
    print("-- Tier 2: paired tests on per-pair success rates (k=%d) --" % k)
    print("pairs that differ   : %d of %d" % (nz.size, len(common)))
    p_perm = signflip_p(nz, rng, args.iters)
    p_rank = signflip_p(signed_ranks(nz), rng, args.iters)
    print("permutation (mean)  : p=%.4f   %s"
          % (p_perm, "SIGNIFICANT" if p_perm < 0.05 else "not significant"))
    print("Wilcoxon signed-rank: p=%.4f   %s"
          % (p_rank, "SIGNIFICANT" if p_rank < 0.05 else "not significant"))
    try:
        from scipy import stats
        t = stats.ttest_rel(ra, rb)
        print("paired t-test       : t=%+.3f p=%.4f" % (t.statistic, t.pvalue))
    except Exception:
        print("paired t-test       : scipy unavailable -- the permutation test "
              "above needs no normality assumption and supersedes it")

    boot = np.array([np.mean(rng.choice(d, d.size, replace=True))
                     for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("95%% bootstrap CI     : [%+.2f, %+.2f] pp" % (100 * lo, 100 * hi))

    print()
    print("-- rollout noise floor (how far a policy disagrees with itself) --")
    for lab, src in ((la, A), (lb, B)):
        var, disc = noise_floor([src[p][:k] for p in common], k)
        print("%-20s: within-pair var=%.4f -> %.1f%% self-discordance "
              "(~%.0f of %d pairs would flip against a rerun of itself)"
              % (lab, var, 100 * disc, disc * len(common), len(common)))

    sd = float(np.std(d, ddof=1))
    if sd > 0:
        print()
        print("-- resolution --")
        print("sd of per-pair diff : %.4f" % sd)
        print("MDE at n=%-4d       : %.2f pp (80%% power, alpha=0.05)"
              % (len(common), 100 * 2.8 * sd / math.sqrt(len(common))))
        if abs(d.mean()) > 1e-9:
            print("pairs needed to resolve the observed %+.2f pp: %d"
                  % (100 * d.mean(), math.ceil((2.8 * sd / abs(d.mean())) ** 2)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
