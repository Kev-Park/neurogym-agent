"""Pooled-stats analyzer for the conclusive SPS A/B sweep (round 2).

Reads slurm_outputs/abtest2-<arm>-s<seed>-*.out for arms base/q1b/q2b, pools
valid iters across seeds, and reports per-seed means (reproducibility) + pooled
distribution + pooled straggler fraction. Also dumps both probe variants.

    python scripts/abtest2_analyze.py
"""
from __future__ import annotations

import glob
import re
import statistics as st

ITER = re.compile(r"^iter \d+: .*steps=(\S+) t=([0-9.]+)s sps=(\S+)")
ARMS = {
    "base": "auto/600/escalate (baseline)",
    "q1b": "frag8/3840/timeout30 (Q1 sampler)",
    "q2b": "in_place (Q2 glitch-at-source)",
}


def parse(f):
    """Return list of (t, sps) for valid (non-None) iters in one run."""
    out = []
    for line in open(f, errors="ignore"):
        m = ITER.match(line)
        if not m:
            continue
        steps, t, sps = m.groups()
        if sps == "None" or steps == "None":
            continue
        tv, sv = float(t), float(sps)
        # Skip non-finite iters (a nan-loss iter can log sps=nan) so they don't
        # poison mean/stdev.
        if sv != sv or tv != tv:  # NaN check
            continue
        out.append((tv, sv))
    return out


def main():
    print("###### CONCLUSIVE SWEEP — training arms (150 iters x seeds) ######\n")
    for arm, desc in ARMS.items():
        files = sorted(glob.glob(f"slurm_outputs/abtest2-{arm}-s*.out"))
        if not files:
            print(f"----- {arm}: no logs -----\n")
            continue
        print(f"----- {arm} : {desc} -----")
        pooled_sps, pooled_t = [], []
        seed_means = []
        for f in files:
            data = parse(f)
            if not data:
                print(f"  {f.split('/')[-1]}: no valid iters")
                continue
            ts = [d[0] for d in data]
            sp = [d[1] for d in data]
            pooled_sps += sp
            pooled_t += ts
            seed_means.append(st.mean(sp))
            strag = sum(1 for x in ts if x > 120)
            print(f"  {f.split('/')[-1]}: n={len(sp)} "
                  f"sps_mean={st.mean(sp):.1f} sps_med={st.median(sp):.1f} "
                  f"straggler>120s={strag}({100*strag/len(ts):.0f}%)")
        if pooled_sps:
            strag = sum(1 for x in pooled_t if x > 120)
            across = (f"  ACROSS SEEDS: sps_mean={st.mean(seed_means):.1f}"
                      + (f" ± {st.pstdev(seed_means):.1f}" if len(seed_means) > 1 else "")
                      + f"  (n_seeds={len(seed_means)})")
            print(across)
            print(f"  POOLED (n={len(pooled_sps)}): sps_mean={st.mean(pooled_sps):.1f} "
                  f"sps_med={st.median(pooled_sps):.1f} "
                  f"sps_min={min(pooled_sps):.1f} sps_max={max(pooled_sps):.1f} | "
                  f"straggler>120s={strag}({100*strag/len(pooled_t):.0f}%)")
        print()

    print("###### PROBES ######")
    for label, pat in [("STEP-path", "slurm_outputs/abtest-probe-*.out"),
                       ("RESET-path (variant)", "slurm_outputs/abtest-probe-reset-*.out")]:
        print(f"--- {label} ---")
        for f in sorted(glob.glob(pat)):
            for line in open(f, errors="ignore"):
                if "RESULT" in line:
                    print("  " + line.strip())
        print()


if __name__ == "__main__":
    main()
