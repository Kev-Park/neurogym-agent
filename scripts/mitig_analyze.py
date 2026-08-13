"""Analyzer for the mitigation sweep (M1a stagger + M2 tuned sampler + M4).

Per arm (base / m1 / m1m2), pooled across seeds:
  - SPS distribution + straggler-iter fraction (the headline: does the worst
    seed rise toward the peak?)
  - WAVE CHECK (M1a's direct mechanism): max resets per 10s window per run —
    stagger should collapse this from ~30-50 to single digits.
  - F4 attribution (M3): slow-step events + per-episode step-time stats from
    the reset events, so "stall elsewhere" stragglers become explainable.

    python scripts/mitig_analyze.py
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
from collections import defaultdict

ITER = re.compile(r"^iter \d+: .*steps=(\S+) t=([0-9.]+)s sps=(\S+)")
ARMS = {
    "base": "auto/4000/600, no stagger (M4 node pool)",
    "m1": "+ first-episode stagger (M1a)",
    "m1m2": "stagger + frag8/5376/timeout60 (M1a+M2)",
    "m1m2m5": "m1m2 + warm-context reset-ahead (M5)",
}


def parse_iters(f):
    out = []
    for line in open(f, errors="ignore"):
        m = ITER.match(line)
        if not m:
            continue
        steps, t, sps = m.groups()
        if "None" in (steps, sps):
            continue
        tv, sv = float(t), float(sps)
        if sv != sv or tv != tv:
            continue
        out.append((tv, sv))
    return out


def wave_stats(evdir):
    """(max resets/10s window, n_resets, n_slow_step_events, slow_steps_total)."""
    resets, slow_evt, slow_total = [], 0, 0
    for f in glob.glob(f"{evdir}/ev-*.jsonl"):
        for line in open(f, errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("evt") == "reset":
                resets.append(e["ts"])
                slow_total += e.get("prev_slow_steps") or 0
            elif e.get("evt") == "slow_step":
                slow_evt += 1
    if not resets:
        return (0, 0, slow_evt, slow_total)
    t0 = min(resets)
    wins = defaultdict(int)
    for ts in resets:
        wins[int((ts - t0) // 10)] += 1
    return (max(wins.values()), len(resets), slow_evt, slow_total)


def main():
    print("###### MITIGATION SWEEP — 150 iters x seeds per arm ######\n")
    for arm, desc in ARMS.items():
        files = sorted(glob.glob(f"slurm_outputs/mitig-mitig-*.out"))
        # run-name is embedded in the log content; match by ARM line instead
        files = [f for f in files
                 if any(f"ARM: mitig-{arm}-s" in line
                        for line in open(f, errors="ignore").readlines()[:5])]
        if not files:
            print(f"----- {arm}: no logs -----\n")
            continue
        print(f"----- {arm} : {desc} -----")
        pooled_sps, pooled_t, seed_means = [], [], []
        for f in files:
            data = parse_iters(f)
            run = re.search(r"mitig-{a}-s(\d+)".format(a=arm),
                            open(f, errors="ignore").read(2000))
            seed = run.group(1) if run else "?"
            if not data:
                print(f"  s{seed}: no valid iters ({f.split('/')[-1]})")
                continue
            ts = [d[0] for d in data]
            sp = [d[1] for d in data]
            pooled_sps += sp
            pooled_t += ts
            seed_means.append(st.mean(sp))
            strag = sum(1 for x in ts if x > 120)
            mx, nres, sevt, stot = wave_stats(f"event_logs/mitig-{arm}-s{seed}")
            print(f"  s{seed}: n={len(sp)} sps_mean={st.mean(sp):.1f} "
                  f"med={st.median(sp):.1f} straggler>120s={strag}({100*strag/len(ts):.0f}%) "
                  f"| max-wave={mx} resets/10s (n={nres}) "
                  f"| slow_step_evts={sevt} slow_steps_total={stot}")
        if pooled_sps:
            strag = sum(1 for x in pooled_t if x > 120)
            spread = f" ± {st.pstdev(seed_means):.1f}" if len(seed_means) > 1 else ""
            print(f"  ACROSS SEEDS: sps_mean={st.mean(seed_means):.1f}{spread} "
                  f"worst-seed={min(seed_means):.1f} best-seed={max(seed_means):.1f}")
            print(f"  POOLED (n={len(pooled_sps)}): med={st.median(pooled_sps):.1f} "
                  f"min={min(pooled_sps):.1f} | straggler={100*strag/len(pooled_t):.0f}%")
        print()


if __name__ == "__main__":
    main()
