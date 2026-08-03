"""Storm-precursor analysis — what triggers the FIRST reset/glitch before a storm.

Reads the instrumented-trial artifacts (per-env event JSONL + per-node 1Hz GPU/load
CSV + train.py MARK lines) and, for each detected glitch storm, reconstructs the
onset context WITHOUT assuming a cause:
  - onset glitch: phase (step/reset), browser age, settle polls, signature
  - resets in the pre-window: count / reasons / durations  (was a heavy reset the trigger?)
  - resource trend in the pre-window: GPU mem / util / load rise?  (contention spike?)
  - alignment to iteration / checkpoint boundaries  (external trigger?)
Then aggregates across storms so the dominant precursor is visible from data.

    python scripts/analyze_storm_precursors.py <event_dir> <slurm_out> [gap_s] [min_storm]
"""
from __future__ import annotations

import glob
import json
import re
import sys
from bisect import bisect_left


def load_events(evdir):
    ev = []
    for f in glob.glob(f"{evdir}/ev-*.jsonl"):
        for line in open(f, errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                ev.append(json.loads(line))
            except Exception:
                pass
    ev.sort(key=lambda e: e.get("ts", 0))
    return ev


def load_gpu(evdir):
    """host -> sorted list of (ts, gpu_mem, gpu_util, mem_util, load1)."""
    by_host = {}
    for f in glob.glob(f"{evdir}/gpu-*.csv"):
        rows = []
        for line in open(f, errors="ignore"):
            p = line.strip().split(",")
            if len(p) < 6 or p[0] == "ts":
                continue
            try:
                rows.append((float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
            except Exception:
                pass
        if rows:
            rows.sort()
            by_host[rows[0] and f.split("gpu-")[-1].replace(".csv", "")] = rows
    return by_host


def load_marks(out):
    iters, ckpts = [], []
    for line in open(out, errors="ignore"):
        m = re.search(r"MARK iter_start ts=([0-9.]+)", line)
        if m:
            iters.append(float(m.group(1)))
        m = re.search(r"MARK checkpoint it=\d+ ts=([0-9.]+)", line)
        if m:
            ckpts.append(float(m.group(1)))
    return sorted(iters), sorted(ckpts)


def gpu_window(rows, t0, t1):
    """Return (mem_start, mem_end, util_mean, load_mean) over [t0,t1]."""
    if not rows:
        return None
    ts = [r[0] for r in rows]
    a, b = bisect_left(ts, t0), bisect_left(ts, t1)
    seg = rows[a:b] or rows[max(0, a - 1):a + 1]
    if not seg:
        return None
    mem = [r[1] for r in seg]
    util = [r[2] for r in seg]
    load = [r[4] for r in seg]
    return (mem[0], mem[-1], sum(util) / len(util), sum(load) / len(load))


def nearest(sorted_ts, t):
    if not sorted_ts:
        return None
    i = bisect_left(sorted_ts, t)
    cands = [sorted_ts[j] for j in (i - 1, i) if 0 <= j < len(sorted_ts)]
    return min((t - c for c in cands), key=abs) if cands else None


def main():
    evdir = sys.argv[1]
    out = sys.argv[2]
    gap = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    min_storm = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    ev = load_events(evdir)
    gpu = load_gpu(evdir)
    iters, ckpts = load_marks(out)
    glitches = [e for e in ev if e.get("evt") == "glitch"]
    resets = [e for e in ev if e.get("evt") == "reset"]
    print(f"loaded {len(ev)} events ({len(glitches)} glitch, {len(resets)} reset) "
          f"from {len(glob.glob(evdir+'/ev-*.jsonl'))} runners; "
          f"{len(gpu)} node GPU traces; {len(iters)} iters, {len(ckpts)} ckpts\n")
    if not glitches:
        print("no glitch events — no storms to analyze")
        return

    # cluster glitches into storms by inter-arrival gap
    storms, cur = [], [glitches[0]]
    for g in glitches[1:]:
        if g["ts"] - cur[-1]["ts"] <= gap:
            cur.append(g)
        else:
            if len(cur) >= min_storm:
                storms.append(cur)
            cur = [g]
    if len(cur) >= min_storm:
        storms.append(cur)

    print(f"### {len(storms)} storms (>= {min_storm} glitches within {gap}s gaps) ###\n")
    reset_ts_by_pid = {}
    for r in resets:
        reset_ts_by_pid.setdefault(r["pid"], []).append(r)

    agg = {"onset_phase": {}, "pre_reset_reason": {}, "iter_align": 0, "ckpt_align": 0,
           "mem_rise": 0, "util_high": 0, "onset_age": []}
    PRE = 20.0
    for k, s in enumerate(storms):
        t0 = s[0]["ts"]
        onset = s[0]
        phase = onset.get("phase", "?")
        agg["onset_phase"][phase] = agg["onset_phase"].get(phase, 0) + 1
        agg["onset_age"].append(onset.get("episode", 0))
        # resets in pre-window (any pid)
        pre_resets = [r for r in resets if t0 - PRE <= r["ts"] < t0]
        heavy = [r for r in pre_resets if (r.get("total_ms") or 0) > 8000]
        for r in pre_resets:
            reason = ("success" if r.get("prev_terminated") else
                      "glitch" if r.get("prev_glitched") else
                      "timelimit" if (r.get("prev_steps") or 0) >= 250 else "early/first")
            agg["pre_reset_reason"][reason] = agg["pre_reset_reason"].get(reason, 0) + 1
        # resource trend
        host = onset.get("host")
        gw = gpu_window(gpu.get(host), t0 - PRE, t0)
        mem_note = ""
        if gw:
            mem0, mem1, util, load = gw
            if mem1 - mem0 > 300:
                agg["mem_rise"] += 1
            if util > 85:
                agg["util_high"] += 1
            mem_note = f"gpu_mem {mem0:.0f}->{mem1:.0f}MiB util~{util:.0f}% load~{load:.1f}"
        di = nearest(iters, t0)
        dc = nearest(ckpts, t0)
        if di is not None and abs(di) <= 5:
            agg["iter_align"] += 1
        if dc is not None and abs(dc) <= 5:
            agg["ckpt_align"] += 1
        print(f"storm {k+1}: {len(s)} glitches, onset host={host} pid={onset.get('pid')} "
              f"phase={phase} age={onset.get('episode')} settle_polls={onset.get('settle_polls')}")
        print(f"    pre-{PRE:.0f}s: resets={len(pre_resets)} (heavy>8s={len(heavy)}) | {mem_note} | "
              f"iter_boundary {di and round(di,1)}s  ckpt {dc and round(dc,1)}s")
        if s[0].get("signature"):
            print(f"    onset sig: {s[0]['signature'][:90]}")

    n = max(1, len(storms))
    print(f"\n### AGGREGATE over {len(storms)} storms ###")
    print(f"  onset phase: {agg['onset_phase']}")
    print(f"  onset browser-age (episodes): min={min(agg['onset_age'])} "
          f"med={sorted(agg['onset_age'])[len(agg['onset_age'])//2]} max={max(agg['onset_age'])}")
    print(f"  reset reasons in pre-window (all storms pooled): {agg['pre_reset_reason']}")
    print(f"  storms with GPU-mem rising >300MiB in pre-window: {agg['mem_rise']}/{len(storms)}")
    print(f"  storms with GPU-util >85% in pre-window:          {agg['util_high']}/{len(storms)}")
    print(f"  storms within 5s of an iteration boundary:        {agg['iter_align']}/{len(storms)}")
    print(f"  storms within 5s of a checkpoint write:           {agg['ckpt_align']}/{len(storms)}")


if __name__ == "__main__":
    main()
