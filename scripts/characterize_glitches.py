"""Characterize glitch storms across the A/B sweep logs.

Mines slurm_outputs/abtest2-*.out for:
  - TRUE glitch counts (Ray dedups identical lines as "[repeated Nx across
    cluster]" -> each such line = 1+N events; naive grep -c undercounts).
  - type/signature breakdown (missing-fields vs state-read vs screenshot vs ...).
  - reset-path vs step-path split.
  - attribution: per-runner (pid) and per-node (ip) counts + concentration
    (top-runner share) -> is it a few sick browsers or uniform?
  - temporal clustering: inter-arrival coefficient of variation (Poisson~1,
    bursty>1), peak glitches in any 60s window -> are they STORMS or independent?

Run:  python scripts/characterize_glitches.py [glob]   (default abtest2-*.out)
"""
from __future__ import annotations

import glob
import re
import statistics as st
import sys
from collections import Counter
from datetime import datetime

GLITCH = re.compile(r"resilient: (reset|env) glitch (\w+) \((.*?)\);")
PID = re.compile(r"pid=(\d+)")
IP = re.compile(r"ip=([\d.]+)")
TS = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+)")
REPEAT = re.compile(r"repeated (\d+)x")
WATCHDOG = re.compile(r"watchdog killed hung Chrome")
RESTART = re.compile(r"periodic browser restart|reset attempt \d+ failed")


def signature(msg: str) -> str:
    if "missing fields" in msg:
        return "viewer-state missing-fields (render not settled)"
    if "could not read" in msg:
        return "viewer-state unreadable (evaluate->None)"
    if "captureScreenshot" in msg or "Unable to capture" in msg:
        return "screenshot capture failed"
    if "NoneType" in msg:
        return "NoneType obs (partial state)"
    if "hung" in msg or "watchdog" in msg:
        return "watchdog-killed hang"
    return msg[:48]


def epoch(line: str):
    m = TS.search(line)
    if not m:
        return None
    dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return dt.timestamp() + int(m.group(2)) / 1000.0


def parse(f):
    events = []  # (t, phase, sig, pid, ip, mult)
    watchdogs = restarts = 0
    for line in open(f, errors="ignore"):
        if WATCHDOG.search(line):
            watchdogs += 1
        if RESTART.search(line):
            restarts += 1
        g = GLITCH.search(line)
        if not g:
            continue
        phase, _typ, msg = g.groups()
        pid = PID.search(line)
        ip = IP.search(line)
        rep = REPEAT.search(line)
        mult = 1 + int(rep.group(1)) if rep else 1
        events.append((epoch(line), phase, signature(msg),
                       pid.group(1) if pid else "?",
                       ip.group(1) if ip else "?", mult))
    return events, watchdogs, restarts


def summarize(name, files):
    all_ev, wd, rs = [], 0, 0
    for f in files:
        ev, w, r = parse(f)
        all_ev += ev
        wd += w
        rs += r
    if not all_ev:
        print(f"  {name}: no glitches parsed\n")
        return
    total = sum(e[5] for e in all_ev)          # true count incl repeats
    lines = len(all_ev)                         # distinct log lines
    by_phase = Counter()
    by_sig = Counter()
    by_pid = Counter()
    by_ip = Counter()
    for t, phase, sig, pid, ip, mult in all_ev:
        by_phase[phase] += mult
        by_sig[sig] += mult
        by_pid[pid] += mult
        by_ip[ip] += mult
    print(f"  {name}: TRUE_glitches={total} (from {lines} log lines; "
          f"Ray-deduped {total-lines} extra)  watchdog_kills={wd}  browser_restarts={rs}")
    print(f"    phase: " + ", ".join(f"{k}={v}" for k, v in by_phase.most_common()))
    for sig, c in by_sig.most_common(5):
        print(f"    sig[{100*c/total:2.0f}%]: {sig}")
    # attribution / concentration
    top_pid, top_c = by_pid.most_common(1)[0]
    known_pids = {p: c for p, c in by_pid.items() if p != "?"}
    if known_pids:
        share = 100 * max(known_pids.values()) / sum(known_pids.values())
        print(f"    runners with glitches={len(known_pids)}  "
              f"top-runner share={share:.0f}% of attributed  "
              f"per-node: " + ", ".join(f"{ip.split('.')[-1]}={c}" for ip, c in by_ip.most_common() if ip != "?"))
    # temporal clustering (needs timestamps)
    ts = sorted(t for t, *_ in all_ev if t is not None)
    if len(ts) > 3:
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        cv = st.pstdev(gaps) / st.mean(gaps) if st.mean(gaps) > 0 else 0
        # peak glitches in any 60s sliding window
        peak = 0
        j = 0
        for i in range(len(ts)):
            while ts[i] - ts[j] > 60:
                j += 1
            peak = max(peak, i - j + 1)
        within10 = sum(1 for g in gaps if g <= 10)
        print(f"    temporal: inter-arrival CV={cv:.1f} (Poisson~1, storm>1)  "
              f"peak={peak} glitches/60s  {100*within10/len(gaps):.0f}% within 10s of prior")
    print()


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "abtest2-*.out"
    base = "slurm_outputs/"
    print(f"###### GLITCH CHARACTERIZATION ({pat}) ######\n")
    # per-arm aggregate
    for arm in ["base", "q1b", "q2b"]:
        summarize(f"ARM {arm}", sorted(glob.glob(f"{base}{pat.replace('*', arm+'-s*')}")))
    print("###### worst vs best baseline seed ######")
    for tag, gl in [("base-s1 (WORST 17% strag)", "abtest2-base-s1-*.out"),
                    ("base-s2 (BEST 4% strag)", "abtest2-base-s2-*.out")]:
        summarize(tag, sorted(glob.glob(base + gl)))


if __name__ == "__main__":
    main()
