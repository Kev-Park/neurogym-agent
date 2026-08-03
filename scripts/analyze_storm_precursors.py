"""Storm/straggler precursor analysis — what makes an iteration go slow.

Anchors on the EXPENSIVE ITERATIONS (t>120s) rather than glitch-count clusters,
because stragglers can come from individual heavy resets, not only clustered
storms. For straggler vs normal iterations it contrasts, purely from data:
  - resets/iter, retried resets (nav_attempts>1 = a glitched reset), heavy
    resets (total_ms>8s), navigate_ms, browser age
  - glitches/iter and their phase (step vs reset)
  - GPU mem/util/load trend during the iteration
  - alignment of the iteration's expensive events to iter/checkpoint boundaries
Then per straggler it dumps the earliest expensive event + what preceded it.

    python scripts/analyze_storm_precursors.py <event_dir> <slurm_out> [straggler_s]
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
            if line:
                try:
                    ev.append(json.loads(line))
                except Exception:
                    pass
    ev.sort(key=lambda e: e.get("ts", 0))
    return ev


def load_gpu(evdir):
    by_host = {}
    for f in glob.glob(f"{evdir}/gpu-*.csv"):
        host = f.split("gpu-")[-1].replace(".csv", "")
        rows = []
        for line in open(f, errors="ignore"):
            p = line.strip().split(",")
            if len(p) < 6 or p[0] == "ts":
                continue
            try:
                mem = float(p[1]); util = float(p[2]) if p[2] not in ("[N/A]", "") else -1
                rows.append((float(p[0]), mem, util, float(p[4])))
            except Exception:
                pass
        if rows:
            rows.sort()
            by_host[host] = rows
    return by_host


def load_iters(out):
    """[(n, t, sps, ts_end)] from 'iter N: ... t=Xs sps=Y ... ts=T'."""
    iters = []
    for line in open(out, errors="ignore"):
        m = re.search(r"iter (\d+): .*t=([0-9.]+)s sps=(\S+).*ts=([0-9.]+)", line)
        if m:
            iters.append((int(m.group(1)), float(m.group(2)),
                          m.group(3), float(m.group(4))))
    ck = sorted(float(m.group(1)) for line in open(out, errors="ignore")
                if (m := re.search(r"MARK checkpoint .*ts=([0-9.]+)", line)))
    return iters, ck


def win_gpu(rows, t0, t1):
    if not rows:
        return None
    ts = [r[0] for r in rows]
    seg = rows[bisect_left(ts, t0):bisect_left(ts, t1)]
    if not seg:
        return None
    mem = [r[1] for r in seg]
    util = [r[2] for r in seg if r[2] >= 0]
    load = [r[3] for r in seg]
    return (min(mem), max(mem), (sum(util) / len(util) if util else -1),
            max(load))


def in_win(events, t0, t1, evt=None, phase=None):
    out = []
    for e in events:
        if t0 <= e.get("ts", 0) < t1 and (evt is None or e.get("evt") == evt) \
           and (phase is None or e.get("phase") == phase):
            out.append(e)
    return out


def stats(iters, resets, glitches, gpu, ck, thresh):
    def summarize(group):
        if not group:
            return "  (none)"
        nres = nheavy = nretry = nglitch = 0
        maxnav = 0.0
        ages = []
        memrise = utilhi = 0
        for (n, t, sps, te) in group:
            t0 = te - t
            rs = in_win(resets, t0, te)
            gs = in_win(glitches, t0, te)
            nres += len(rs); nglitch += len(gs)
            nheavy += sum(1 for r in rs if (r.get("total_ms") or 0) > 8000)
            nretry += sum(1 for r in rs if (r.get("nav_attempts") or 1) > 1)
            maxnav = max([maxnav] + [(r.get("navigate_ms") or 0) for r in rs])
            ages += [r.get("episode", 0) for r in rs]
            hosts = {r.get("host") for r in rs} | {g.get("host") for g in gs}
            for h in hosts:
                gw = win_gpu(gpu.get(h), t0, te)
                if gw and gw[1] - gw[0] > 300:
                    memrise += 1
                if gw and gw[2] > 85:
                    utilhi += 1
        k = len(group)
        agem = sorted(ages)[len(ages) // 2] if ages else 0
        return (f"  iters={k}  resets/iter={nres/k:.1f}  retried(nav>1)/iter={nretry/k:.2f}  "
                f"heavy(>8s)/iter={nheavy/k:.2f}  glitches/iter={nglitch/k:.2f}\n"
                f"    max navigate_ms={maxnav:.0f}  median browser-age={agem}  "
                f"iters w/ gpu-mem-rise>300MiB={memrise}  gpu-util>85%={utilhi}")

    strag = [it for it in iters if it[1] > thresh]
    normal = [it for it in iters if it[1] <= thresh]
    print(f"### STRAGGLER iters (t>{thresh}s): {len(strag)}/{len(iters)} ###")
    print(summarize(strag))
    print(f"\n### NORMAL iters (t<={thresh}s): {len(normal)}/{len(iters)} ###")
    print(summarize(normal))
    return strag


def main():
    evdir, out = sys.argv[1], sys.argv[2]
    thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
    ev = load_events(evdir)
    resets = [e for e in ev if e.get("evt") == "reset"]
    glitches = [e for e in ev if e.get("evt") == "glitch"]
    gpu = load_gpu(evdir)
    iters, ck = load_iters(out)
    runners = len(glob.glob(evdir + "/ev-*.jsonl"))
    print(f"loaded {len(resets)} resets, {len(glitches)} glitches from {runners} runners; "
          f"{len(gpu)} node GPU traces; {len(iters)} iters, {len(ck)} ckpts\n")
    if not iters:
        print("no iter lines parsed"); return

    strag = stats(iters, resets, glitches, gpu, ck, thresh)

    # per-straggler: earliest expensive event + preceding 15s context
    print(f"\n### per-straggler precursors ###")
    for (n, t, sps, te) in strag[:12]:
        t0 = te - t
        exp = sorted(
            [("reset", r["ts"], r) for r in in_win(resets, t0, te) if (r.get("total_ms") or 0) > 8000]
            + [("glitch", g["ts"], g) for g in in_win(glitches, t0, te)],
            key=lambda x: x[1])
        if not exp:
            print(f"  iter {n} (t={t:.0f}s): no heavy reset/glitch in window (stall elsewhere)")
            continue
        kind, ts0, e0 = exp[0]
        pre = [("reset", r) for r in in_win(resets, ts0 - 15, ts0)] \
            + [("glitch", g) for g in in_win(glitches, ts0 - 15, ts0)]
        det = (f"total_ms={e0.get('total_ms'):.0f} nav_attempts={e0.get('nav_attempts')} "
               f"age={e0.get('episode')}" if kind == "reset"
               else f"phase={e0.get('phase')} age={e0.get('episode')}")
        print(f"  iter {n} (t={t:.0f}s): first-expensive={kind} host={e0.get('host')} {det}; "
              f"preceding 15s: {len(pre)} events "
              f"({sum(1 for k,x in pre if k=='reset' and (x.get('nav_attempts') or 1)>1)} retried-resets, "
              f"{sum(1 for k,x in pre if k=='glitch')} glitches)")


if __name__ == "__main__":
    main()
