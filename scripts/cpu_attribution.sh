#!/bin/bash
# Per-process CPU attribution for a running simulator training, to settle which
# role actually owns the node-CPU wall (fetch/decode pool vs Ray env-runner
# actors vs Ray infra). Reads /proc utime+stime deltas over a window and buckets
# by cmdline. Pure stdlib python, no deps. Run INSIDE a running job's node
# (from the .slurm driver, or via `srun --jobid=<JID> --overlap --ntasks=1`).
#   bash scripts/cpu_attribution.sh [window_seconds]
set -u
DUR="${1:-8}"
python3 - "$DUR" <<'PY'
import os, sys, time, glob
dur = float(sys.argv[1])

def snap():
    out = {}
    for p in glob.glob('/proc/[0-9]*'):
        pid = p.rsplit('/', 1)[-1]
        try:
            with open(p + '/stat') as f:
                parts = f.read().split()
            jiffies = int(parts[13]) + int(parts[14])  # utime + stime
            with open(p + '/cmdline', 'rb') as f:
                cmd = f.read().replace(b'\x00', b' ').decode('utf8', 'replace').strip()
        except Exception:
            continue
        if cmd:
            out[pid] = (jiffies, cmd)
    return out

def bucket(cmd):
    if 'spawn_main' in cmd or 'from multiprocessing' in cmd:
        return 'fetch/decode pool (spawn workers)'
    if 'ray::' in cmd:
        return 'ray env-runner actors (render+step+dino-launch)'
    if any(k in cmd for k in ('raylet', 'gcs_server', 'plasma', 'dashboard',
                              'log_monitor', 'ray/autoscaler', 'ray/_private',
                              'runtime_env', 'ray.util')):
        return 'ray infra'
    if 'ngllib_agent.train' in cmd:
        return 'trainer main / learner'
    return 'other'

a = snap(); time.sleep(dur); b = snap()
hz = os.sysconf('SC_CLK_TCK'); ncpu = os.cpu_count()
agg = {}
for pid, (j1, cmd) in b.items():
    if pid in a:
        dj = j1 - a[pid][0]
        if dj > 0:
            agg[bucket(cmd)] = agg.get(bucket(cmd), 0.0) + dj
tot = sum(agg.values()) or 1.0
busy = tot / hz / dur
print(f"window={dur:.0f}s ncpu={ncpu} busy_cores~{busy:.1f} ({100*busy/ncpu:.0f}% of node)")
for k, v in sorted(agg.items(), key=lambda x: -x[1]):
    print(f"  {100*v/tot:5.1f}%  {v/hz/dur:5.1f} cores  {k}")
PY
