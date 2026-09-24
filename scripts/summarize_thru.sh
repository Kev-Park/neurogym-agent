#!/usr/bin/env bash
# sps per arm from a paired loginthru log
f=${1:?log}
python3 - "$f" <<'PY'
import re, sys, statistics
arm=None; arms={}
for line in open(sys.argv[1], errors="ignore"):
    m=re.search(r"=== ARM (\w+)", line)
    if m: arm=m.group(1); arms.setdefault(arm, [])
    m=re.search(r"^iter \d+:.*sps=([0-9.]+)", line)
    if m and arm: arms[arm].append(float(m.group(1)))
for a,v in arms.items():
    warm=v[1:] if len(v)>1 else v
    print(f"{a:9s} n={len(v):2d} all={['%.1f'%x for x in v]}")
    if warm:
        print(f"{'':9s} warm median={statistics.median(warm):.1f} mean={statistics.mean(warm):.1f} "
              f"min={min(warm):.1f} max={max(warm):.1f}")
PY
