#!/usr/bin/env bash
# Launch P concurrent probe_e2e_batched processes on ONE GPU (process-packing
# sweep, opt 2) and sum their sps. Total browsers P*M must fit VRAM (<=~32/GPU).
#   bash scripts/run_e2e_sweep.sh <P> <M> <baseline|proto>
set -u
cd /scratch/kp0374/neurogym-agent
P=$1; M=$2; MODE=$3
pids=""; logs=""
for g in $(seq 1 "$P"); do
  log="/tmp/e2e_${MODE}_P${P}_M${M}_${g}.out"
  UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
    uv run --no-sync python scripts/probe_e2e_batched.py "$M" "$MODE" >"$log" 2>&1 &
  pids="$pids $!"; logs="$logs $log"
done
for p in $pids; do wait "$p"; done
total=0; detail=""
for l in $logs; do
  s=$(grep -oE 'sps=[0-9.]+' "$l" | head -1 | cut -d= -f2)
  if [ -z "$s" ]; then s=FAIL; else total=$(awk "BEGIN{print $total+$s}"); fi
  detail="$detail ${s}"
done
echo "[e2e-sweep] mode=${MODE} P=${P} M=${M} (total_browsers=$((P*M))): AGGREGATE=${total} sps  per-proc:${detail}"
