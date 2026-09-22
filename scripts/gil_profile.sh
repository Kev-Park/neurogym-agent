#!/bin/bash
# Attribute the throughput bottleneck: GIL contention vs CPU-under-utilization vs
# RLlib overhead. Launches training at (M,N), warms it, then during steady-state
# SAMPLING captures, for a runner (env-runner) process:
#   - mpstat per-core CPU utilization (are cores actually busy, or idle?)
#   - py-spy dumps (per-thread active/idle + stack: how many threads make progress
#     at once = GIL parallelism; WHICH code they're in = env-glue vs RLlib)
#   - py-spy flamegraph (full) + GIL-only flamegraph (what holds the GIL)
# Interpretation:
#   cores idle + few threads active + GIL flame in env/RLlib Python -> GIL-bound
#   cores saturated + threads in C (torch/fetch)                     -> compute-bound
#   GIL/active time dominated by RLlib connector/sampler frames      -> RLlib overhead
#   bash scripts/gil_profile.sh <M> <N> <label>
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
M="$1"; N="$2"; LBL="$3"; J="${SLURM_JOB_ID:-manual}"
OUT=slurm_outputs; PYSPY=.venv/bin/py-spy
NPROC=$(nproc)
echo "[gil] arm=$LBL M=$M N=$N cores=$NPROC job=$J"

UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
  uv run --no-sync python -m ngllib_agent.train \
    --config configs/native_rb_interop.yaml --run-name "gil-$LBL-$J" \
    --no-spawn-curriculum --learner-gpu --num-env-runners "$N" --num-envs-per-env-runner "$M" \
    --num-gpus-per-env-runner 0.03 --num-cpus-per-env-runner 0.5 --vector threads \
    --iters 30 --train-batch-size 24000 --checkpoint-every 999999 --no-degraded-exit \
    --wandb-mode disabled > "$OUT/gil-train-$LBL-$J.out" 2>&1 &
TRAIN_PID=$!
echo "[gil] train pid $TRAIN_PID; warming 170s"
sleep 170

RPID=$(pgrep -f 'SingleAgentEnvRunner' | head -1)
[ -z "$RPID" ] && RPID=$(pgrep -f 'ray::' | grep -v "$$" | head -1)
if [ -z "$RPID" ]; then echo "[gil] NO RUNNER PID FOUND"; ps -u kp0374 -o pid,pcpu,nlwp,comm | sort -k2 -rn | head; fi
echo "[gil] runner pid=$RPID  (threads/cpu:)"; ps -o pid,pcpu,nlwp,comm --pid "$RPID" 2>/dev/null || true

echo "[gil] === mpstat per-core, 30s (machine-wide CPU utilization) ==="
mpstat -P ALL 3 10 > "$OUT/gil-mpstat-$LBL-$J.txt" 2>&1
echo "[gil] avg %idle (last 'Average' block):"
grep -E 'Average' "$OUT/gil-mpstat-$LBL-$J.txt" | tail -n $((NPROC+2))

echo "[gil] === py-spy dumps x12 (thread active/idle + GIL + stack) ==="
: > "$OUT/gil-dumps-$LBL-$J.txt"
for i in $(seq 1 12); do
  echo "----- dump $i (pid $RPID) -----" >> "$OUT/gil-dumps-$LBL-$J.txt"
  $PYSPY dump --pid "$RPID" >> "$OUT/gil-dumps-$LBL-$J.txt" 2>&1 || echo "dump failed" >> "$OUT/gil-dumps-$LBL-$J.txt"
  sleep 2
done
echo "[gil] active-thread lines across dumps:"; grep -cE '\(active' "$OUT/gil-dumps-$LBL-$J.txt" || true
echo "[gil] idle-thread lines across dumps:";   grep -cE '\(idle'   "$OUT/gil-dumps-$LBL-$J.txt" || true

echo "[gil] === py-spy flamegraphs (full + GIL-only) ==="
$PYSPY record --pid "$RPID" --duration 20 --rate 120 --format flamegraph \
  -o "$OUT/gil-flame-$LBL-$J.svg" 2>&1 | tail -2 || echo "flame failed"
$PYSPY record --pid "$RPID" --duration 20 --rate 120 --gil --format flamegraph \
  -o "$OUT/gil-flamegil-$LBL-$J.svg" 2>&1 | tail -2 || echo "flamegil failed"

echo "[gil] stopping training"
kill "$TRAIN_PID" 2>/dev/null || true
pkill -f "gil-$LBL-$J" 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
sleep 8
echo "[gil] arm=$LBL done; leftover python:"; pgrep -u kp0374 -f 'ngllib_agent.train' | wc -l
