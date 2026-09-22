#!/bin/bash
# Diagnose the N>=~30 GL/render context-init HANG: launch N=36 (hangs at init), then
# py-spy-dump the stuck runner processes to see WHERE init is blocked -- eglCreate*
# / moderngl create_context? a threading/mutex acquire? MPS/CUDA context create?
# CloudVolume? Full stacks + (best-effort) native stacks to reach the C deadlock.
# Also GPU context count + VRAM. No stagger (diagnose the raw hang).
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J="${SLURM_JOB_ID:-manual}"; OUT=slurm_outputs; PYSPY=.venv/bin/py-spy
echo "[glinit] launching N=36 (expected to hang at init) job=$J"
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
  uv run --no-sync python -m ngllib_agent.train \
    --config configs/native_rb_interop.yaml --run-name "glinit-$J" \
    --no-spawn-curriculum --learner-gpu --num-env-runners 36 --num-envs-per-env-runner 2 \
    --num-gpus-per-env-runner 0.03 --num-cpus-per-env-runner 0.5 --vector threads \
    --iters 3 --train-batch-size 24000 --checkpoint-every 999999 --no-degraded-exit \
    --wandb-mode disabled > "$OUT/glinit-train-$J.out" 2>&1 &
TP=$!
echo "[glinit] train pid $TP; waiting 210s for the hung-init state"
sleep 210

echo "[glinit] === runner PIDs ==="
pgrep -f 'SingleAgentEnvRunner' | head -8
echo "[glinit] === render-progress markers (0 => all stuck before first render) ==="
grep -c 'tiles fine on screen' "$OUT/glinit-train-$J.out" 2>/dev/null || true
for p in $(pgrep -f 'SingleAgentEnvRunner' | head -3); do
  echo "===== py-spy dump (python) pid $p ====="
  $PYSPY dump --pid "$p" 2>&1 | head -55
  echo "===== py-spy dump (native) pid $p ====="
  $PYSPY dump --pid "$p" --native 2>&1 | head -60
done

echo "[glinit] === GPU state (VRAM + compute-app/context count) ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null
echo "compute apps:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | wc -l
echo "[glinit] === MPS server log tail ==="
tail -8 "$CUDA_MPS_LOG_DIRECTORY/control.log" 2>/dev/null || echo "no mps control.log"

kill "$TP" 2>/dev/null || true
pkill -f "glinit-$J" 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
echo "[glinit] done"
