#!/usr/bin/env bash
# Corrected GPU-scaling sweep (A-series only; B-series already valid from run 1).
#
# Why v2: ngllib sets no per-GPU Vulkan device selection (environment.py passes
# --use-angle=vulkan and lets Chrome pick the first device = GPU0).
# CUDA_VISIBLE_DEVICES steers CUDA/DINO but NOT Chrome's render device, so the
# v1 A-series piled every browser onto GPU0 and OOM'd once the node total
# exceeded ~32 browsers (A2=64, A3=96 -> FAIL). The only way to make each
# process render on a distinct GPU is cgroup device isolation: run each in its
# own `srun --gres=gpu:1` step so it physically sees exactly one /dev/nvidia.
#
# Each step prints a DIAG line (visible GPU count) so we can verify isolation
# actually happened before trusting the scaling numbers.
set -u
cd /scratch/kp0374/neurogym-agent
M=32
N=100

run_series () {
  k=$1
  pids=""; logs=""
  for g in $(seq 1 "$k"); do
    log="/tmp/ps2_k${k}_g${g}.out"
    srun --exact --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=48G \
      bash -lc 'echo "DIAG CVD=[$CUDA_VISIBLE_DEVICES] visible_gpus=$(nvidia-smi -L | wc -l)"; cd /scratch/kp0374/neurogym-agent && UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python scripts/probe_throughput.py '"$M $N" >"$log" 2>&1 &
    pids="$pids $!"; logs="$logs $log"
  done
  for p in $pids; do wait "$p"; done

  total=0; detail=""
  for l in $logs; do
    diag=$(grep -oE 'visible_gpus=[0-9]+' "$l" | head -1)
    s=$(grep -oE 'sps=[0-9.]+' "$l" | head -1 | cut -d= -f2)
    if [ -z "$s" ]; then s=FAIL; else total=$(awk "BEGIN{print $total+$s}"); fi
    detail="$detail ${s}(${diag})"
  done
  echo "[procscale2] A${k}_${k}gpu_${k}x${M}: AGGREGATE=${total} sps  per-proc:${detail}"
}

for k in 1 2 3; do run_series "$k"; done
