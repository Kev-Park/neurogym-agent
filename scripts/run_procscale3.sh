#!/usr/bin/env bash
# Node-kernel-limit vs CPU-starvation test (the definitive one).
#
# The v2 GPU-scaling flat result was confounded: each process had only 8 cores
# and shared the node with a highpri array. This runs on an EXCLUSIVE node
# (r_procscale3.slurm) so each cgroup-isolated GPU-process gets ~16 cores with
# ZERO co-tenant, then scales 1->2->3->4 GPUs at full M=32/GPU.
#   - aggregate climbs ~linearly  => earlier flat was starvation; NO node lock,
#                                     the ceiling is per-GPU (copy engine).
#   - aggregate stays flat         => genuine node-level serializer (shared
#                                     driver / host mem / PCIe) above the GPU.
set -u
cd /scratch/kp0374/neurogym-agent
M=32; N=100; CPUS=16; MEM=48G

run_series () {
  k=$1
  pids=""; logs=""
  for g in $(seq 1 "$k"); do
    log="/tmp/ps3_k${k}_g${g}.out"
    srun --exact --ntasks=1 --gres=gpu:1 --cpus-per-task=$CPUS --mem=$MEM \
      bash -lc 'echo "DIAG CVD=[$CUDA_VISIBLE_DEVICES] gpus=$(nvidia-smi -L | wc -l) cpus=$(nproc)"; cd /scratch/kp0374/neurogym-agent && UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python scripts/probe_throughput.py '"$M $N" >"$log" 2>&1 &
    pids="$pids $!"; logs="$logs $log"
  done
  for p in $pids; do wait "$p"; done
  total=0; detail=""
  for l in $logs; do
    diag=$(grep -oE 'gpus=[0-9]+ cpus=[0-9]+' "$l" | head -1)
    s=$(grep -oE 'sps=[0-9.]+' "$l" | head -1 | cut -d= -f2)
    if [ -z "$s" ]; then s=FAIL; else total=$(awk "BEGIN{print $total+$s}"); fi
    detail="$detail ${s}[${diag}]"
  done
  echo "[procscale3] k=${k} (${k}gpu x M${M}, ${CPUS}cpu/proc): AGGREGATE=${total} sps  per-proc:${detail}"
}

for k in 1 2 3 4; do run_series "$k"; done
