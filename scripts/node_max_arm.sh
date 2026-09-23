#!/bin/bash
# One node-max arm: launch G independent trainings (one per GPU, CUDA_VISIBLE_DEVICES
# 0..G-1), each N runners x M=2, graphs on (native_rb_interop.yaml SOTA). Waits, then
# prints per-GPU warm sps so the aggregate = sum. Sweeps G vs N to find the per-NODE
# max under the 64-core / 376G budget (8 GPUs available but cores/RAM are the cap).
#   bash scripts/node_max_arm.sh <G> <N>
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
G="$1"; N="$2"; J="${SLURM_JOB_ID:-manual}"
echo "[nodemax] G=$G N=$N (total runners=$((G*N))) cores=$(nproc) start $(date +%H:%M:%S)"
pids=""
for i in $(seq 0 $((G-1))); do
  CUDA_VISIBLE_DEVICES=$i UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
    NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J-$i.lock \
    uv run --no-sync python -m ngllib_agent.train \
      --config configs/native_rb_interop.yaml --run-name "nodemax-g$G-n$N-gpu$i-$J" \
      --no-spawn-curriculum --learner-gpu --num-env-runners "$N" --num-envs-per-env-runner 2 \
      --num-gpus-per-env-runner 0.03 --num-cpus-per-env-runner 0.5 --vector threads \
      --iters "${ITERS:-8}" --train-batch-size 24000 --checkpoint-every 999999 \
      --no-degraded-exit --wandb-mode disabled \
      > "slurm_outputs/nodemax-g$G-n$N-gpu$i-$J.out" 2>&1 &
  pids="$pids $!"
done
wait $pids
echo "[nodemax] G=$G N=$N per-GPU warm sps (sum = node aggregate):"
for i in $(seq 0 $((G-1))); do
  printf "  gpu%s: " "$i"
  grep -hE '^iter ' "slurm_outputs/nodemax-g$G-n$N-gpu$i-$J.out" 2>/dev/null | grep -oE 'sps=[0-9.]+|H=[0-9]+/[0-9]+' | tail -6 | tr '\n' ' '
  echo
done
echo "[nodemax] G=$G N=$N done $(date +%H:%M:%S)"
