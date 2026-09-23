#!/bin/bash
# Node-max arm with CORRECT CPU pinning via taskset to nvidia-smi's reported per-GPU
# CPU affinity (GPU0-3 -> cores 0-15,32-47 = NUMA0; GPU4-7 -> 16-31,48-63 = NUMA1),
# NOT numactl --cpunodebind (whose node core-lists don't match the GPU affinity on
# this box). PIN=1 taskset-pins each training to its GPU-local cores + numactl membind
# to the GPU's NUMA node; PIN=0 unpinned baseline. Prints all warm iters per GPU.
#   bash scripts/node_max_arm_taskset.sh <G> <N> <PIN:0|1>
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
G="$1"; N="$2"; PIN="${3:-1}"; J="${SLURM_JOB_ID:-manual}"
echo "[ts] G=$G N=$N PIN=$PIN start $(date +%H:%M:%S)"
pids=""
for i in $(seq 0 $((G - 1))); do
  if [ "$i" -lt $(( G / 2 )) ]; then cores="0-15,32-47"; node=0; else cores="16-31,48-63"; node=1; fi
  PRE=""
  if [ "$PIN" = "1" ]; then PRE="taskset -c $cores numactl --membind=$node"; fi
  CUDA_VISIBLE_DEVICES=$i UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
    NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J-$i.lock \
    $PRE uv run --no-sync python -m ngllib_agent.train \
      --config configs/native_rb_interop.yaml --run-name "ts$PIN-g$G-n$N-gpu$i-$J" \
      --no-spawn-curriculum --learner-gpu --num-env-runners "$N" --num-envs-per-env-runner 2 \
      --num-gpus-per-env-runner 0.03 --num-cpus-per-env-runner 0.5 --vector threads \
      --iters "${ITERS:-10}" --train-batch-size 24000 --checkpoint-every 999999 \
      --no-degraded-exit --wandb-mode disabled \
      > "slurm_outputs/ts$PIN-g$G-n$N-gpu$i-$J.out" 2>&1 &
  pids="$pids $!"
  echo "[ts] gpu$i cores=$cores node=$node pin=$PIN pid $!"
done
wait $pids
echo "[ts] G=$G N=$N PIN=$PIN done $(date +%H:%M:%S)"
