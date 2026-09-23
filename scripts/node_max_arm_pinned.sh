#!/bin/bash
# Node-max arm with optional NUMA/CPU pinning, for the variance-mitigation A/B.
# Launches G trainings (one per GPU). With PIN=1, each is numactl-bound to its GPU's
# NUMA node (cpunodebind + membind) so trainings don't bounce across sockets or fight
# for remote memory -- the recommended fix for the 6-8-way co-tenancy variance. Prints
# ALL warm iters (>=4) per GPU so mean AND spread (std) can be computed.
#   bash scripts/node_max_arm_pinned.sh <G> <N> <PIN:0|1>
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
G="$1"; N="$2"; PIN="${3:-1}"; J="${SLURM_JOB_ID:-manual}"
have_numactl=1; command -v numactl >/dev/null 2>&1 || have_numactl=0
nvidia-smi topo -m > "slurm_outputs/pin-topo-$J.txt" 2>&1 || true
echo "[pin] G=$G N=$N PIN=$PIN numactl=$have_numactl start $(date +%H:%M:%S)"
pids=""
for i in $(seq 0 $((G - 1))); do
  numa=$(( i * 2 / G ))     # G=8: gpu0-3->numa0, gpu4-7->numa1 (verify vs pin-topo)
  PRE=""
  if [ "$PIN" = "1" ] && [ "$have_numactl" = "1" ]; then
    PRE="numactl --cpunodebind=$numa --membind=$numa"
  fi
  CUDA_VISIBLE_DEVICES=$i UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
    NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J-$i.lock \
    $PRE uv run --no-sync python -m ngllib_agent.train \
      --config configs/native_rb_interop.yaml --run-name "pin$PIN-g$G-n$N-gpu$i-$J" \
      --no-spawn-curriculum --learner-gpu --num-env-runners "$N" --num-envs-per-env-runner 2 \
      --num-gpus-per-env-runner 0.03 --num-cpus-per-env-runner 0.5 --vector threads \
      --iters "${ITERS:-10}" --train-batch-size 24000 --checkpoint-every 999999 \
      --no-degraded-exit --wandb-mode disabled \
      > "slurm_outputs/pin$PIN-g$G-n$N-gpu$i-$J.out" 2>&1 &
  pids="$pids $!"
  echo "[pin] gpu$i numa=$numa pin=$PIN pid $!"
done
wait $pids
echo "[pin] G=$G N=$N PIN=$PIN per-GPU warm sps (iters>=4, for mean+std):"
for i in $(seq 0 $((G - 1))); do
  printf "  gpu%s: " "$i"
  grep -hE '^iter (4|5|6|7|8|9|1[0-9])' "slurm_outputs/pin$PIN-g$G-n$N-gpu$i-$J.out" 2>/dev/null | grep -oE 'sps=[0-9.]+' | tr '\n' ' '
  echo
done
echo "[pin] G=$G N=$N PIN=$PIN done $(date +%H:%M:%S)"
