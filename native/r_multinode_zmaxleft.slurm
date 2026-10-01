#!/bin/bash
#SBATCH --job-name=zl-multinode
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3090:4
#SBATCH --cpus-per-task=44
#SBATCH --mem=360G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=00:55:00
#SBATCH --output=slurm_outputs/slurm-%x-%j.out
#
# Does the co-pane SOTA layout scale across NODES for ONE training? SOTA per node =
# 24 runners x M=2 over 4 GPUs (6 runners/GPU), v6 config (CUDA graphs), ngllib EGL
# fix. Here one RLlib training's env runners span a 2-node Ray cluster (head on
# node 0, worker joined on node 1), learner stays on the head's CPU. Two arms on
# the SAME head node so node variance cancels:
#   A  1 node : cluster = head only,  24 runners, batch 16000
#   B  2 nodes: worker joined,        48 runners (exactly 24/node: 4 GPUs x 6
#              slots each), batch 32000 so per-iteration sampling time and the
#              learner-barrier share stay comparable to A.
# Reported per arm: MEDIAN per-iteration sps (cold first iter excluded — the
# ~200 s reset/fetch-storm stalls make short-window means lie), per-node
# nvidia-smi memory.used (placement check: ~equal across each node's 4 GPUs),
# and the raw per-iteration list. Follows r_train_val.slurm's Ray bootstrap.
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_outputs
J=$SLURM_JOB_ID
PKL=${1:-/scratch/kp0374/checkpoints/zmaxleft-v5/ckpt_000490.pkl}
WINDOW=${WINDOW:-780}
GPU_FRAC=${GPU_FRAC:-0.166}      # 24 x 0.166 -> exactly 6 runners per GPU
TMP=/tmp/zlmn_$J

export PYTHONUNBUFFERED=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_memory_monitor_refresh_ms=0
export UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_FETCH_WORKERS=2 NGL_NATIVE_MESH_LRU_MB=128 NGL_NATIVE_CHUNK_LRU_MB=256
export NGL_NATIVE_VAO_LRU_MB=128 NGL_NATIVE_MESH_PREFETCH=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J.lock      # node-local, fine
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J

NODES=($(scontrol show hostnames "$SLURM_NODELIST"))
HEAD=${NODES[0]}; WORKER=${NODES[1]}
HEAD_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" hostname -I | awk '{print $1}')
echo "[mn] nodes: ${NODES[*]}  head=$HEAD ($HEAD_IP)  worker=$WORKER  window=${WINDOW}s"

# MPS daemon on EACH node (env is inherited by the step; runners spawned there
# find the pipe dir).
for n in "${NODES[@]}"; do
  srun --overlap --nodes=1 --ntasks=1 -w "$n" bash -c \
    'mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" && nvidia-cuda-mps-control -d && echo "[mn] MPS up on $(hostname)"' || true
done

# Ray head on node 0 (4 GPUs = the per-node gres; SLURM exposes them as 0-3).
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
  uv run --no-sync ray start --head --node-ip-address="$HEAD_IP" --port=6379 \
    --num-cpus=44 --num-gpus=4 --block &
HEAD_PID=$!
sleep 25
export RAY_ADDRESS="$HEAD_IP:6379"

run_arm() {
  local name="$1" nrun="$2" batch="$3"; shift 3
  local dir="$TMP/$name" f="slurm_outputs/mn-$name-$J.out"
  mkdir -p "$dir"; cp -f "$PKL" "$dir/ckpt_000490.pkl"
  echo "[mn] === arm $name: $nrun runners (M=2) batch=$batch on nodes: $* ==="
  srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
    uv run --no-sync python -m ngllib_agent.train \
      --config configs/native_zmaxleft_v6.yaml --run-name "mn-$name-$J" \
      --obs dino --vector threads \
      --num-env-runners "$nrun" --num-envs-per-env-runner 2 \
      --num-gpus-per-env-runner "$GPU_FRAC" --num-cpus-per-env-runner 1.0 \
      --no-spawn-curriculum --no-degraded-exit --stagger-first-episode \
      --iters 999999 --train-batch-size "$batch" \
      --checkpoint-dir "$dir" --checkpoint-every 999999 --resume \
      --wandb-mode disabled > "$f" 2>&1 &
  local pid=$!
  sleep "$WINDOW"
  local n
  for n in "$@"; do
    srun --overlap --nodes=1 --ntasks=1 -w "$n" nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
      | sed "s/^/[mn] $name place $n gpu /"
  done
  local list med iters healthy
  list=$(grep -hE '^iter ' "$f" 2>/dev/null | grep -oE 'sps=[0-9.]+' | sed 's/sps=//' | tr '\n' ' ')
  med=$(grep -hE '^iter ' "$f" 2>/dev/null | grep -oE 'sps=[0-9.]+' | sed 's/sps=//' | tail -n +2 | sort -n \
        | awk '{a[NR]=$1} END{if(NR==0){print 0}else if(NR%2){printf "%.1f",a[(NR+1)/2]}else{printf "%.1f",(a[NR/2]+a[NR/2+1])/2}}')
  iters=$(grep -cE '^iter ' "$f" 2>/dev/null); iters=${iters:-0}
  healthy=$(grep -hoE 'H=[0-9]+/[0-9]+' "$f" 2>/dev/null | tail -1)
  echo "[mn] $name per-iter sps: $list"
  echo "[mn] RESULT arm=$name nodes=$# runners=$nrun batch=$batch median_sps=$med iters=$iters last_H=${healthy:-NA}"
  kill "$pid" 2>/dev/null || true
  sleep 10
  for n in "$@"; do
    srun --overlap --nodes=1 --ntasks=1 -w "$n" bash -c "pkill -9 -f 'mn-$name-$J' ; pkill -9 -f 'multiprocessing.spawn' ; true" 2>/dev/null || true
  done
  sleep 10
}

# ---- A: single node (cluster = head only) --------------------------------------
run_arm A 24 16000 "$HEAD"

# ---- join the worker, then B: two nodes ----------------------------------------
srun --overlap --nodes=1 --ntasks=1 -w "$WORKER" \
  uv run --no-sync ray start --address="$HEAD_IP:6379" --num-cpus=44 --num-gpus=4 --block &
WORKER_PID=$!
sleep 25
echo "[mn] === ray status (2 nodes) ==="
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" uv run --no-sync ray status --address="$HEAD_IP:6379" 2>&1 | grep -E 'GPU|CPU|node' | head -8 || true
run_arm B 48 32000 "$HEAD" "$WORKER"

echo "[mn] === SUMMARY ==="
grep -hE '^\[mn\] RESULT ' "slurm_outputs/slurm-zl-multinode-$J.out" 2>/dev/null || true
for n in "${NODES[@]}"; do
  srun --overlap --nodes=1 --ntasks=1 -w "$n" bash -c 'uv run --no-sync ray stop --force >/dev/null 2>&1; echo quit | nvidia-cuda-mps-control >/dev/null 2>&1; true' || true
done
kill "$HEAD_PID" "${WORKER_PID:-}" 2>/dev/null || true
rm -rf "$TMP" 2>/dev/null || true
echo "[mn] done"
