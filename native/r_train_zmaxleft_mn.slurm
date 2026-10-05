#!/bin/bash
#SBATCH --job-name=zmaxleft-mn
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3090:4
#SBATCH --cpus-per-task=44
#SBATCH --mem=360G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_outputs/slurm-%x-%j.out

# zmax-left MULTI-NODE production training: ONE RLlib training whose env
# runners span a Ray cluster of all the job's nodes, at the per-node co-pane
# SOTA layout (24 runners x M=2 over 4 GPUs, 6 runners/GPU, CUDA graphs, MPS;
# r_train_zmaxleft_v6.slurm). Ray head + learner (GPU) on node 0, workers
# joined from the other nodes. Bootstrap follows r_multinode_zmaxleft.slurm
# (the 1-vs-2-node scaling test). Batch stays FIXED at 16000: more nodes =
# more envs = shorter fragments and faster iterations, NOT a bigger batch
# with fewer updates (fragment length is a null effect, iteration count is
# ~20pp -- project_v9test_chimera notes, 2026-09-02).
#
# Usage: sbatch --nodes=N [--job-name=...] \
#          --export=ALL,TRAIN_CONFIG=configs/native_zmaxleft_v7.yaml \
#          native/r_train_zmaxleft_mn.slurm <run-name> <target-iters>
set -u
WT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$WT"
mkdir -p slurm_outputs
RUN=${1:-zmaxleft-mn}
ITERS=${2:-450}
CKPT=/scratch/kp0374/checkpoints/${RUN}
mkdir -p "$CKPT"
J=$SLURM_JOB_ID

export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_memory_monitor_refresh_ms=0
export UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_FETCH_WORKERS=${FETCH_WORKERS:-2}
export NGL_NATIVE_MESH_LRU_MB=128
export NGL_NATIVE_CHUNK_LRU_MB=${CHUNK_LRU_MB:-256}
export NGL_NATIVE_VAO_LRU_MB=${VAO_LRU_MB:-128}
export NGL_NATIVE_MESH_PREFETCH=${MESH_PREFETCH:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J.lock      # node-local
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
export CURRICULUM_PROGRESS_FILE="$CKPT/meta.json"
PANE_MODE=${PANE_MODE:-atomic}
RUNNERS_PER_NODE=${RUNNERS_PER_NODE:-24}
ENVS_PER_RUNNER=${ENVS_PER_RUNNER:-2}
GPU_FRAC=${GPU_FRAC:-0.166}   # 24 x 0.166 -> Ray packs exactly 6 runners per GPU
BATCH=${BATCH:-16000}
WANDB_MODE=${WANDB_MODE:-online}

NODES=($(scontrol show hostnames "$SLURM_NODELIST"))
NNODES=${#NODES[@]}
HEAD=${NODES[0]}
HEAD_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" hostname -I | awk '{print $1}')
# The GPU learner moves the whole train batch to its device before the update:
# with the token obs (16000 x 2 panes x 256 x 384 fp16) that is ~6 GB, which
# OOM'd GPU0 when 6 runners' GL+DINO already sat on it (job 999395). So the
# head node's Ray gets GPUs 1-3 only and GPU0 is the learner's alone; the head
# runs 3 GPUs x 6 = 18 runners instead of 24.
HEAD_RUNNERS=$((RUNNERS_PER_NODE * 3 / 4))
RUNNERS=$((HEAD_RUNNERS + RUNNERS_PER_NODE * (NNODES - 1)))
echo "[mn] nodes(${NNODES}): ${NODES[*]}  head=$HEAD ($HEAD_IP)  runners=$RUNNERS x M=$ENVS_PER_RUNNER batch=$BATCH"

cleanup() {
  for n in "${NODES[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "$n" bash -c \
      'uv run --no-sync ray stop --force >/dev/null 2>&1; echo quit | nvidia-cuda-mps-control >/dev/null 2>&1; true' || true
  done
}
trap cleanup EXIT

# MPS daemon on each node (the step inherits the pipe-dir env).
for n in "${NODES[@]}"; do
  srun --overlap --nodes=1 --ntasks=1 -w "$n" bash -c \
    'mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" && nvidia-cuda-mps-control -d && echo "[mn] MPS up on $(hostname)"' || true
done

# Ray head on node 0, then every other node joins.
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
  /usr/bin/env CUDA_VISIBLE_DEVICES=1,2,3 \
  uv run --no-sync ray start --head --node-ip-address="$HEAD_IP" --port=6379 \
    --num-cpus=44 --num-gpus=3 --block &
sleep 25
for n in "${NODES[@]:1}"; do
  srun --overlap --nodes=1 --ntasks=1 -w "$n" \
    uv run --no-sync ray start --address="$HEAD_IP:6379" --num-cpus=44 --num-gpus=4 --block &
done
sleep 25
export RAY_ADDRESS="$HEAD_IP:6379"
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" uv run --no-sync ray status --address="$HEAD_IP:6379" 2>&1 \
  | grep -E 'GPU|CPU|node' | head -8 || true

# train.py exits 43 on sustained degradation after checkpointing; a fresh
# process resuming from that checkpoint is the cure, so relaunch in place
# (train.py disables the exit itself after 3 in 2h). Any OTHER non-zero exit
# (e.g. a transient learner CUDA OOM right after a runner restart, job
# 1002322 at iter 778) is also relaunched from the last checkpoint, at most
# MAX_CRASH_RETRIES times, so one fault does not end the job.
crashes=0
while :; do
  srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
    /usr/bin/env CUDA_VISIBLE_DEVICES=0 \
    uv run --no-sync python -m ngllib_agent.train \
      --config "${TRAIN_CONFIG:-configs/native_zmaxleft_v7.yaml}" \
      --run-name "$RUN" --pane-mode "$PANE_MODE" \
      --num-env-runners "$RUNNERS" \
      --num-envs-per-env-runner "$ENVS_PER_RUNNER" \
      --num-gpus-per-env-runner "$GPU_FRAC" --num-cpus-per-env-runner 1.0 \
      --vector threads \
      --sample-timeout-s "${SAMPLE_TIMEOUT:-600}" \
      --iters "$ITERS" \
      --train-batch-size "$BATCH" \
      --checkpoint-dir "$CKPT" --checkpoint-every 10 \
      --wandb-project neurogym-agent --wandb-mode "$WANDB_MODE" \
      --learner-gpu \
      --resume
  rc=$?
  [ "$rc" -eq 0 ] && break
  if [ "$rc" -eq 43 ]; then
    echo "[mn] degraded exit (43) -> relaunching from checkpoint"
    continue
  fi
  crashes=$((crashes + 1))
  if [ "$crashes" -gt "${MAX_CRASH_RETRIES:-3}" ]; then
    echo "[mn] train crashed (rc=$rc) $crashes times -> giving up"
    break
  fi
  echo "[mn] train crashed (rc=$rc) -> relaunching from checkpoint ($crashes/${MAX_CRASH_RETRIES:-3})"
  sleep 30
done
echo "[mn] train exited rc=$rc"
exit "$rc"
