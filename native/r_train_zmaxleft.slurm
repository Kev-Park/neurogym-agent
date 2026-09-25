#!/bin/bash
#SBATCH --job-name=zmaxleft
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=44
#SBATCH --mem=200G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_outputs/slurm-%x-%j.out

# zmax-left training — free-climb z across neurons via left-pane hops.
# From native/r_train_native.slurm; task deltas:
#   - config native_zmaxleft.yaml (reward.mode z_free, 500-step episodes,
#     uniform spawns, batch 16000/512 for real runs — BATCH overrides for smokes)
#   - CURRICULUM_PROGRESS_FILE now also drives the selection-novelty ANNEAL
#     (rewards/z_free.py), not just the spawn curriculum.
#   - USE_MPS=1 (default): per-job nvidia-cuda-mps daemon — the deployable
#     SOTA throughput lever on main (+~50% for per-process DINO, zero code).
#   - --learner-gpu: at batch 16000 the CPU-in-driver update would be a real
#     tax; the GPU learner measured 2-3 s/iter at this scale.
#
# Usage: sbatch native/r_train_zmaxleft.slurm [run-name] [target-iters]
#   smoke:  USE_MPS=0 BATCH=4000 ITERS=6 WANDB_MODE=disabled \
#           sbatch native/r_train_zmaxleft.slurm zmaxleft-smoke0 6
set -eu
WT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$WT"
RUN=${1:-zmaxleft-v1}
ITERS=${2:-200}
CKPT=/scratch/kp0374/checkpoints/${RUN}
mkdir -p "$CKPT"

export PYTHONUNBUFFERED=1
export RAY_NUM_CPUS=44
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NGL_NATIVE_FETCH_WORKERS=${FETCH_WORKERS:-2}
export NGL_NATIVE_MESH_LRU_MB=128
export NGL_NATIVE_CHUNK_LRU_MB=${CHUNK_LRU_MB:-256}
export NGL_NATIVE_VAO_LRU_MB=${VAO_LRU_MB:-256}
PANE_MODE=${PANE_MODE:-atomic}
RUNNERS=${RUNNERS:-32}
ENVS_PER_RUNNER=${ENVS_PER_RUNNER:-2}
BATCH=${BATCH:-16000}
WANDB_MODE=${WANDB_MODE:-online}
# Step-anchored spawn curriculum AND the z_free novelty-bonus anneal read this.
export CURRICULUM_PROGRESS_FILE="$CKPT/meta.json"

# --- Per-job CUDA MPS daemon (USE_MPS=0 disables, e.g. smokes) ---
USE_MPS=${USE_MPS:-1}
if [ "$USE_MPS" = "1" ]; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${SLURM_JOB_ID}
  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-${SLURM_JOB_ID}
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d && echo "[mps] daemon up" || echo "[mps] daemon FAILED (continuing without)"
fi

uv run --no-sync python -m ngllib_agent.train \
  --config ${TRAIN_CONFIG:-configs/native_zmaxleft.yaml} \
  --run-name "$RUN" --pane-mode "$PANE_MODE" \
  --num-env-runners "$RUNNERS" \
  --num-envs-per-env-runner "$ENVS_PER_RUNNER" \
  --num-gpus-per-env-runner 0.02 \
  --learner-gpu \
  --vector threads \
  --iters "$ITERS" \
  --train-batch-size "$BATCH" \
  --checkpoint-dir "$CKPT" --checkpoint-every 10 \
  --wandb-project neurogym-agent --wandb-mode "$WANDB_MODE" \
  --resume

if [ "$USE_MPS" = "1" ]; then
  echo quit | nvidia-cuda-mps-control || true
fi
