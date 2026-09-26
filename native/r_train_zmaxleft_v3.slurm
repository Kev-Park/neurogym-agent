#!/bin/bash
#SBATCH --job-name=zmaxleft-v3
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=44
#SBATCH --mem=360G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_outputs/slurm-%x-%j.out

# zmax-left v3 — efficient free climb. From r_train_zmaxleft.slurm; deltas:
#   - config native_zmaxleft_v3.yaml: per-episode random-budget TERMINATION
#     (U[100,500], unobserved) + stagnation terminal (125 steps no new z-best)
#     = minimal-bias earliness pressure (see the config header).
#   - NGL_NATIVE_MESH_PREFETCH=4: speculative coarse-mesh prefetch of the four
#     largest on-slice 2D-pane segments, claimed on selection, so hop payoffs
#     aren't fetch-delayed and climb-vs-hop competes on merit.
#   - shorter mean episodes => more resets/batch; same 24x2 topology (resets
#     already overlap via warm()).
#
# Usage: sbatch native/r_train_zmaxleft_v3.slurm [run-name] [target-iters]
#   smoke:  USE_MPS=0 BATCH=4000 ITERS=6 WANDB_MODE=disabled \
#           sbatch native/r_train_zmaxleft_v3.slurm zmaxleft-v3-smoke0 6
set -eu
WT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$WT"
RUN=${1:-zmaxleft-v3}
ITERS=${2:-300}
CKPT=/scratch/kp0374/checkpoints/${RUN}
mkdir -p "$CKPT"

export PYTHONUNBUFFERED=1
# RAY_CPUS lets a downsized submit (--cpus-per-task=32 --mem=240G
# --export=ALL,RAY_CPUS=32,RUNNERS=16) fit co-tenanted nodes; memory and env
# count co-scale (~7.5G/env at the 24x2 profile, v2 MaxRSS ~360G at 48 envs).
export RAY_NUM_CPUS=${RAY_CPUS:-44}
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NGL_NATIVE_FETCH_WORKERS=${FETCH_WORKERS:-2}
export NGL_NATIVE_MESH_LRU_MB=128
export NGL_NATIVE_CHUNK_LRU_MB=${CHUNK_LRU_MB:-256}
export NGL_NATIVE_VAO_LRU_MB=${VAO_LRU_MB:-128}
export NGL_NATIVE_MESH_PREFETCH=${MESH_PREFETCH:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PANE_MODE=${PANE_MODE:-atomic}
RUNNERS=${RUNNERS:-24}
ENVS_PER_RUNNER=${ENVS_PER_RUNNER:-2}
BATCH=${BATCH:-16000}
WANDB_MODE=${WANDB_MODE:-online}
export CURRICULUM_PROGRESS_FILE="$CKPT/meta.json"

USE_MPS=${USE_MPS:-1}
if [ "$USE_MPS" = "1" ]; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${SLURM_JOB_ID}
  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-${SLURM_JOB_ID}
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d && echo "[mps] daemon up" || echo "[mps] daemon FAILED (continuing without)"
fi

uv run --no-sync python -m ngllib_agent.train \
  --config ${TRAIN_CONFIG:-configs/native_zmaxleft_v3.yaml} \
  --run-name "$RUN" --pane-mode "$PANE_MODE" \
  --num-env-runners "$RUNNERS" \
  --num-envs-per-env-runner "$ENVS_PER_RUNNER" \
  --num-gpus-per-env-runner 0.02 \
  --vector threads \
  --sample-timeout-s "${SAMPLE_TIMEOUT:-600}" \
  --iters "$ITERS" \
  --train-batch-size "$BATCH" \
  --checkpoint-dir "$CKPT" --checkpoint-every 10 \
  --wandb-project neurogym-agent --wandb-mode "$WANDB_MODE" \
  --resume

if [ "$USE_MPS" = "1" ]; then
  echo quit | nvidia-cuda-mps-control || true
fi
