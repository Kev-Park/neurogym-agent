#!/bin/bash
#SBATCH --job-name=zmaxleft-v6
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:4
#SBATCH --cpus-per-task=44
#SBATCH --mem=360G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_outputs/slurm-%x-%j.out

# zmax-left v6 — the v5 recipe at the measured throughput SOTA layout.
# From r_train_zmaxleft_v3.slurm; THROUGHPUT-ONLY deltas (RL dynamics identical
# to v5, checkpoints interchangeable, warm-start from zmaxleft-v5 ckpt_490):
#   - configs/native_zmaxleft_v6.yaml: CUDA graphs on the frozen ViT
#     (bitwise-identical to eager, parity 0.00e+00).
#   - ONE training's 24 runners spread over 4 GPUs (--gres=4, GPU_FRAC 0.166
#     => Ray packs exactly 6 runners per GPU). Requires the ngllib EGL fix
#     (neurogym zmax-left >= 2b5e97c) so each runner's GL context follows its
#     assigned GPU; before it every GL context sat on physical GPU0 and extra
#     GPUs bought nothing. Measured (job 998146, per-iter medians): 1 GPU 218,
#     4 GPU 269 sps (fast iters ~285); 8 GPUs add nothing (host-bound), so 4.
#   - 48 envs cost ~95 GB at start growing toward ~300 GB at full chunk-LRU
#     cap (~10 handles x 256 MB per fetch worker + runner); 360G holds ONE
#     training. Do NOT co-tenant a second training on the node at this LRU.
#
# Usage: sbatch native/r_train_zmaxleft_v6.slurm [run-name] [target-iters]
#   smoke:  USE_MPS=0 BATCH=4000 ITERS=6 WANDB_MODE=disabled \
#           sbatch native/r_train_zmaxleft_v6.slurm zmaxleft-v6-smoke0 6
set -eu
WT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$WT"
RUN=${1:-zmaxleft-v6}
ITERS=${2:-750}
CKPT=/scratch/kp0374/checkpoints/${RUN}
mkdir -p "$CKPT"

export PYTHONUNBUFFERED=1
export RAY_NUM_CPUS=${RAY_CPUS:-44}
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NGL_NATIVE_FETCH_WORKERS=${FETCH_WORKERS:-2}
export NGL_NATIVE_MESH_LRU_MB=128
export NGL_NATIVE_CHUNK_LRU_MB=${CHUNK_LRU_MB:-256}
export NGL_NATIVE_VAO_LRU_MB=${VAO_LRU_MB:-128}
export NGL_NATIVE_MESH_PREFETCH=${MESH_PREFETCH:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# One capture at a time across the 24 runners (MPS capture race otherwise).
export NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-${SLURM_JOB_ID}.lock
PANE_MODE=${PANE_MODE:-atomic}
RUNNERS=${RUNNERS:-24}
ENVS_PER_RUNNER=${ENVS_PER_RUNNER:-2}
GPU_FRAC=${GPU_FRAC:-0.166}   # 24 x 0.166 -> 6 runners per GPU on 4 GPUs
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
  --config ${TRAIN_CONFIG:-configs/native_zmaxleft_v6.yaml} \
  --run-name "$RUN" --pane-mode "$PANE_MODE" \
  --num-env-runners "$RUNNERS" \
  --num-envs-per-env-runner "$ENVS_PER_RUNNER" \
  --num-gpus-per-env-runner "$GPU_FRAC" \
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
