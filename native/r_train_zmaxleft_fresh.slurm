#!/bin/bash
#SBATCH --job-name=zmaxleft-fresh
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_outputs/slurm-%x-%j.out

# zmax-left FRESH training — pane_mode=fresh: zero staleness (every observe
# blocks on exact fine tiles + full lod0 meshes; the frame is always the
# NG-settled state). Blocking turns envs wait-heavy, so the topology trades
# GPU density for WAIT HIDING (measured: M=2 gave 3.2x under wait-heavy random
# actions vs +17% when waits are hidden by the policy):
#   4 GPUs x 10 runners, M=4 envs/runner (160 envs), FW=3 fetch workers.
#   RAM 40*(1+0.6*3+0.45*4)=184G; 160 procs on 64 threads (2.5/thr);
#   GPU frac 0.1/runner; batch 16000 = 100-step fragments (validated range).
#
# Usage: sbatch native/r_train_zmaxleft_fresh.slurm [run-name] [target-iters]
set -eu
WT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$WT"
RUN=${1:-zmaxleft-fresh-v1}
ITERS=${2:-200}
CKPT=/scratch/kp0374/checkpoints/${RUN}
mkdir -p "$CKPT"

export PYTHONUNBUFFERED=1
export RAY_NUM_CPUS=64
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NGL_NATIVE_FETCH_WORKERS=${FETCH_WORKERS:-3}
export NGL_NATIVE_MESH_LRU_MB=128
export NGL_NATIVE_CHUNK_LRU_MB=${CHUNK_LRU_MB:-128}
export NGL_NATIVE_VAO_LRU_MB=${VAO_LRU_MB:-128}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUNNERS=${RUNNERS:-40}
ENVS_PER_RUNNER=${ENVS_PER_RUNNER:-4}
GPU_FRAC=${GPU_FRAC:-0.1}
BATCH=${BATCH:-16000}
WANDB_MODE=${WANDB_MODE:-online}
export CURRICULUM_PROGRESS_FILE="$CKPT/meta.json"

# One per-user MPS daemon serves all the node's GPUs.
USE_MPS=${USE_MPS:-1}
if [ "$USE_MPS" = "1" ]; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${SLURM_JOB_ID}
  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-${SLURM_JOB_ID}
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d && echo "[mps] daemon up" || echo "[mps] daemon FAILED (continuing without)"
fi

uv run --no-sync python -m ngllib_agent.train \
  --config ${TRAIN_CONFIG:-configs/native_zmaxleft.yaml} \
  --run-name "$RUN" --pane-mode fresh \
  --num-env-runners "$RUNNERS" \
  --num-envs-per-env-runner "$ENVS_PER_RUNNER" \
  --num-gpus-per-env-runner "$GPU_FRAC" \
  --vector threads \
  --sample-timeout-s 600 \
  --iters "$ITERS" \
  --train-batch-size "$BATCH" \
  --checkpoint-dir "$CKPT" --checkpoint-every 10 \
  --wandb-project neurogym-agent --wandb-mode "$WANDB_MODE" \
  --resume

if [ "$USE_MPS" = "1" ]; then
  echo quit | nvidia-cuda-mps-control || true
fi
