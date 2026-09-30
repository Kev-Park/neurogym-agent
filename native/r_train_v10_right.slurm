#!/bin/bash
#SBATCH --job-name=v10-right
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=44
#SBATCH --mem=250G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_outputs/slurm-%x-%j.out

# v10: right-pane z-navigate (native.yaml recipe, v9-test action space) with
# the patch-token spatial policy -- configs/native_v10_right.yaml.
# Layout = the right-pane single-mesh throughput SOTA (THROUGHPUT_SCALING.md):
# ONE GPU, CUDA graphs, 24 runners x 8 threaded envs = 192 envs under MPS
# (~380 sps; 32 runners degrades, runners die). Batch 4000 x 740 iters =
# 2.96M steps, step-identical to native-v9-val / pace740, so the eval_d0
# holdout comparison is like for like. Learner on the GPU (conv trunk).
# RAM: 24 runners x (1 + 2 fetch workers) = 72 procs x ~10 handles x 256 MB
# chunk LRU = ~184 GB worst case -> 250G.
#
# Usage: sbatch native/r_train_v10_right.slurm [run-name] [target-iters]
set -eu
WT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$WT"
RUN=${1:-v10-right}
ITERS=${2:-740}
CKPT=/scratch/kp0374/checkpoints/${RUN}
mkdir -p "$CKPT"

export PYTHONUNBUFFERED=1
export RAY_NUM_CPUS=${RAY_CPUS:-44}
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NGL_NATIVE_FETCH_WORKERS=${FETCH_WORKERS:-2}
export NGL_NATIVE_MESH_LRU_MB=128
export NGL_NATIVE_CHUNK_LRU_MB=${CHUNK_LRU_MB:-256}
export NGL_NATIVE_VAO_LRU_MB=${VAO_LRU_MB:-256}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# One CUDA-graph capture at a time across the runners (MPS capture race).
export NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-${SLURM_JOB_ID}.lock
PANE_MODE=${PANE_MODE:-atomic}
RUNNERS=${RUNNERS:-24}
ENVS_PER_RUNNER=${ENVS_PER_RUNNER:-8}
GPU_FRAC=${GPU_FRAC:-0.04}     # 24 x 0.04 -> all on the one GPU
BATCH=${BATCH:-4000}
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
  --config ${TRAIN_CONFIG:-configs/native_v10_right.yaml} \
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
  --learner-gpu \
  --resume

if [ "$USE_MPS" = "1" ]; then
  echo quit | nvidia-cuda-mps-control || true
fi
