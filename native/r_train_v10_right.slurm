#!/bin/bash
#SBATCH --job-name=v10-right
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:2
#SBATCH --cpus-per-task=44
#SBATCH --mem=250G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_outputs/slurm-%x-%j.out

# v10: right-pane z-navigate (native.yaml recipe, v9-test action space) with
# the patch-token spatial policy -- configs/native_v10_right.yaml.
# Layout: 32 runners x 2 threaded envs (r_train_native.slurm's production
# topology) spread over TWO GPUs (16 runners/GPU, needs the ngllib EGL fix),
# CUDA graphs, MPS. NOT the throughput doc's 24x8=192-env right-pane peak:
# that used the throughput branch's BATCHED renderer (one GL context per
# runner); on zmax-left every env owns a GL context and 192 of them on one
# 3090 OOM'd the card (job 998190, 15/24 runners died). Batch 4000 x 740
# iters = 2.96M steps, step-identical to native-v9-val / pace740, so the
# eval_d0 holdout comparison is like for like. Learner on the GPU (conv
# trunk). RAM: 32 runners x (1 + 2 fetch workers) = 96 procs x ~10 handles
# x 192 MB chunk LRU = ~184 GB worst case -> 250G.
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
export NGL_NATIVE_CHUNK_LRU_MB=${CHUNK_LRU_MB:-192}
export NGL_NATIVE_VAO_LRU_MB=${VAO_LRU_MB:-256}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# One CUDA-graph capture at a time across the runners (MPS capture race).
export NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-${SLURM_JOB_ID}.lock
PANE_MODE=${PANE_MODE:-atomic}
RUNNERS=${RUNNERS:-32}
ENVS_PER_RUNNER=${ENVS_PER_RUNNER:-2}
GPU_FRAC=${GPU_FRAC:-0.06}     # 32 x 0.06 = 1.92 -> Ray packs 16 runners per GPU
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

# train.py exits 43 on sustained degradation (it checkpoints first; the cure
# is a fresh process resuming from that checkpoint -- job 998204 ended this
# way at iter 143 after tile-fetch stalls). Relaunch in place until it ends
# any other way; train.py itself disables the exit after 3 in 2h.
while :; do
  set +e
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
  rc=$?
  set -e
  [ "$rc" -eq 43 ] || break
  echo "[launcher] degraded exit (43) -> relaunching from checkpoint"
done

if [ "$USE_MPS" = "1" ]; then
  echo quit | nvidia-cuda-mps-control || true
fi
