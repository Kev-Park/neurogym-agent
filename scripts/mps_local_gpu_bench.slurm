#!/bin/bash
#SBATCH --job-name=mps-local
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/sps-bench-%x-%j.out
#
# "Best of both" cell (F): per-process DINO + on-GPU feed (no server, no IPC),
# run UNDER an MPS daemon. Each runner keeps both GL panes in VRAM and feeds its
# OWN in-process DINO (encode_gpu) -- no readback (like D), 32 separate forwards
# that MPS runs concurrently (like B), minus the server/RPC/IPC overhead E carries.
# Hypothesis: should beat both B (~256) and E (~243). Compare also vs the no-MPS
# version of this same config (F0, native_local_gpu_bp via bench_single_node_sps).
# Same 32x2 / BATCH 8000 / good-node conditions as the rest of the matrix.
set -u
cd /scratch/kp0374/wt/neurogym-agent-dino-server
export PYTHONUNBUFFERED=1

RUNNERS=${RUNNERS:-32}
ENVS=${ENVS:-2}
BATCH=${BATCH:-8000}
ITERS=${ITERS:-20}

export NGL_NATIVE_FETCH_WORKERS="${NGL_NATIVE_FETCH_WORKERS:-2}"
export NGL_NATIVE_MESH_LRU_MB="${NGL_NATIVE_MESH_LRU_MB:-128}"
export NGL_NATIVE_CHUNK_LRU_MB="${NGL_NATIVE_CHUNK_LRU_MB:-256}"
export NGL_NATIVE_VAO_LRU_MB="${NGL_NATIVE_VAO_LRU_MB:-256}"

# --- Start a per-user CUDA MPS daemon, scoped to this job (exported BEFORE ray
# starts so the driver, the DINO-server actor, and every env runner attach to it).
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${SLURM_JOB_ID}
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-${SLURM_JOB_ID}
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
echo "[mps] node=$(hostname) starting nvidia-cuda-mps-control -d"
nvidia-cuda-mps-control -d && echo "[mps] daemon start rc=$?" || echo "[mps] daemon FAILED to start (continuing without MPS)"
sleep 3
echo get_server_list | nvidia-cuda-mps-control 2>&1 | sed 's/^/[mps] /' || true

echo "[mps] === RUN: both-panes per-process on-GPU (no server) UNDER MPS ==="
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python -m ngllib_agent.train \
  --config configs/native_local_gpu_bp.yaml \
  --run-name "mps-local-${SLURM_JOB_ID}" \
  --no-spawn-curriculum --learner-gpu \
  --num-env-runners "$RUNNERS" --num-envs-per-env-runner "$ENVS" \
  --num-gpus-per-env-runner 0.02 --num-cpus-per-env-runner 1.2 \
  --vector threads --iters "$ITERS" --train-batch-size "$BATCH" \
  --checkpoint-every 999999 --no-degraded-exit --wandb-mode disabled

# --- Stop the daemon ---
echo quit | nvidia-cuda-mps-control 2>&1 | sed 's/^/[mps] /' || true
echo "[mps] done"
