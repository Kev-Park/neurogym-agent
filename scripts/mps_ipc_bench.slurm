#!/bin/bash
#SBATCH --job-name=mps-ipc
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
# COMBINED lever test: MPS + both-panes CUDA-IPC (config D under an MPS daemon).
# The two throughput levers were measured independently (B: MPS on the per-process
# numpy path = +48%; D: server + CUDA-IPC = +31%). This stacks them: the DINO
# server + both GL panes staying in VRAM, run UNDER a per-job nvidia-cuda-mps
# daemon so the 32 runner interop CUDA contexts + the server context can overlap
# kernels instead of driver-time-slicing. Question: does MPS's concurrency lift
# the CUDA-IPC/server path the way it lifted the plain per-process path, or is the
# server's single-context batched forward already the ceiling? Compare vs D (~227)
# and B (~256). Same 32x2 / BATCH 8000 / good-node conditions as the matrix.
set -u
cd /scratch/kp0374/wt/neurogym-agent-dino-server
export PYTHONUNBUFFERED=1

RUNNERS=${RUNNERS:-32}
ENVS=${ENVS:-2}
BATCH=${BATCH:-8000}
ITERS=${ITERS:-20}
GPU_PER_RUNNER=${GPU_PER_RUNNER:-0.02}
CPUS_PER_RUNNER=${CPUS_PER_RUNNER:-1.0}
# M: DINO-server instance count override (bottleneck discrimination — the E
# scaling sweep plateaued at ~245 sps; if M=2 breaks the plateau the single
# server context is the cap, else it's aggregate GPU capacity).
DINO_ARGS=""
[ -n "${M:-}" ] && DINO_ARGS="--dino-server-instances $M"

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

echo "[mps] === RUN: both-panes CUDA-IPC (server) UNDER MPS ==="
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python -m ngllib_agent.train \
  --config configs/native_dinoserver_ipc_bp.yaml \
  --run-name "mps-ipc-${SLURM_JOB_ID}" \
  --no-spawn-curriculum --learner-gpu \
  --num-env-runners "$RUNNERS" --num-envs-per-env-runner "$ENVS" \
  --num-gpus-per-env-runner "$GPU_PER_RUNNER" --num-cpus-per-env-runner "$CPUS_PER_RUNNER" \
  --vector threads --iters "$ITERS" --train-batch-size "$BATCH" \
  --checkpoint-every 999999 --no-degraded-exit --wandb-mode disabled \
  $DINO_ARGS

# --- Stop the daemon ---
echo quit | nvidia-cuda-mps-control 2>&1 | sed 's/^/[mps] /' || true
echo "[mps] done"
