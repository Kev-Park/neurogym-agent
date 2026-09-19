#!/bin/bash
#SBATCH --job-name=mps-bench
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
#SBATCH --output=slurm_outputs/mps-bench-%x-%j.out
#
# CUDA MPS experiment: run the STANDARD both-panes simulator (per-process DINO,
# numpy readback) under a per-user CUDA MPS daemon, to see whether a shared GPU
# context frees VRAM / lets kernels overlap enough to change training SPS vs the
# ~176 non-MPS baseline (32x2). No code changes — MPS is a runtime/daemon change.
#
# Submit (override partition to highpri if preempt is starved):
#   sbatch [--partition=highpri] scripts/mps_bench.slurm
# Caveat: MPS on a co-tenanted GPU may not help (M=1-vs-M=2 showed the encoder
# isn't the bottleneck), and if the node's GPU isn't in a compatible compute mode
# the daemon may refuse — the script logs that and still runs a NON-MPS pass so
# the comparison is captured either way.
set -u
cd /scratch/kp0374/wt/neurogym-agent-mps
export PYTHONUNBUFFERED=1

RUNNERS=${RUNNERS:-32}
ENVS=${ENVS:-2}
BATCH=${BATCH:-8000}
ITERS=${ITERS:-20}

export NGL_NATIVE_FETCH_WORKERS="${NGL_NATIVE_FETCH_WORKERS:-2}"
export NGL_NATIVE_MESH_LRU_MB="${NGL_NATIVE_MESH_LRU_MB:-128}"
export NGL_NATIVE_CHUNK_LRU_MB="${NGL_NATIVE_CHUNK_LRU_MB:-256}"
export NGL_NATIVE_VAO_LRU_MB="${NGL_NATIVE_VAO_LRU_MB:-256}"

# --- Start a per-user CUDA MPS daemon, scoped to this job ---
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${SLURM_JOB_ID}
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-${SLURM_JOB_ID}
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
echo "[mps] node=$(hostname) starting nvidia-cuda-mps-control -d"
nvidia-cuda-mps-control -d && echo "[mps] daemon start rc=$?" || echo "[mps] daemon FAILED to start (continuing without MPS)"
sleep 3
echo get_server_list | nvidia-cuda-mps-control 2>&1 | sed 's/^/[mps] /' || true

run_bench () {
  UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python -m ngllib_agent.train \
    --config configs/native.yaml --run-name "mps-${SLURM_JOB_ID}" \
    --no-spawn-curriculum --learner-gpu \
    --num-env-runners "$RUNNERS" --num-envs-per-env-runner "$ENVS" \
    --num-gpus-per-env-runner 0.02 --num-cpus-per-env-runner 1.0 \
    --vector threads --iters "$ITERS" --train-batch-size "$BATCH" \
    --checkpoint-every 999999 --no-degraded-exit --wandb-mode disabled
}

echo "[mps] === RUN UNDER MPS ==="
run_bench

# --- Stop the daemon ---
echo quit | nvidia-cuda-mps-control 2>&1 | sed 's/^/[mps] /' || true
echo "[mps] done"
