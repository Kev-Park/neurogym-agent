#!/bin/bash
#SBATCH --job-name=mgpu-base
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:3
#SBATCH --cpus-per-task=44
#SBATCH --mem=320G
#SBATCH --time=00:45:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/disagg-mgpu-base-%j.out
#
# Multi-GPU-per-node BASELINE: G independent trainings (one per GPU), LOCAL process
# fetch, M=2 (recipe-stable, avoids the resource_tracker race). Aggregate sps = sum
# of the per-GPU logs -- this is the fetch-CPU-bound regime (the ~600-sps ceiling) the
# socket-disagg arm tries to beat by moving fetch to the CPU node's cores.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_memory_monitor_refresh_ms=0
export NGL_NATIVE_FETCH_BACKEND=process NGL_NATIVE_FETCH_WORKERS=${FW:-2}
export NGL_NATIVE_CHUNK_LRU_MB=120 NGL_NATIVE_VAO_LRU_MB=128 NGL_RENDER_BATCH_SIZE=${M:-2}
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
export MGPU_TAG=mgpu-base
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d && echo "[mgpu-base] mps up on $(hostname)"

bash scripts/launch_mgpu_trainings.sh

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[mgpu-base] done ; per-GPU steady sps below"
grep -hE '^iter (8|9|1[0-2])' slurm_outputs/mgpu-base-$J-g*.out 2>/dev/null | grep -oE 'sps=[0-9.]+' || true
