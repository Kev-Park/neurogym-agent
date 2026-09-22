#!/bin/bash
#SBATCH --job-name=dino-graphrel
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-graphrel-%j.out
#
# Test flock-serialized + extra-warmup CUDA-graph capture at N=24 (the regime where
# 8/16 captured before). Success = "captured CUDA graph" count ~= 24 and "capture
# failed" ~= 0. Then does freeing the GIL from the DINO forward dispatch lift SPS
# above the ~340 M=2 baseline? (graph attacks the 78%-of-GIL torch dispatch.)
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0 ITERS=10
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
export NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J.lock
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[graphrel] mps up on $(hostname) job=$J"

bash scripts/dino_lever_arm.sh graph 48 24 2 > "slurm_outputs/dino-graphrel-arm-$J.out" 2>&1

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[graphrel] captured OK count:"; grep -c 'captured CUDA graph' "slurm_outputs/dino-graphrel-arm-$J.out" || true
echo "[graphrel] capture FAILED count:"; grep -c 'CUDA-graph capture failed' "slurm_outputs/dino-graphrel-arm-$J.out" || true
echo "[graphrel] SPS (iter lines):"; grep -hE '^iter ' "slurm_outputs/dino-graphrel-arm-$J.out" | grep -oE 'iter [0-9]+|sps=[0-9.]+|H=[0-9]+/[0-9]+'
