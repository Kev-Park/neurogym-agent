#!/bin/bash
#SBATCH --job-name=dino-ab-base
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=00:20:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-ab-base-%j.out
# DINO-lever A/B, arm A: in-process DINO per runner (SOTA). Single GPU, 24x2, MPS.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[dino-ab-base] mps up on $(hostname)"
bash scripts/dino_lever_arm.sh baseline
echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
