#!/bin/bash
#SBATCH --job-name=dino-ab-seq
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=140G
#SBATCH --time=00:50:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-ab-seq-%j.out
#
# DINO-lever A/B, CLEAN design: the 3 arms run SEQUENTIALLY on ONE GPU (same
# node, same hardware, back-to-back, MPS) -- no inter-arm interference and no
# 3-job-process-tree spawn storm (which tripped the Py3.12 resource_tracker race
# when the arms ran as concurrent jobs). Each arm's train stdout -> its own file
# so iters are cleanly attributable. Single GPU keeps it schedulable.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[dino-ab-seq] mps up on $(hostname) job=$J"

for ARM in baseline server graph; do
  echo "[dino-ab-seq] === arm=$ARM start $(date +%H:%M:%S) ==="
  bash scripts/dino_lever_arm.sh "$ARM" > "slurm_outputs/dino-seq-$ARM-$J.out" 2>&1
  echo "[dino-ab-seq] === arm=$ARM done exit=$? ==="
  sleep 15   # let Ray + GPU memory fully release before the next arm
done

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[dino-ab-seq] all arms done; per-arm steady sps in slurm_outputs/dino-seq-*-$J.out"
