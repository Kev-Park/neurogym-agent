#!/bin/bash
#SBATCH --job-name=dino-nrun
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=140G
#SBATCH --time=01:10:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-nrun-%j.out
#
# WHY don't the DINO levers help? -> map the per-GPU SM-saturation curve.
# baseline (in-process DINO) at NRUN in {8,16,24}: if sps plateaus by 24, the GPU
# is compute-saturated under MPS packing, so batch-consolidation/CUDA-graphs (which
# only fill a lone process's idle SM gaps) are absorbed -> can't raise aggregate sps.
# Plus one server arm at NRUN=24 with DINO-SERVER-STATS to confirm the big-batch
# test actually achieved a big mean_batch. Sequential on ONE GPU (clean, MPS, highpri).
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0 ITERS=6
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[dino-nrun] mps up on $(hostname) job=$J"

for NR in 8 16 24; do
  echo "[dino-nrun] === baseline NRUN=$NR start $(date +%H:%M:%S) ==="
  bash scripts/dino_lever_arm.sh baseline 48 "$NR" > "slurm_outputs/dino-nrun-base-n$NR-$J.out" 2>&1
  echo "[dino-nrun] === baseline NRUN=$NR done exit=$? ==="
  sleep 12
done

echo "[dino-nrun] === server NRUN=24 (mean_batch readout) start $(date +%H:%M:%S) ==="
bash scripts/dino_lever_arm.sh server 48 24 > "slurm_outputs/dino-nrun-srv-n24-$J.out" 2>&1
echo "[dino-nrun] === server done exit=$? ==="

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[dino-nrun] all done; sps per arm in slurm_outputs/dino-nrun-*-$J.out"
