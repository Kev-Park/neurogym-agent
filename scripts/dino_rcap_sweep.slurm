#!/bin/bash
#SBATCH --job-name=dino-rcap
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=140G
#SBATCH --time=00:45:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-rcap-%j.out
#
# R_cap: render+interop+env pipeline capacity per GPU (DINO forward skipped via
# obs.dino.noop). Sweep NRUN {16,24,32} to find where render saturates. Combined
# with D_cap (~982 img/s) this gives predicted dedicated-DINO disagg sps/GPU =
# R_cap*D_cap/(R_cap+D_cap) -- compare to shared 317. Sequential, 1 GPU, MPS, highpri.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0 ITERS=6
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[dino-rcap] mps up on $(hostname) job=$J"

for NR in 16 24 32; do
  echo "[dino-rcap] === noop NRUN=$NR start $(date +%H:%M:%S) ==="
  bash scripts/dino_lever_arm.sh noop 48 "$NR" > "slurm_outputs/dino-rcap-noop-n$NR-$J.out" 2>&1
  echo "[dino-rcap] === noop NRUN=$NR done exit=$? ==="
  sleep 12
done

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[dino-rcap] all done; R_cap per NRUN in slurm_outputs/dino-rcap-noop-*-$J.out"
