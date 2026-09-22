#!/bin/bash
#SBATCH --job-name=gl-init-probe
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=44
#SBATCH --mem=300G
#SBATCH --time=00:20:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/gl-init-probe-%j.out
#
# Diagnose the GL-context-init hang at N=36 (one GL+CUDA context per runner). py-spy
# dumps the STUCK runners to reveal the exact blocking call (EGL/moderngl create vs
# a mutex vs MPS/CUDA). See gl_init_probe.sh.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0
export NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J.lock
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[gl-init-probe] mps up on $(hostname) job=$J"

bash scripts/gl_init_probe.sh

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[gl-init-probe] done"
