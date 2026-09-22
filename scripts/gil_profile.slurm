#!/bin/bash
#SBATCH --job-name=gil-profile
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/gil-profile-%j.out
#
# Bottleneck attribution (GIL vs CPU-util vs RLlib). Profiles the GIL-STRESS arm
# (M=16/N=3: 16 threads sharing one GIL per process) and the MANY-PROCESS arm
# (M=2/N=24: 2 threads/GIL, high parallelism) for contrast. Full CPUs so CPU-util
# numbers are meaningful. See gil_profile.sh for what's captured + how to read it.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0
export NGL_NATIVE_CHUNK_LRU_MB=300 NGL_NATIVE_VAO_LRU_MB=256
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[gil-profile] mps up on $(hostname) job=$J cores=$(nproc)"

bash scripts/gil_profile.sh 16 3 stress
sleep 10
bash scripts/gil_profile.sh 2 24 many

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[gil-profile] done; artifacts: slurm_outputs/gil-{mpstat,dumps,flame,flamegil}-{stress,many}-$J.*"
