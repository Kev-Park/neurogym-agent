#!/bin/bash
#SBATCH --job-name=dino-g8
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:3090:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=00:50:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-g8-%j.out
#
# The missing point: G=8 (all GPUs) -- both runs cut it. Aggregate rose G3=930 ->
# G4=960 -> G6=~1300 (more GPUs = fewer runners/GPU = less contention). Does the 8th
# GPU keep it rising or plateau/degrade from 8-way node contention? Two densities:
# N=8 (64 runners, full core budget) and N=6 (48 runners, thinnest/GPU). ITERS=8.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0 ITERS=8
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_CHUNK_LRU_MB=140 NGL_NATIVE_VAO_LRU_MB=140
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[g8] mps up on $(hostname) job=$J cores=$(nproc) gpus=$(nvidia-smi -L | wc -l)"

bash scripts/node_max_arm.sh 8 8      # 64 runners
bash scripts/node_max_arm.sh 8 6      # 48 runners (thinnest per GPU)

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[g8] all arms done"
