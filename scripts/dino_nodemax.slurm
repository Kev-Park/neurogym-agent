#!/bin/bash
#SBATCH --job-name=dino-nodemax
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:3090:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=01:30:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-nodemax-%j.out
#
# PER-NODE MAX: full node (8x3090, 64 cores, 376G, exclusive), graphs SOTA. Each
# training caps ~360 sps and needs ~24 runners/cores, so 64 cores -> ~2-3 well-fed
# trainings; GPUs (8) are NOT the cap. Sweep (G GPUs, N runners/GPU) at ~total runners
# 48-64 to find the aggregate max and whether >3 GPUs helps (predict: no, CPU-bound
# ~900-1000). All levers maxed: cores 64, RAM 360G, graphs, M=2.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0 ITERS=8
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_CHUNK_LRU_MB=160 NGL_NATIVE_VAO_LRU_MB=160
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[nodemax] mps up on $(hostname) job=$J cores=$(nproc) gpus=$(nvidia-smi -L | wc -l)"

# (G GPUs, N runners/GPU): total runners kept ~48-60 (64-core budget). Does more GPUs
# at fixed core budget beat fewer? Sequential so each has the whole node.
bash scripts/node_max_arm.sh 2 24     # 48 runners
bash scripts/node_max_arm.sh 3 20     # 60 runners
bash scripts/node_max_arm.sh 4 15     # 60 runners
bash scripts/node_max_arm.sh 6 10     # 60 runners
bash scripts/node_max_arm.sh 3 24     # 72 runners (oversubscribe 64 cores -- ref)

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[nodemax] all arms done"
