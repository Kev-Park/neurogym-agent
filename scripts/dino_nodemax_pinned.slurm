#!/bin/bash
#SBATCH --job-name=dino-pin
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:3090:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=01:35:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-pin-%j.out
#
# VARIANCE MITIGATION A/B (same node = clean comparison): G=8/N=8, ITERS=10 (longer,
# ~7 warm iters for mean+std). Arm A unpinned (baseline, the noisy ~1300-1600 with
# per-GPU 81-294), Arm B NUMA/CPU-pinned (numactl cpunodebind+membind per GPU's socket
# -> no cross-socket bounce/remote memory). Does pinning shrink the per-GPU variance
# AND stabilize/raise the mean? Longer iters average out per-iter noise.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0 ITERS=10
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_CHUNK_LRU_MB=140 NGL_NATIVE_VAO_LRU_MB=140
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[dino-pin] mps up on $(hostname) job=$J cores=$(nproc) gpus=$(nvidia-smi -L | wc -l)"
echo "[dino-pin] numa layout:"; numactl --hardware 2>/dev/null | grep -E 'available|node [0-9]+ cpus' | head

bash scripts/node_max_arm_pinned.sh 8 8 0     # unpinned baseline
sleep 15
bash scripts/node_max_arm_pinned.sh 8 8 1     # NUMA/CPU-pinned

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[dino-pin] done"
