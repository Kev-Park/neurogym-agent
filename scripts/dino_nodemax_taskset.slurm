#!/bin/bash
#SBATCH --job-name=dino-ts
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
#SBATCH --output=slurm_outputs/dino-ts-%j.out
#
# CORRECTED variance A/B: prior numactl pin was misaligned (numactl node cores !=
# nvidia-smi GPU CPU-affinity). Pin via TASKSET to the real per-GPU affinity
# (GPU0-3=0-15,32-47; GPU4-7=16-31,48-63) + membind. G=8/N=8, ITERS=10. Arm A
# unpinned, Arm B taskset-pinned. Goal: uniform (low CV) AND full throughput (the
# misaligned numactl pin gave uniform-but-~10%-low; correct locality should recover it).
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0 ITERS=10
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_CHUNK_LRU_MB=140 NGL_NATIVE_VAO_LRU_MB=140
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[dino-ts] mps up on $(hostname) job=$J cores=$(nproc) gpus=$(nvidia-smi -L | wc -l)"

bash scripts/node_max_arm_taskset.sh 8 8 0     # unpinned
sleep 15
bash scripts/node_max_arm_taskset.sh 8 8 1     # taskset-pinned to real GPU affinity

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[dino-ts] done"
