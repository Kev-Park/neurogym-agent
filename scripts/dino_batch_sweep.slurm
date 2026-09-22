#!/bin/bash
#SBATCH --job-name=dino-batch
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=220G
#SBATCH --time=01:10:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-batch-%j.out
#
# BATCH-SIZE lever (user's insight): render+DINO batch = M (envs/runner). We've run
# M=2 (inefficient, DINO 363 img/s) everywhere; big batches are efficient (982 at 48).
# Warm LRU caches make fetch cheap, so we can pack more envs into FEWER processes ->
# bigger IN-PROCESS batches (no server IPC). Sweep M at FIXED ~48 envs/GPU. If SPS
# rises with M toward the ~530 host ceiling, big in-process batches beat the M=2 317.
# Generous cache budget (caches concentrate in fewer procs). Sequential, 1 GPU, MPS.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0 ITERS=6
export NGL_NATIVE_CHUNK_LRU_MB=300 NGL_NATIVE_VAO_LRU_MB=256
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[dino-batch] mps up on $(hostname) job=$J"

# (M envs/runner, N runners) at ~48 envs total: batch grows 2->24, procs shrink 24->2.
run() {  # $1=M $2=N
  echo "[dino-batch] === M=$1 N=$2 (batch=$1) start $(date +%H:%M:%S) ==="
  bash scripts/dino_lever_arm.sh baseline 48 "$2" "$1" > "slurm_outputs/dino-batch-m$1-n$2-$J.out" 2>&1
  echo "[dino-batch] === M=$1 N=$2 done exit=$? ==="
  sleep 12
}
run 2 24
run 4 12
run 8 6
run 16 3
run 24 2

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[dino-batch] all done; sps per M in slurm_outputs/dino-batch-m*-$J.out"
