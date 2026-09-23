#!/bin/bash
#SBATCH --job-name=dino-giant
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=340G
#SBATCH --time=00:50:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-giant-%j.out
#
# Q2: the EXTREME config -- ONE runner (N=1) = one GL + one CUDA context for the whole
# GPU, with the biggest DINO+render batch that fits (M=48, then M=96). Sidesteps the
# MPS ~48-context wall (1 context) and maximizes forward-pass efficiency (D_cap ~982
# img/s at big batch). Graphs on. Prediction: host-GIL-bound (1 GIL feeds all M envs'
# Python glue serially) -> likely << 360 despite giant-batch GPU efficiency, but this
# measures whether the tradeoff nets out. Generous RAM (all envs' caches in 1 proc).
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0 ITERS=6
export NGL_NATIVE_CHUNK_LRU_MB=500 NGL_NATIVE_VAO_LRU_MB=384
export NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J.lock
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[giant] mps up on $(hostname) job=$J"

run() {  # $1=M (N=1)
  echo "[giant] === N=1 M=$1 (batch=$1) start $(date +%H:%M:%S) ==="
  bash scripts/dino_lever_arm.sh baseline 48 1 "$1" > "slurm_outputs/dino-giant-m$1-$J.out" 2>&1
  echo "[giant] === N=1 M=$1 done exit=$? ==="
  grep -hE '^iter ' "slurm_outputs/dino-giant-m$1-$J.out" 2>/dev/null | grep -oE 'sps=[0-9.]+' | tail -6
  sleep 12
}
run 48
run 96

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[giant] done"
