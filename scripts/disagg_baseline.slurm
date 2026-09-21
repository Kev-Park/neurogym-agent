#!/bin/bash
#SBATCH --job-name=disagg-base
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=22
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/disagg-base-%j.out
#
# Single-node baseline for the disaggregated-fetch A/B: identical 16x8 interop
# density, PROCESS fetch backend (all CloudVolume read+decode on THIS node's cores,
# 48 spawn workers). Compare its SPS to scripts/disagg_fetch_train.slurm, which
# offloads that decode to a 2nd node's cores.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_memory_monitor_refresh_ms=0          # disable Ray OOM killer (cgroup misfire)
export NGL_NATIVE_FETCH_BACKEND=process NGL_NATIVE_FETCH_WORKERS=1
export NGL_NATIVE_CHUNK_LRU_MB=160 NGL_NATIVE_VAO_LRU_MB=128 NGL_RENDER_BATCH_SIZE=8
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d && echo "[base] mps up on $(hostname)"

UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
  uv run --no-sync python -m ngllib_agent.train \
    --config configs/native_rb_interop.yaml --run-name disagg-base-$J \
    --no-spawn-curriculum --learner-gpu \
    --num-env-runners 16 --num-envs-per-env-runner 8 \
    --num-gpus-per-env-runner 0.03 --num-cpus-per-env-runner 1.0 \
    --vector threads --iters 15 --train-batch-size 24000 \
    --checkpoint-every 999999 --no-degraded-exit --wandb-mode disabled

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[base] done"
