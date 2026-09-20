#!/bin/bash
#SBATCH --job-name=rb-util
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/rb-util-%j.out
#
# Instrumented run of the FASTEST STABLE config (in-process DINO + batched render
# + interop, 24 runners x 8 envs, MPS) with GPU + CPU utilization sampled during
# training, to find the per-step bottleneck (ignoring the PPO barrier).
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
export NGL_RENDER_BATCH_SIZE=8
export NGL_NATIVE_FETCH_WORKERS=2 NGL_NATIVE_VAO_LRU_MB=128 NGL_NATIVE_CHUNK_LRU_MB=192

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d && echo "[mps] up"

# Samplers: GPU SM/mem util + per-core CPU, every 3s, to side logs.
nvidia-smi dmon -s u -d 3 -o T > slurm_outputs/util-gpu-$J.log 2>&1 &
GPUMON=$!
( command -v mpstat >/dev/null && mpstat 3 > slurm_outputs/util-cpu-$J.log 2>&1 \
  || vmstat 3 > slurm_outputs/util-cpu-$J.log 2>&1 ) &
CPUMON=$!

UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python -m ngllib_agent.train \
  --config configs/native_rb_interop.yaml --run-name rb-util-$J \
  --no-spawn-curriculum --learner-gpu \
  --num-env-runners 24 --num-envs-per-env-runner 8 \
  --num-gpus-per-env-runner 0.02 --num-cpus-per-env-runner 1.0 \
  --vector threads --iters 15 --train-batch-size 24000 \
  --checkpoint-every 999999 --no-degraded-exit --wandb-mode disabled

kill $GPUMON $CPUMON 2>/dev/null || true
echo quit | nvidia-cuda-mps-control 2>&1 | sed 's/^/[mps] /' || true
echo "[util] done — GPU: util-gpu-$J.log  CPU: util-cpu-$J.log"
