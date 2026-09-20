#!/bin/bash
#SBATCH --job-name=rb-mgpu
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --time=00:40:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/rb-mgpu-%j.out
#
# Multi-GPU-per-NODE throughput test: launch N INDEPENDENT 24x8 in-process-interop
# trainings, one pinned to each visible GPU, sharing the node's CPU/RAM. Measures
# how many GPUs/node we can drive before CPU/RAM saturate or per-GPU SPS degrades
# (aggregate SPS = sum of the N jobs). Density fixed at 24x8 (goal is sps, not
# efficiency). Override GPU count at submit: sbatch --gres=gpu:3090:4 [--mem=...].
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
N=$(nvidia-smi -L | wc -l)
echo "[mgpu] node=$(hostname) GPUs=$N cpus=$SLURM_CPUS_PER_TASK"

export NGL_RENDER_BATCH_SIZE=8
export NGL_NATIVE_FETCH_WORKERS=2 NGL_NATIVE_VAO_LRU_MB=128 NGL_NATIVE_CHUNK_LRU_MB=160

# One MPS daemon (per-GPU servers spawned on demand; each client scopes via
# CUDA_VISIBLE_DEVICES).
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d && echo "[mgpu] mps up"

# Node CPU + RAM samplers.
( command -v mpstat >/dev/null && mpstat 5 || vmstat 5 ) > slurm_outputs/mg-cpu-$J.log 2>&1 &
CPUMON=$!
free -m -s 5 > slurm_outputs/mg-ram-$J.log 2>&1 &
RAMMON=$!

pids=""
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
    uv run --no-sync python -m ngllib_agent.train \
      --config configs/native_rb_interop.yaml --run-name mg-$J-g$i \
      --no-spawn-curriculum --learner-gpu \
      --num-env-runners 24 --num-envs-per-env-runner 8 \
      --num-gpus-per-env-runner 0.02 --num-cpus-per-env-runner 1.0 \
      --vector threads --iters 15 --train-batch-size 24000 \
      --checkpoint-every 999999 --no-degraded-exit --wandb-mode disabled \
      > slurm_outputs/mg-$J-g$i.out 2>&1 &
  pids="$pids $!"
  echo "[mgpu] launched GPU $i pid $!"
done

wait $pids
kill $CPUMON $RAMMON 2>/dev/null || true
echo quit | nvidia-cuda-mps-control 2>&1 | sed 's/^/[mgpu] /' || true
echo "[mgpu] done — per-GPU logs mg-$J-g{0..$((N-1))}.out ; cpu mg-cpu-$J.log ; ram mg-ram-$J.log"
