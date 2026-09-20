#!/bin/bash
#SBATCH --job-name=cpu-attr
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=00:25:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/cpu-attr-%j.out
#
# Attribute the node-CPU wall for one full-density single-GPU 24x8 interop run:
# run the training in the background, wait for steady state, then sample
# per-process CPU (cpu_attribution.sh) + mpstat. Answers whether fetch/decode
# (spawn pool) or the Ray env-runner actors (render/step) own the CPU.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
echo "[attr] node=$(hostname) cpus=$SLURM_CPUS_PER_TASK"

export NGL_RENDER_BATCH_SIZE=8
export NGL_NATIVE_FETCH_WORKERS=6 NGL_NATIVE_VAO_LRU_MB=128 NGL_NATIVE_CHUNK_LRU_MB=160

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d && echo "[attr] mps up"

CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
  uv run --no-sync python -m ngllib_agent.train \
    --config configs/native_rb_interop.yaml --run-name cpuattr-$J \
    --no-spawn-curriculum --learner-gpu \
    --num-env-runners 24 --num-envs-per-env-runner 8 \
    --num-gpus-per-env-runner 0.02 --num-cpus-per-env-runner 1.0 \
    --vector threads --iters 12 --train-batch-size 24000 \
    --checkpoint-every 999999 --no-degraded-exit --wandb-mode disabled \
    > slurm_outputs/cpuattr-$J.out 2>&1 &
TRAIN=$!

# Wait for steady state (a few iters in), up to ~10 min.
for i in $(seq 1 60); do
  grep -q '^iter 3' slurm_outputs/cpuattr-$J.out 2>/dev/null && { echo "[attr] steady (iter 3)"; break; }
  kill -0 $TRAIN 2>/dev/null || { echo "[attr] training exited early"; break; }
  sleep 10
done

echo "=== mpstat (node aggregate, 3x5s) ==="
mpstat 5 3 || vmstat 5 3
echo "=== per-process CPU attribution (3 windows) ==="
for k in 1 2 3; do
  echo "--- window $k ---"
  bash scripts/cpu_attribution.sh 8
done

# Also capture the training's own reported sps around the sampling window.
echo "=== training iters ==="
grep '^iter' slurm_outputs/cpuattr-$J.out | tail -4

kill $TRAIN 2>/dev/null || true
echo quit | nvidia-cuda-mps-control 2>&1 | sed 's/^/[attr] /' || true
echo "[attr] done"
