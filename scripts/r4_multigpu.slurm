#!/bin/bash
#SBATCH --job-name=r4-mgpu
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive                  # whole node — no co-tenant to muddy per-GPU reads
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --gres=gpu:3090:2            # TWO 3090s on one node
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-5,sarekl15-7,sarekl15-8,sarekl16-1,sarekl16-3,sarekl16-5,sarekl16-6,sarekl16-7,sarekl16-8
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4
#SBATCH --time=00:40:00
#SBATCH --output=slurm_outputs/r4-mgpu-%j.out

# REFINEMENT R4 extension: does rendering scale across 2 GPUs/node?
# 2 env-runners x 1 GPU each x M=16 = 32 browsers. Ray sets CUDA_VISIBLE_DEVICES
# per runner; DINO (torch) will split for sure — the question is whether Chrome's
# VULKAN rendering follows it. Sample BOTH allocated GPUs to find out.
#   PASS (native 2-GPU): both GPUs utilized + aggregate sps ~2x single-GPU (~50-60).
#   FAIL (renderer pins to GPU0): GPU0 pegged, GPU1 light, sps flat (~30).
set -e
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

echo "=== node=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="
GPU_LOG=slurm_outputs/r4-mgpu-$SLURM_JOB_ID.gpu.log
( while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader \
      -i "$CUDA_VISIBLE_DEVICES" | paste -sd'|' | sed "s/^/$(date +%H:%M:%S) /" >> "$GPU_LOG"
    sleep 5
  done ) &
GPU_PID=$!

FULL=slurm_outputs/r4-mgpu-$SLURM_JOB_ID.full.log
uv run --no-sync python scripts/ppo_smoke.py \
    --obs dino --num-env-runners 2 --num-gpus-per-env-runner 1 \
    --num-envs-per-env-runner 16 --vector threads \
    --iters 5 --train-batch-size 4096 --rollout-fragment-length 128 \
  2>&1 | tee "$FULL" | grep -E "iter [0-9]+:|Traceback|complete" | tail -12

kill $GPU_PID 2>/dev/null || true
echo "########## PER-GPU UTILIZATION (during run) ##########"
# max util + max mem per physical GPU index across the whole log
awk -F'[ ,|]+' '
  /%/ {
    for (i=1; i<=NF; i++) if ($i ~ /%$/) {
      idx=$(i-1); gsub(/%/,"",$i);
      if ($i+0 > umax[idx]) umax[idx]=$i+0;
      # memory is the field after "MiB" pair — grab any number followed by MiB
    }
  }
  { for (i=1;i<=NF;i++) if ($i=="MiB") { m=$(i-1)+0; if (m>mmax[pidx]) mmax[pidx]=m } }
  END { for (k in umax) printf "GPU %s: util_max=%d%%\n", k, umax[k] }
' "$GPU_LOG" | sort
echo "--- raw GPU log tail ---"; tail -4 "$GPU_LOG"
echo "########## THROUGHPUT ##########"
grep -E "iter [0-9]+:" "$FULL" | grep -oE "steps_per_s=[0-9.]+" | tr '\n' ' '; echo
echo "=== r4-mgpu done ==="
