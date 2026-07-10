#!/bin/bash
#SBATCH --job-name=r4-mgpu
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=250G
#SBATCH --gres=gpu:3090:2            # exactly TWO 3090s (no --exclusive: keeps
                                     # CUDA_VISIBLE_DEVICES = just our 2)
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

echo "=== node=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (expect 2) ==="
GPU_LOG=slurm_outputs/r4-mgpu-$SLURM_JOB_ID.gpu.log
# Sample our 2 GPUs, one CSV row per (time,gpu): "HH:MM:SS gpuUUID util mem"
( while true; do
    nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader \
      -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null | while read -r line; do
        echo "$(date +%H:%M:%S) $line" >> "$GPU_LOG"; done
    sleep 5
  done ) &
GPU_PID=$!

FULL=slurm_outputs/r4-mgpu-$SLURM_JOB_ID.full.log
uv run --no-sync python scripts/ppo_smoke.py \
    --obs dino --num-env-runners 2 --num-gpus-per-env-runner 1 \
    --num-envs-per-env-runner 16 --vector threads \
    --iters 6 --train-batch-size 4096 --rollout-fragment-length 128 \
  2>&1 | tee "$FULL" | grep -E "iter [0-9]+:|Traceback|complete" | tail -14

kill $GPU_PID 2>/dev/null || true
echo "########## PER-GPU (max util%, max mem MiB) — split? ##########"
# group by GPU uuid (col 2); the 2 uuids are our 2 physical GPUs
awk '{u=$2; gsub(/%/,"",$3); m=$4;
      if ($3+0>um[u]) um[u]=$3+0; if (m+0>mm[u]) mm[u]=m+0}
     END{i=0; for (k in um) printf "GPU#%d(%s): util_max=%d%% mem_max=%dMiB\n", i++, substr(k,1,12), um[k], mm[k]}' "$GPU_LOG"
echo "########## THROUGHPUT ##########"
grep -E "iter [0-9]+:" "$FULL" | grep -oE "steps_per_s=[0-9.]+" | tr '\n' ' '; echo
echo "glitches=$(grep -c 'env glitch' "$FULL")  restarts=$(grep -c 'browser restart\|watchdog killed' "$FULL")"
echo "=== r4-mgpu done: PASS if BOTH GPUs util>10% and aggregate sps ~2x single-GPU (~40+) ==="
