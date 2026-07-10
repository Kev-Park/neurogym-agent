#!/bin/bash
#SBATCH --job-name=r4-scale
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64            # whole node's cores — threads are I/O-bound
#SBATCH --mem=256G                    # node cap ~512G but 400G was rejected under gres
#SBATCH --gres=gpu:3090:4             # up to 4 GPUs; sweep uses 1/2/4 of them
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-5,sarekl15-7,sarekl15-8,sarekl16-1,sarekl16-3,sarekl16-5,sarekl16-6,sarekl16-7,sarekl16-8
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4
#SBATCH --time=03:00:00
#SBATCH --output=slurm_outputs/r4-scale-%j.out

# REFINEMENT R4 frontier: maximize SPS/node over (GPU count x threads/GPU).
# One allocation of 4 GPUs; each config uses R runners (1 GPU each) x M browsers.
# Reports aggregate sps + per-GPU util so we see the saturation frontier.
set -e
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
echo "=== node=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="

GPU_LOG=slurm_outputs/r4-scale-$SLURM_JOB_ID.gpu.log
( while true; do
    echo "$(date +%s) $(nvidia-smi --query-gpu=uuid,utilization.gpu --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES" | tr '\n' ';')" >> "$GPU_LOG"
    sleep 5
  done ) &
GPU_PID=$!

run() {  # R=runners(GPUs)  M=browsers/runner
  local R=$1 M=$2 total=$(( $1 * $2 ))
  echo "########## SCALE R=$R(GPUs) M=$M -> $total browsers ##########"
  T0=$SECONDS
  uv run --no-sync python scripts/ppo_smoke.py \
      --obs dino --num-env-runners "$R" --num-gpus-per-env-runner 1 \
      --num-envs-per-env-runner "$M" --vector threads \
      --iters 6 --train-batch-size $(( 128 * total )) --rollout-fragment-length 128 \
    2>&1 | tee "slurm_outputs/r4-scale-$SLURM_JOB_ID-R${R}M${M}.log" \
    | grep -E "iter [0-9]+:" | grep -oE "steps_per_s=[0-9.]+" | tr '\n' ' '
  echo
  echo "  R=$R M=$M browsers=$total glitches=$(grep -c 'env glitch' slurm_outputs/r4-scale-$SLURM_JOB_ID-R${R}M${M}.log) wall=$((SECONDS-T0))s"
  sleep 5   # let ray fully release GPUs between configs
}

# Frontier: hold M and grow GPUs (scaling), then grow M per GPU (density).
run 1 16
run 1 24
run 1 32
run 2 24
run 2 32
run 4 16
run 4 24

kill $GPU_PID 2>/dev/null || true
echo "=== r4-scale done (see per-R${R}M${M} logs + gpu.log for utilization) ==="
