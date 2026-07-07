#!/bin/bash
#SBATCH --job-name=r4-bench
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:3090:1
#SBATCH --nodelist=sarekl15-2,sarekl15-4,sarekl15-5,sarekl15-7,sarekl15-8,sarekl16-1,sarekl16-3,sarekl16-5,sarekl16-6,sarekl16-7,sarekl16-8
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4
#SBATCH --time=02:30:00
#SBATCH --output=slurm_outputs/r4-bench-%j.out

# REFINEMENT R4: threads-vs-spawn A/B at M=8, then threaded density sweep.
# Per phase: 3 PPO iters (dino obs), batch = 128*M, VRAM sampled every 10s.
set -e
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

VRAM_LOG=slurm_outputs/r4-vram-$SLURM_JOB_ID.log
( while true; do
    echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> "$VRAM_LOG"
    sleep 10
  done ) &
VRAM_PID=$!

run_phase() {
  local label=$1 vector=$2 m=$3
  echo "########## PHASE $label (vector=$vector M=$m) ##########"
  echo "PHASE $label start" >> "$VRAM_LOG"
  uv run --no-sync python scripts/ppo_smoke.py \
      --obs dino --num-env-runners 1 --num-gpus-per-env-runner 0.6 \
      --num-envs-per-env-runner "$m" --vector "$vector" \
      --iters 3 --train-batch-size $((128 * m)) --rollout-fragment-length 128 \
    2>&1 | grep -E "iter [0-9]+:|Traceback|glitch|complete" | tail -8
  echo "PHASE $label end" >> "$VRAM_LOG"
}

run_phase A-threads-m8  threads 8
run_phase B-spawn-m8    spawn   8
run_phase C-threads-m16 threads 16
run_phase D-threads-m24 threads 24

kill $VRAM_PID 2>/dev/null || true
echo "########## VRAM summary (max per phase) ##########"
awk '/PHASE .* start/{p=$2} /MiB/{gsub(/ MiB/,""); if ($2+0>mx[p]) mx[p]=$2} END{for (k in mx) print k, mx[k] " MiB"}' "$VRAM_LOG" | sort
echo "=== r4 bench done ==="
