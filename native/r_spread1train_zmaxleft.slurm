#!/bin/bash
#SBATCH --job-name=zl-spread1t
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=00:40:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/slurm-%x-%j.out
#
# SINGLE-EXPERIMENT multi-GPU test: does placing ONE training's 24 env runners
# across 2 GPUs (num_gpus_per_env_runner=0.08 => Ray bin-packs ~12/GPU, each
# runner's DINO on its assigned GPU) beat 1 GPU x 24 (~170-180 sps)? This is the
# single-experiment speedup lever the 2-independent-training spread sweep did NOT
# measure (that gave node AGGREGATE across 2 policies). 48 envs = ~360G (RAM wall,
# same as 1 GPU). batch 16000 (production), graphs on. Two arms: 2-GPU vs 1-GPU,
# same node. Steady sps = mean of iters after the first 3.
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_outputs
J=$SLURM_JOB_ID
PKL=${1:-/scratch/kp0374/checkpoints/zmaxleft-v5/ckpt_000490.pkl}
BATCH=${BATCH:-16000}
WINDOW=${WINDOW:-700}
TMP=/tmp/zlspread1t_$J

export PYTHONUNBUFFERED=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_memory_monitor_refresh_ms=0
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_MESH_LRU_MB=128 NGL_NATIVE_CHUNK_LRU_MB=256 NGL_NATIVE_VAO_LRU_MB=128
export NGL_NATIVE_MESH_PREFETCH=4 NGL_NATIVE_FETCH_WORKERS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d && echo "[spread1t] MPS up $(hostname) job=$J batch=$BATCH"

# arm name | GPUs visible to the single training | num-gpus-per-env-runner
run_arm() {
  local name="$1" gpus="$2" gpufrac="$3" dir="$TMP/$name" f="slurm_outputs/spread1t-$name-$J.out"
  mkdir -p "$dir"; cp -f "$PKL" "$dir/ckpt_000490.pkl"
  echo "[spread1t] === arm $name  CUDA_VISIBLE_DEVICES=$gpus  gpu_frac=$gpufrac ==="
  env CUDA_VISIBLE_DEVICES=$gpus UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
    NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J-$name.lock \
    uv run --no-sync python -m ngllib_agent.train \
      --config configs/native_zmaxleft_v6.yaml --run-name "spread1t-$name-$J" \
      --obs dino --vector threads \
      --num-env-runners 24 --num-envs-per-env-runner 2 \
      --num-gpus-per-env-runner "$gpufrac" --num-cpus-per-env-runner 1.0 \
      --no-spawn-curriculum --no-degraded-exit --stagger-first-episode \
      --iters 999999 --train-batch-size "$BATCH" \
      --checkpoint-dir "$dir" --checkpoint-every 999999 --resume \
      --wandb-mode disabled > "$f" 2>&1 &
  local pid=$!
  sleep "$WINDOW"
  local s iters caps
  s=$(grep -hoE 'sps=[0-9.]+' "$f" 2>/dev/null | sed 's/sps=//' \
      | awk 'NR>3{n++; sum+=$1} END{if(n>0) printf "%.1f", sum/n; else print "0"}')
  iters=$(grep -cE '^iter ' "$f" 2>/dev/null || echo 0)
  caps=$(grep -c 'captured CUDA graph' "$f" 2>/dev/null || echo 0)
  echo "[spread1t] RESULT arm=$name sps=$s iters=$iters graph_captures=$caps"
  kill $pid 2>/dev/null || true
  pkill -9 -f "spread1t-$name-$J" 2>/dev/null || true
  pkill -9 -f 'ray::' 2>/dev/null || true
  sleep 10
}

# 2-GPU: 24 runners x 0.08 = 1.92 GPU => Ray spreads ~12/GPU. 1-GPU: 24 x 0.02 on 1.
run_arm 2gpu "0,1" 0.08
run_arm 1gpu "0"   0.02

echo "[spread1t] === SUMMARY ==="
grep -hE '^\[spread1t\] RESULT ' slurm_outputs/slurm-zl-spread1t-$J.out 2>/dev/null || true
echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
rm -rf "$TMP" 2>/dev/null || true
echo "[spread1t] done"
