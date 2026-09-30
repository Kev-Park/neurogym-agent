#!/bin/bash
#SBATCH --job-name=zl-spread1t
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:3090:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --time=01:10:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/slurm-%x-%j.out
#
# SINGLE-EXPERIMENT multi-GPU test: does placing ONE training's 24 env runners
# across 2 GPUs (num_gpus_per_env_runner=0.08 => Ray bin-packs ~12/GPU, each
# runner's DINO on its assigned GPU) beat 1 GPU x 24 (~170-180 sps)? This is the
# single-experiment speedup lever the 2-independent-training spread sweep did NOT
# measure (that gave node AGGREGATE across 2 policies). 48 envs = ~360G (RAM wall,
# same as 1 GPU). batch 16000 (production), graphs on (v6). Four arms on the SAME
# node: the 48 envs on 1 / 2 / 4 / 8 GPUs (24 / 12 / 6 / 3 runners per GPU) —
# i.e. "fewer runners on more GPUs" for ONE experiment. Steady sps = mean of
# iters after the first 3. Expect gains to flatten once the GPU stops being the
# bottleneck and the 24 GILs + the synchronous learner barrier bind instead.
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_outputs
J=$SLURM_JOB_ID
PKL=${1:-/scratch/kp0374/checkpoints/zmaxleft-v5/ckpt_000490.pkl}
BATCH=${BATCH:-16000}
WINDOW=${WINDOW:-700}
NR=${RUNNERS:-24}     # 24 x M=2 = 48 envs ~360G = the node RAM wall (needs a free node)
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
  local name="$1" gpus="$2" gpufrac="$3"
  # Separate `local`: bash expands a whole `local` line before assigning, so
  # "$name" on the same line as name="$1" is unbound under set -u (job 997617).
  local dir="$TMP/$name" f="slurm_outputs/spread1t-$name-$J.out"
  mkdir -p "$dir"; cp -f "$PKL" "$dir/ckpt_000490.pkl"
  echo "[spread1t] === arm $name  CUDA_VISIBLE_DEVICES=$gpus  gpu_frac=$gpufrac ==="
  env CUDA_VISIBLE_DEVICES=$gpus UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
    NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J-$name.lock \
    uv run --no-sync python -m ngllib_agent.train \
      --config configs/native_zmaxleft_v6.yaml --run-name "spread1t-$name-$J" \
      --obs dino --vector threads \
      --num-env-runners "$NR" --num-envs-per-env-runner 2 \
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

# Placement: Ray PACKS fractional GPU requests onto one device until it is full
# (floor(1/frac) runners per GPU), so an even spread needs frac = 1/(runners per
# GPU) exactly — e.g. 0.08 would put 12 runners on GPU0 and only the rest on
# GPU1. Fractions below are 1/k for k = NR/G runners per GPU (24 runners):
#   1 GPU: 24/GPU   2 GPU: 12/GPU   4 GPU: 6/GPU   8 GPU: 3/GPU
# Baseline first so a walltime cut still leaves the key comparison.
run_arm 1gpu "0"               0.041
run_arm 2gpu "0,1"             0.083
run_arm 4gpu "0,1,2,3"         0.166
run_arm 8gpu "0,1,2,3,4,5,6,7" 0.333

echo "[spread1t] === SUMMARY ==="
grep -hE '^\[spread1t\] RESULT ' slurm_outputs/slurm-zl-spread1t-$J.out 2>/dev/null || true
echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
rm -rf "$TMP" 2>/dev/null || true
echo "[spread1t] done"
