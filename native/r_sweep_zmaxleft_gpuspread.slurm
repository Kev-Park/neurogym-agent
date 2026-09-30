#!/bin/bash
#SBATCH --job-name=zl-gpuspread
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:3090:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=01:15:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/slurm-%x-%j.out
#
# Task 2: does SPREADING the co-pane (v5) workload across MORE GPUs (fewer
# runners/GPU) beat concentrating it? The live v5 run is 1 GPU x 24 runners and
# is GPU-compute-100% + VRAM-98% — spreading should relieve both. Sweep (G GPUs,
# N runners/GPU), each GPU an independent training warm-started from the v5 ckpt
# (realistic mesh load), MPS on, learner on CPU (production-faithful; the GPU is
# the bottleneck so the learner must not steal from render/DINO). Aggregate node
# sps = sum of per-GPU steady sps (mean of the last 3 iters).
#
# Usage: sbatch native/r_sweep_zmaxleft_gpuspread.slurm [state-pkl] [batch]
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_outputs
J=$SLURM_JOB_ID
PKL=${1:-/scratch/kp0374/checkpoints/zmaxleft-v5/ckpt_000490.pkl}
BATCH=${2:-4000}
# Cold start is heavy for this workload (24 DINO loads + cold mesh/chunk caches +
# runner stagger => ~100s before the 1st iter, which itself is cold-cache slow).
# WINDOW must clear cold start AND leave >=6 steady iters; parsing drops the
# first 2 iters. VRAM-thrashing high-density configs simply report low steady sps.
WINDOW=${WINDOW:-540}
TMP=/tmp/zlsweep_$J

export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_memory_monitor_refresh_ms=0
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_FETCH_WORKERS=2
export NGL_NATIVE_MESH_LRU_MB=128
export NGL_NATIVE_CHUNK_LRU_MB=256
export NGL_NATIVE_VAO_LRU_MB=128
export NGL_NATIVE_MESH_PREFETCH=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MPS: lets the G per-GPU learners+runners share each device without hard
# context-switch serialization (same as the production launcher).
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d && echo "[sweep] MPS up on $(hostname) job=$J cores=$(nproc) gpus=$(nvidia-smi -L | wc -l) batch=$BATCH pkl=$PKL"

# (G GPUs, N runners/GPU). Key comparisons first (survive an early timeout):
#   c0 (1,24) baseline = live v5    c1..c3 (48 runners spread over 2/4/8 GPUs)
#   c4 (8,8) 64-runner max spread
CFG_G=(1 2 4 8 8)
CFG_N=(24 24 12 6 8)

run_config() {
  local c=$1 G=${2} N=${3}
  echo "[sweep] === config c$c: G=$G GPUs x N=$N runners/GPU (total $((G*N)) runners, M=2) ==="
  # --iters is an ABSOLUTE target (train.py:397); warm-start resumes at iter
  # ~490, so a small --iters exits after 1 iter. We pass a huge --iters so the
  # WINDOW (not the iter count) bounds each config and yields many steady iters.
  local pids="" i
  for ((i=0; i<G; i++)); do
    mkdir -p "$TMP/c${c}_g${i}"
    cp -f "$PKL" "$TMP/c${c}_g${i}/ckpt_000490.pkl"
    CUDA_VISIBLE_DEVICES=$i UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
      uv run --no-sync python -m ngllib_agent.train \
        --config configs/native_zmaxleft_v5.yaml --run-name "sweep-c${c}-g${i}-$J" \
        --obs dino --vector threads \
        --num-env-runners "$N" --num-envs-per-env-runner 2 \
        --num-gpus-per-env-runner 0.01 --num-cpus-per-env-runner 0.5 \
        --no-spawn-curriculum --no-degraded-exit --stagger-first-episode \
        --iters 999999 --train-batch-size "$BATCH" \
        --checkpoint-dir "$TMP/c${c}_g${i}" --checkpoint-every 999999 --resume \
        --wandb-mode disabled > "slurm_outputs/sweep-c${c}-g${i}-$J.out" 2>&1 &
    pids="$pids $!"
  done
  echo "[sweep] c$c: $G trainings launched, window ${WINDOW}s"
  sleep "$WINDOW"
  # steady per-GPU sps = mean of iters AFTER the first 2 (drop cold-cache iters);
  # node sps = sum over GPUs. Also report the cold 1st-iter sps for context.
  local total=0 got=0
  for ((i=0; i<G; i++)); do
    local f="slurm_outputs/sweep-c${c}-g${i}-$J.out"
    local s cold iters
    s=$(grep -hoE 'sps=[0-9.]+' "$f" 2>/dev/null | sed 's/sps=//' \
        | awk 'NR>2{n++; sum+=$1} END{if(n>0) printf "%.1f", sum/n; else print "0"}')
    cold=$(grep -hoE 'sps=[0-9.]+' "$f" 2>/dev/null | head -1 | sed 's/sps=//')
    iters=$(grep -cE '^iter ' "$f" 2>/dev/null || echo 0)
    echo "[sweep] c$c gpu$i: steady_sps=$s cold1=${cold:-NA} (iters_done=$iters)"
    total=$(awk -v t="$total" -v s="$s" 'BEGIN{printf "%.1f", t+s}')
    [ "$(awk -v s="$s" 'BEGIN{print (s>0)}')" = 1 ] && got=$((got+1))
  done
  local perrun
  perrun=$(awk -v t="$total" -v g="$((G*N))" 'BEGIN{printf "%.2f", (g>0)? t/g : 0}')
  echo "[sweep] RESULT c$c G=$G N=$N total_runners=$((G*N)) node_sps=$total per_runner=$perrun gpus_ok=$got/$G"
  # teardown before the next config so GPUs/cores are clean
  kill $pids 2>/dev/null || true
  pkill -9 -f "sweep-c${c}-" 2>/dev/null || true
  pkill -9 -f 'ray::' 2>/dev/null || true
  sleep 10
}

for c in 0 1 2 3 4; do
  run_config "$c" "${CFG_G[$c]}" "${CFG_N[$c]}"
done

echo "[sweep] === SUMMARY (node_sps by layout) ==="
grep -hE '^\[sweep\] RESULT ' slurm_outputs/slurm-zl-gpuspread-$J.out 2>/dev/null || true

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
rm -rf "$TMP" 2>/dev/null || true
echo "[sweep] done"
