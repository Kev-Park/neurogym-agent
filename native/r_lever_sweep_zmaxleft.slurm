#!/bin/bash
#SBATCH --job-name=zl-levers
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=300G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=02:30:00
#SBATCH --output=slurm_outputs/slurm-%x-%j.out
#
# Exhaustive per-GPU lever sweep for the v5 co-pane workload. One training per
# lever on ONE GPU, warm-started from the v5 ckpt (realistic mesh load), each run
# bounded by WINDOW; steady sps = mean of iters after the first 2. Levers:
#   SAFE (bitwise / no obs change, ADOPTABLE): cuda_graph, envs/runner (M),
#     batch, mesh/chunk/VAO LRU, fetch workers.
#   DYNAMICS-GATED (numerics/obs change, need a transfer check): fp16, compile,
#     input_size, graph+fp16.
# RAM WALL (2026-09-30): the co-pane workload costs ~7.5G/env host RAM, so at
# M=2 a 300G job holds ~40 envs (~20 runners). Default N=16 (32 envs, ~240G) is
# RAM-safe and schedulable on a shared node; lever RANKING is density-robust.
# Raise --mem toward 370G (near-exclusive) if you need N=24 for a production-
# density measurement. Set RUNNERS to override.
#
# Usage: RUNNERS=<N> sbatch native/r_lever_sweep_zmaxleft.slurm [state-pkl] [batch]
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_outputs
J=$SLURM_JOB_ID
PKL=${1:-/scratch/kp0374/checkpoints/zmaxleft-v5/ckpt_000490.pkl}
BATCH=${2:-4000}
N=${RUNNERS:-16}
WINDOW=${WINDOW:-360}
TMP=/tmp/zllever_$J

export PYTHONUNBUFFERED=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_memory_monitor_refresh_ms=0
export NGL_RUNNER_STAGGER_SEC=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Baseline renderer LRU budgets (a lever overrides these per-config via env).
export NGL_NATIVE_MESH_LRU_MB=128 NGL_NATIVE_CHUNK_LRU_MB=256 NGL_NATIVE_VAO_LRU_MB=128
export NGL_NATIVE_MESH_PREFETCH=4 NGL_NATIVE_FETCH_WORKERS=2

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d && echo "[lever] MPS up on $(hostname) job=$J N=$N batch=$BATCH pkl=$PKL"

run_lever() {
  local name="$1" envs="$2" extra="$3"
  local dir="$TMP/$name" f="slurm_outputs/lever-$name-$J.out"
  mkdir -p "$dir"; cp -f "$PKL" "$dir/ckpt_000490.pkl"
  echo "[lever] === $name  env=[$envs]  args=[$extra] ==="
  # env applies the per-lever var overrides; empty $envs is a no-op. $extra is
  # intentionally unquoted (word-split into flags). Repeated --train-batch-size /
  # --num-envs-per-env-runner let a lever override the defaults (argparse: last wins).
  env $envs CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
    NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J.lock \
    uv run --no-sync python -m ngllib_agent.train \
      --config configs/native_zmaxleft_v5.yaml --run-name "lever-$name-$J" \
      --obs dino --vector threads \
      --num-env-runners "$N" --num-envs-per-env-runner 2 \
      --num-gpus-per-env-runner 0.02 --num-cpus-per-env-runner 1.0 \
      --no-spawn-curriculum --no-degraded-exit --stagger-first-episode \
      --iters 999999 --train-batch-size "$BATCH" \
      --checkpoint-dir "$dir" --checkpoint-every 999999 --resume \
      --wandb-mode disabled $extra > "$f" 2>&1 &
  local pid=$!
  sleep "$WINDOW"
  local s cold iters caps
  s=$(grep -hoE 'sps=[0-9.]+' "$f" 2>/dev/null | sed 's/sps=//' \
      | awk 'NR>2{n++; sum+=$1} END{if(n>0) printf "%.1f", sum/n; else print "0"}')
  cold=$(grep -hoE 'sps=[0-9.]+' "$f" 2>/dev/null | head -1 | sed 's/sps=//')
  iters=$(grep -cE '^iter ' "$f" 2>/dev/null || echo 0)
  caps=$(grep -c 'captured CUDA graph' "$f" 2>/dev/null || echo 0)
  echo "[lever] RESULT $name steady_sps=$s cold1=${cold:-NA} iters=$iters graph_captures=$caps"
  kill $pid 2>/dev/null || true
  pkill -9 -f "lever-$name-$J" 2>/dev/null || true
  pkill -9 -f 'ray::' 2>/dev/null || true
  sleep 8
}

# name | per-lever env | extra train args
run_lever baseline   ""                                                             ""
run_lever graph      ""                                                             "--set obs.dino.cuda_graph=true"
run_lever m1         ""                                                             "--num-envs-per-env-runner 1"
run_lever m4         ""                                                             "--num-envs-per-env-runner 4"
run_lever graph_m1   ""                                                             "--set obs.dino.cuda_graph=true --num-envs-per-env-runner 1"
run_lever batch8k    ""                                                             "--train-batch-size 8000"
run_lever bigLRU     "NGL_NATIVE_MESH_LRU_MB=256 NGL_NATIVE_CHUNK_LRU_MB=512 NGL_NATIVE_VAO_LRU_MB=256" ""
run_lever fetch4     "NGL_NATIVE_FETCH_WORKERS=4"                                   ""
run_lever fp16       ""                                                             "--set obs.dino.fp16=true"
run_lever compile    ""                                                             "--set obs.dino.compile=true"
run_lever in168      ""                                                             "--set obs.dino.input_size=168"
run_lever graph_fp16 ""                                                             "--set obs.dino.cuda_graph=true --set obs.dino.fp16=true"

echo "[lever] === SUMMARY ==="
grep -hE '^\[lever\] RESULT ' slurm_outputs/slurm-zl-levers-$J.out 2>/dev/null || true
echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
rm -rf "$TMP" 2>/dev/null || true
echo "[lever] done"
