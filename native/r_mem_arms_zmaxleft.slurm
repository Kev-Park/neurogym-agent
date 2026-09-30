#!/bin/bash
#SBATCH --job-name=zl-memarms
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=00:45:00
# Defaults = the PRODUCTION profile (24 runners x M=2 = 48 envs ~360G, needs a
# near-empty node). On a busy fleet: N2=12 sbatch --mem=260G <this> (24 envs).
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/slurm-%x-%j.out
#
# Measured host-RAM breakdown + the "bigger M" trade, at FIXED envs (N2*2):
#   arm m2: N2 runners x M=2      arm m4: N2/2 runners x M=4    (N2 default 24)
# Same env count, half the runner PROCESSES (and their per-process fetch/mesh
# worker pools, which are class-level = per process). Each arm: warm to >=3
# iters, then probe_mem_breakdown.sh (by-role RSS/PSS, cgroup memory.current/
# peak, workers per runner, smaps of a runner/worker/driver) and steady sps.
#   RAM(m2) - RAM(m4)  = what 6 fewer runner processes (+their pools) cost
#   sps(m2) vs sps(m4) = the GIL/process-count price of bigger M at fixed envs
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_outputs
J=$SLURM_JOB_ID
PKL=${1:-/scratch/kp0374/checkpoints/zmaxleft-v5/ckpt_000490.pkl}
BATCH=${BATCH:-8000}          # faster iters => warm sooner; RAM is ~batch-independent
N2=${N2:-24}                  # runners in the M=2 arm; the M=4 arm uses N2/2 (same envs)
WARM_ITERS=${WARM_ITERS:-3}
WARM_MAX_S=${WARM_MAX_S:-600}
TMP=/tmp/zlmemarms_$J

export PYTHONUNBUFFERED=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_memory_monitor_refresh_ms=0
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_MESH_LRU_MB=128 NGL_NATIVE_CHUNK_LRU_MB=256 NGL_NATIVE_VAO_LRU_MB=128
export NGL_NATIVE_MESH_PREFETCH=4 NGL_NATIVE_FETCH_WORKERS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d && echo "[memarms] MPS up $(hostname) job=$J batch=$BATCH"

run_arm() {
  local name="$1" nr="$2" m="$3"
  local dir="$TMP/$name" f="slurm_outputs/memarms-$name-$J.out" probe="slurm_outputs/memprobe-$name-$J.txt"
  mkdir -p "$dir"; cp -f "$PKL" "$dir/ckpt_000490.pkl"
  echo "[memarms] === arm $name: $nr runners x M=$m = $((nr*m)) envs ==="
  env CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
    NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J.lock \
    uv run --no-sync python -m ngllib_agent.train \
      --config configs/native_zmaxleft_v6.yaml --run-name "memarms-$name-$J" \
      --obs dino --vector threads \
      --num-env-runners "$nr" --num-envs-per-env-runner "$m" \
      --num-gpus-per-env-runner 0.02 --num-cpus-per-env-runner 1.0 \
      --no-spawn-curriculum --no-degraded-exit --stagger-first-episode \
      --iters 999999 --train-batch-size "$BATCH" \
      --checkpoint-dir "$dir" --checkpoint-every 999999 --resume \
      --wandb-mode disabled > "$f" 2>&1 &
  local pid=$! t0=$SECONDS n=0
  while [ $((SECONDS - t0)) -lt "$WARM_MAX_S" ]; do
    n=$(grep -cE '^iter ' "$f" 2>/dev/null); n=${n:-0}   # grep -c prints 0 itself; `|| echo 0` doubled it
    [ "$n" -ge "$WARM_ITERS" ] && break
    sleep 5
  done
  echo "[memarms] $name warm after $((SECONDS - t0))s with $n iters; probing memory"
  bash native/probe_mem_breakdown.sh > "$probe" 2>&1
  grep -E 'memory\.(current|peak)|^(driver|runner|fetch_worker|mp_tracker|ray_infra|mps|other|TOTAL) |runners have|not parented' "$probe" | sed "s/^/[memarms] $name  /"
  local s
  s=$(grep -hoE 'sps=[0-9.]+' "$f" 2>/dev/null | sed 's/sps=//' \
      | awk 'NR>1{k++; sum+=$1} END{if(k>0) printf "%.1f", sum/k; else print "0"}')
  echo "[memarms] RESULT arm=$name runners=$nr M=$m envs=$((nr*m)) sps=$s iters=$n probe=$probe"
  kill $pid 2>/dev/null || true
  pkill -9 -f "memarms-$name-$J" 2>/dev/null || true
  pkill -9 -f 'ray::' 2>/dev/null || true
  pkill -9 -f 'multiprocessing.spawn' 2>/dev/null || true
  sleep 12
}

run_arm m2 "$N2" 2
run_arm m4 "$((N2 / 2))" 4

echo "[memarms] === SUMMARY ==="
grep -hE '^\[memarms\] RESULT ' slurm_outputs/slurm-zl-memarms-$J.out 2>/dev/null || true
echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
rm -rf "$TMP" 2>/dev/null || true
echo "[memarms] done"
