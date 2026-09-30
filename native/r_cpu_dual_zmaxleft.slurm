#!/bin/bash
#SBATCH --job-name=zl-cpudual
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=00:50:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/slurm-%x-%j.out
#
# IS CPU THE NODE-AGGREGATE BOTTLENECK for the co-pane workload? Evidence so
# far: RAM is ~2.6-2.9 GB/env (not binding until ~128 env), yet 2 trainings x
# 24 runners on one node gave 168 aggregate sps < 180 for ONE training (job
# 997415 c1), and one 24-runner training already burned ~43 of 64 cores (live
# v5 probe: 62.6% nice + 4.9% sys). Same-node phases, each with mpstat:
#   single_w2 : 1 training  x 24 runners, default 2 tile + 2 mesh workers/runner
#   dual_w2   : 2 trainings x 24 runners (48 procs + 192 workers) -> cores ~100%?
#   dual_w1   : 2 trainings x 24 runners, 1 tile + 1 mesh worker/runner (half the
#               worker PROCESSES) -> if CPU-bound, aggregate should recover
# Per phase: cores busy (64 x (100-idle)/100), memory.current, per-training
# steady sps (mean of iters after the first 2), aggregate. Launches staggered
# 60s to avoid the simultaneous-cold-start transient. Needs a whole node's cores
# (--cpus-per-task=64) so a co-tenant cannot confound the CPU read.
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_outputs
J=$SLURM_JOB_ID
PKL=${1:-/scratch/kp0374/checkpoints/zmaxleft-v5/ckpt_000490.pkl}
BATCH=${BATCH:-8000}
NR=${RUNNERS:-24}
WARM=${WARM:-480}
STAGGER=${STAGGER:-60}
TMP=/tmp/zlcpudual_$J

export PYTHONUNBUFFERED=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_memory_monitor_refresh_ms=0
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_MESH_LRU_MB=128 NGL_NATIVE_CHUNK_LRU_MB=256 NGL_NATIVE_VAO_LRU_MB=128
export NGL_NATIVE_MESH_PREFETCH=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d && echo "[cpudual] MPS up $(hostname) job=$J cores=$(nproc) batch=$BATCH runners=$NR"
CG=$(awk -F: '{print $3}' /proc/self/cgroup | head -1)
MEMCUR="/sys/fs/cgroup${CG}/memory.current"

run_phase() {
  local phase="$1" ntrain="$2" tile_w="$3" mesh_w="$4"
  local pids="" i t0=$SECONDS
  echo "[cpudual] === phase $phase: $ntrain training(s) x $NR runners, workers/runner tile=$tile_w mesh=$mesh_w ==="
  for ((i=0; i<ntrain; i++)); do
    mkdir -p "$TMP/$phase/g$i"; cp -f "$PKL" "$TMP/$phase/g$i/ckpt_000490.pkl"
    env CUDA_VISIBLE_DEVICES=$i UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
      NGL_NATIVE_FETCH_WORKERS="$tile_w" NGL_NATIVE_MESH_WORKERS="$mesh_w" \
      NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J-$i.lock \
      uv run --no-sync python -m ngllib_agent.train \
        --config configs/native_zmaxleft_v6.yaml --run-name "cpudual-$phase-g$i-$J" \
        --obs dino --vector threads \
        --num-env-runners "$NR" --num-envs-per-env-runner 2 \
        --num-gpus-per-env-runner 0.02 --num-cpus-per-env-runner 0.5 \
        --no-spawn-curriculum --no-degraded-exit --stagger-first-episode \
        --iters 999999 --train-batch-size "$BATCH" \
        --checkpoint-dir "$TMP/$phase/g$i" --checkpoint-every 999999 --resume \
        --wandb-mode disabled > "slurm_outputs/cpudual-$phase-g$i-$J.out" 2>&1 &
    pids="$pids $!"
    [ "$i" -lt $((ntrain - 1)) ] && sleep "$STAGGER"
  done
  while [ $((SECONDS - t0)) -lt "$WARM" ]; do sleep 10; done
  echo "[cpudual] $phase: warm ($((SECONDS - t0))s); mpstat 5x6"
  local idle busy mem
  idle=$(mpstat 5 6 2>/dev/null | awk '/^Average/ && $2=="all"{print $NF}')
  busy=$(awk -v idle="${idle:-0}" -v n="$(nproc)" 'BEGIN{printf "%.1f", n*(100-idle)/100}')
  mem=$(awk '{printf "%.1f", $1/1073741824}' "$MEMCUR" 2>/dev/null); mem=${mem:-NA}
  local per="" agg=0 s nproc_all
  for ((i=0; i<ntrain; i++)); do
    s=$(grep -hoE 'sps=[0-9.]+' "slurm_outputs/cpudual-$phase-g$i-$J.out" 2>/dev/null | sed 's/sps=//' \
        | awk 'NR>2{k++; sum+=$1} END{if(k>0) printf "%.1f", sum/k; else print "0"}')
    per="$per g$i=$s"; agg=$(awk -v a="$agg" -v s="$s" 'BEGIN{printf "%.1f", a+s}')
  done
  nproc_all=$(pgrep -c -f 'ray::|multiprocessing.spawn' 2>/dev/null); nproc_all=${nproc_all:-0}
  echo "[cpudual] RESULT phase=$phase trainings=$ntrain agg_sps=$agg per=[$per] cores_busy=$busy/$(nproc) idle_pct=${idle:-NA} mem_gb=$mem procs=$nproc_all"
  kill $pids 2>/dev/null || true
  pkill -9 -f "cpudual-$phase-" 2>/dev/null || true
  pkill -9 -f 'ray::' 2>/dev/null || true
  pkill -9 -f 'multiprocessing.spawn' 2>/dev/null || true
  sleep 15
}

run_phase single_w2 1 2 2
run_phase dual_w2   2 2 2
run_phase dual_w1   2 1 1

echo "[cpudual] === SUMMARY ==="
grep -hE '^\[cpudual\] RESULT ' slurm_outputs/slurm-zl-cpudual-$J.out 2>/dev/null || true
echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
rm -rf "$TMP" 2>/dev/null || true
echo "[cpudual] done"
