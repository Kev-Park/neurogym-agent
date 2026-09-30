#!/bin/bash
#SBATCH --job-name=zl-meminst
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --time=00:35:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/slurm-%x-%j.out
#
# WHY did >=4 concurrent trainings OOM in the spread sweep (job 997415) when
# ONE 48-env training measures only ~95 GB? c1 (2 instances, 96 env) survived;
# c2 (4 instances, SAME runners/workers/envs) OOM'd at cold start -- the only
# change was the number of `ray.init()` instances. Suspect: Ray's default
# object store = 30% of NODE RAM (~113 GB) per instance, charged to the cgroup.
#
# Phase A: launch G instances of NR runners 45 s APART with Ray defaults and
#          sample the cgroup's memory.current + /dev/shm every 10 s -> the
#          per-instance step is read straight off the timeline.
# Phase B: same, but --object-store-gb 2 -> if the steps shrink to the ~25 GB a
#          6-runner training really needs, the multi-training "RAM wall" is a
#          Ray default, and the aggregate-throughput ceiling reopens.
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_outputs
J=$SLURM_JOB_ID
PKL=${1:-/scratch/kp0374/checkpoints/zmaxleft-v5/ckpt_000490.pkl}
G=${INSTANCES:-4}
NR=${RUNNERS:-6}
STAGGER=${STAGGER:-45}
HOLD=${HOLD:-150}            # seconds to keep sampling after the last launch
BATCH=${BATCH:-4000}
TMP=/tmp/zlmeminst_$J

export PYTHONUNBUFFERED=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_memory_monitor_refresh_ms=0
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_MESH_LRU_MB=128 NGL_NATIVE_CHUNK_LRU_MB=256 NGL_NATIVE_VAO_LRU_MB=128
export NGL_NATIVE_MESH_PREFETCH=4 NGL_NATIVE_FETCH_WORKERS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d && echo "[meminst] MPS up $(hostname) job=$J G=$G NR=$NR"
CG=$(awk -F: '{print $3}' /proc/self/cgroup | head -1)
MEMCUR="/sys/fs/cgroup${CG}/memory.current"

sample() {  # phase csv t0 -> append one row
  local cur shm nray
  cur=$(awk '{printf "%.1f", $1/1073741824}' "$MEMCUR" 2>/dev/null); cur=${cur:-NA}
  shm=$(df -BG /dev/shm | awk 'NR==2{sub("G","",$3); print $3}')
  nray=$(pgrep -c -f 'ray::' 2>/dev/null); nray=${nray:-0}
  echo "$(( SECONDS - $3 )),$cur,${shm:-NA},$nray" >> "$2"
}

run_phase() {
  local phase="$1" extra="$2"
  # separate `local`: bash expands the whole line before assigning (set -u trap)
  local csv="slurm_outputs/meminst-$phase-$J.csv" t0=$SECONDS pids="" i
  echo "t_s,mem_current_gb,shm_used_gb,ray_procs" > "$csv"
  echo "[meminst] === phase $phase  extra=[$extra] ==="
  for ((i=0; i<G; i++)); do
    mkdir -p "$TMP/$phase/g$i"; cp -f "$PKL" "$TMP/$phase/g$i/ckpt_000490.pkl"
    echo "[meminst] $phase: launching instance $i at t=$((SECONDS - t0))s"
    env CUDA_VISIBLE_DEVICES=$i UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
      NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J-$i.lock \
      uv run --no-sync python -m ngllib_agent.train \
        --config configs/native_zmaxleft_v6.yaml --run-name "meminst-$phase-g$i-$J" \
        --obs dino --vector threads \
        --num-env-runners "$NR" --num-envs-per-env-runner 2 \
        --num-gpus-per-env-runner 0.01 --num-cpus-per-env-runner 0.5 \
        --no-spawn-curriculum --no-degraded-exit --stagger-first-episode \
        --iters 999999 --train-batch-size "$BATCH" \
        --checkpoint-dir "$TMP/$phase/g$i" --checkpoint-every 999999 --resume \
        --wandb-mode disabled $extra > "slurm_outputs/meminst-$phase-g$i-$J.out" 2>&1 &
    pids="$pids $!"
    local tl=$SECONDS
    while [ $((SECONDS - tl)) -lt "$STAGGER" ]; do sample "$phase" "$csv" "$t0"; sleep 10; done
  done
  local th=$SECONDS
  while [ $((SECONDS - th)) -lt "$HOLD" ]; do sample "$phase" "$csv" "$t0"; sleep 10; done
  echo "[meminst] $phase: per-role breakdown after all $G instances up"
  bash native/probe_mem_breakdown.sh > "slurm_outputs/memprobe-meminst-$phase-$J.txt" 2>&1
  grep -E 'memory\.(current|peak)|/dev/shm|^(driver|runner|fetch_worker|ray_infra|TOTAL) ' \
    "slurm_outputs/memprobe-meminst-$phase-$J.txt" | sed "s/^/[meminst] $phase  /"
  echo "[meminst] $phase timeline (t_s,mem_gb,shm_gb,ray_procs):"
  awk -F, 'NR==1 || (NR-2)%3==0' "$csv" | sed 's/^/[meminst]   /'
  echo "[meminst] RESULT phase=$phase first=$(sed -n 2p "$csv") last=$(tail -1 "$csv") oom=$(dmesg 2>/dev/null | grep -c 'oom_kill' || echo NA)"
  kill $pids 2>/dev/null || true
  pkill -9 -f "meminst-$phase-" 2>/dev/null || true
  pkill -9 -f 'ray::' 2>/dev/null || true
  pkill -9 -f 'multiprocessing.spawn' 2>/dev/null || true
  sleep 15
}

run_phase default ""
run_phase smallstore "--object-store-gb 2"

echo "[meminst] === SUMMARY ==="
grep -hE '^\[meminst\] RESULT ' slurm_outputs/slurm-zl-meminst-$J.out 2>/dev/null || true
echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
rm -rf "$TMP" 2>/dev/null || true
echo "[meminst] done"
