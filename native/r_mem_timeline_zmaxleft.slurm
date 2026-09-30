#!/bin/bash
#SBATCH --job-name=zl-memtl
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=00:55:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/slurm-%x-%j.out
#
# Does host RAM GROW over a run? The 6-minute breakdown (job 998019) measured
# ~95 GB for the production 24x2 = 48-env training, yet >48-env configs OOM'd
# at 360G in the spread sweep — inconsistent unless memory grows with time (mesh
# accumulation, caches beyond the LRU caps, fragmentation) or per-Ray-instance
# overhead is large. Run the production config for WINDOW seconds, logging the
# job cgroup's memory.current/peak + iters/sps every 30s to a CSV (survives an
# OOM kill), then take a second full breakdown to see WHICH role grew.
# --mem=220G is deliberate: well above the 95 GB measured, well below 360 — if
# this OOMs, the CSV shows when and the wall is real; if flat, it is a phantom.
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_outputs
J=$SLURM_JOB_ID
PKL=${1:-/scratch/kp0374/checkpoints/zmaxleft-v5/ckpt_000490.pkl}
BATCH=${BATCH:-16000}
NR=${RUNNERS:-24}
WINDOW=${WINDOW:-2400}
TMP=/tmp/zlmemtl_$J
CSV="slurm_outputs/memtl-$J.csv"
F="slurm_outputs/memtl-train-$J.out"

export PYTHONUNBUFFERED=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_memory_monitor_refresh_ms=0
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_MESH_LRU_MB=128 NGL_NATIVE_CHUNK_LRU_MB=256 NGL_NATIVE_VAO_LRU_MB=128
export NGL_NATIVE_MESH_PREFETCH=4 NGL_NATIVE_FETCH_WORKERS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" "$TMP"
nvidia-cuda-mps-control -d && echo "[memtl] MPS up $(hostname) job=$J runners=$NR batch=$BATCH window=${WINDOW}s"
cp -f "$PKL" "$TMP/ckpt_000490.pkl"

CG=$(awk -F: '{print $3}' /proc/self/cgroup | head -1)
MEMCUR="/sys/fs/cgroup${CG}/memory.current"; MEMPEAK="/sys/fs/cgroup${CG}/memory.peak"
echo "t_s,mem_current_gb,mem_peak_gb,iters,last_sps" > "$CSV"

env CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
  NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J.lock \
  uv run --no-sync python -m ngllib_agent.train \
    --config configs/native_zmaxleft_v6.yaml --run-name "memtl-$J" \
    --obs dino --vector threads \
    --num-env-runners "$NR" --num-envs-per-env-runner 2 \
    --num-gpus-per-env-runner 0.02 --num-cpus-per-env-runner 1.0 \
    --no-spawn-curriculum --no-degraded-exit --stagger-first-episode \
    --iters 999999 --train-batch-size "$BATCH" \
    --checkpoint-dir "$TMP" --checkpoint-every 999999 --resume \
    --wandb-mode disabled > "$F" 2>&1 &
PID=$!
T0=$SECONDS
while [ $((SECONDS - T0)) -lt "$WINDOW" ] && kill -0 "$PID" 2>/dev/null; do
  cur=$(awk '{printf "%.2f", $1/1073741824}' "$MEMCUR" 2>/dev/null); cur=${cur:-NA}
  pk=$(awk '{printf "%.2f", $1/1073741824}' "$MEMPEAK" 2>/dev/null); pk=${pk:-NA}
  n=$(grep -cE '^iter ' "$F" 2>/dev/null); n=${n:-0}
  s=$(grep -hoE 'sps=[0-9.]+' "$F" 2>/dev/null | tail -1 | sed 's/sps=//'); s=${s:-NA}
  echo "$((SECONDS - T0)),$cur,$pk,$n,$s" >> "$CSV"
  sleep 30
done
if kill -0 "$PID" 2>/dev/null; then
  echo "[memtl] window done, training still alive; final breakdown"
  bash native/probe_mem_breakdown.sh > "slurm_outputs/memprobe-memtl-$J.txt" 2>&1
  grep -E 'memory\.(current|peak)|^(driver|runner|fetch_worker|ray_infra|TOTAL) ' "slurm_outputs/memprobe-memtl-$J.txt" | sed 's/^/[memtl] final  /'
else
  echo "[memtl] TRAINING DIED before the window ended (OOM?) — see $F tail:"
  tail -5 "$F"
fi
echo "[memtl] timeline (every ~5 min):"
awk -F, 'NR==1 || (NR-2)%10==0' "$CSV" | sed 's/^/[memtl]   /'
echo "[memtl] RESULT first=$(sed -n 2p "$CSV") last=$(tail -1 "$CSV")"
kill $PID 2>/dev/null || true
pkill -9 -f "memtl-$J" 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
rm -rf "$TMP" 2>/dev/null || true
echo "[memtl] done"
