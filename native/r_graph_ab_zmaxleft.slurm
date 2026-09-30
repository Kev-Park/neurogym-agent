#!/bin/bash
#SBATCH --job-name=zl-graphab
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=220G
#SBATCH --time=00:50:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/slurm-%x-%j.out
#
# Definitive CUDA-graph decision + SOTA number for the co-pane winner layout
# (2 GPU x 12 runners = 48 envs, from the spread sweep). Two arms back-to-back on
# the SAME node (controls node variance): nograph vs graph, each at the
# production batch with a LONG window so capture is fully amortized over many
# steady iters (the lever sweep's 360s window under-measured graphs: only 4/12
# captured). Steady node_sps = sum over GPUs of mean(iters after first 3).
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_outputs
J=$SLURM_JOB_ID
PKL=${1:-/scratch/kp0374/checkpoints/zmaxleft-v5/ckpt_000490.pkl}
# Default = a schedulable 1-GPU x 12-runner (24 env) clean graph A/B: long window
# so capture amortizes. The 2x12 node SOTA comes from the spread sweep (234 sps);
# apply the graph factor measured here. batch 8000 => more iters/window.
G=${GPUS:-1}
NPG=${NPG:-12}
BATCH=${BATCH:-8000}
WINDOW=${WINDOW:-700}
TMP=/tmp/zlgraphab_$J

export PYTHONUNBUFFERED=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_memory_monitor_refresh_ms=0
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_MESH_LRU_MB=128 NGL_NATIVE_CHUNK_LRU_MB=256 NGL_NATIVE_VAO_LRU_MB=128
export NGL_NATIVE_MESH_PREFETCH=4 NGL_NATIVE_FETCH_WORKERS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d && echo "[graphab] MPS up $(hostname) job=$J G=$G NPG=$NPG batch=$BATCH window=$WINDOW"

run_arm() {
  local name="$1" extra="$2" pids="" i
  echo "[graphab] === arm $name  extra=[$extra] ==="
  for ((i=0; i<G; i++)); do
    mkdir -p "$TMP/$name/g$i"; cp -f "$PKL" "$TMP/$name/g$i/ckpt_000490.pkl"
    env CUDA_VISIBLE_DEVICES=$i UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
      NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J.lock \
      uv run --no-sync python -m ngllib_agent.train \
        --config configs/native_zmaxleft_v5.yaml --run-name "graphab-$name-g$i-$J" \
        --obs dino --vector threads \
        --num-env-runners "$NPG" --num-envs-per-env-runner 2 \
        --num-gpus-per-env-runner 0.01 --num-cpus-per-env-runner 1.0 \
        --no-spawn-curriculum --no-degraded-exit --stagger-first-episode \
        --iters 999999 --train-batch-size "$BATCH" \
        --checkpoint-dir "$TMP/$name/g$i" --checkpoint-every 999999 --resume \
        --wandb-mode disabled $extra > "slurm_outputs/graphab-$name-g$i-$J.out" 2>&1 &
    pids="$pids $!"
  done
  echo "[graphab] arm $name: $G trainings launched, window ${WINDOW}s"
  sleep "$WINDOW"
  local total=0
  for ((i=0; i<G; i++)); do
    local f="slurm_outputs/graphab-$name-g$i-$J.out" s iters caps
    s=$(grep -hoE 'sps=[0-9.]+' "$f" 2>/dev/null | sed 's/sps=//' \
        | awk 'NR>3{n++; sum+=$1} END{if(n>0) printf "%.1f", sum/n; else print "0"}')
    iters=$(grep -cE '^iter ' "$f" 2>/dev/null || echo 0)
    caps=$(grep -c 'captured CUDA graph' "$f" 2>/dev/null || echo 0)
    echo "[graphab] $name gpu$i steady_sps=$s iters=$iters graph_captures=$caps"
    total=$(awk -v t="$total" -v s="$s" 'BEGIN{printf "%.1f", t+s}')
  done
  echo "[graphab] RESULT arm=$name node_sps=$total"
  kill $pids 2>/dev/null || true
  pkill -9 -f "graphab-$name-.*-$J" 2>/dev/null || true
  pkill -9 -f 'ray::' 2>/dev/null || true
  sleep 10
}

run_arm nograph ""
run_arm graph   "--set obs.dino.cuda_graph=true"

echo "[graphab] === SUMMARY ==="
grep -hE '^\[graphab\] RESULT ' slurm_outputs/slurm-zl-graphab-$J.out 2>/dev/null || true
echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
rm -rf "$TMP" 2>/dev/null || true
echo "[graphab] done"
