#!/bin/bash
#SBATCH --job-name=zl-confirm
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:3090:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=370G
#SBATCH --time=00:45:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/slurm-%x-%j.out
#
# Production-batch confirmation of the winning co-pane layout for the headline
# SOTA sps. Runs the chosen (G GPUs x N runners/GPU) at TOTAL ENVS<=48 (RAM wall)
# with the chosen levers, warm-started from the v5 ckpt, at the production batch,
# for a LONG window so the steady sps is stable (mean of iters after the first 3).
#
# Set via env (defaults = the RAM-neutral spread baseline + graphs):
#   GPUS=<G>  NPG=<runners/GPU>  BATCH=16000  WINDOW=600
#   EXTRA="--set obs.dino.cuda_graph=true"   # winning safe levers
# Usage: GPUS=2 NPG=12 EXTRA="--set obs.dino.cuda_graph=true" \
#        sbatch native/r_confirm_zmaxleft_sota.slurm [state-pkl]
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_outputs
J=$SLURM_JOB_ID
PKL=${1:-/scratch/kp0374/checkpoints/zmaxleft-v5/ckpt_000490.pkl}
G=${GPUS:-1}
NPG=${NPG:-24}
BATCH=${BATCH:-16000}
WINDOW=${WINDOW:-600}
EXTRA=${EXTRA:-}
TMP=/tmp/zlconfirm_$J

export PYTHONUNBUFFERED=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_memory_monitor_refresh_ms=0
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_MESH_LRU_MB=128 NGL_NATIVE_CHUNK_LRU_MB=256 NGL_NATIVE_VAO_LRU_MB=128
export NGL_NATIVE_MESH_PREFETCH=4 NGL_NATIVE_FETCH_WORKERS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d && echo "[confirm] MPS up $(hostname) job=$J G=$G NPG=$NPG batch=$BATCH extra=[$EXTRA]"

pids=""
for ((i=0; i<G; i++)); do
  mkdir -p "$TMP/g$i"; cp -f "$PKL" "$TMP/g$i/ckpt_000490.pkl"
  env CUDA_VISIBLE_DEVICES=$i UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
    NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J.lock \
    uv run --no-sync python -m ngllib_agent.train \
      --config configs/native_zmaxleft_v5.yaml --run-name "confirm-g$i-$J" \
      --obs dino --vector threads \
      --num-env-runners "$NPG" --num-envs-per-env-runner 2 \
      --num-gpus-per-env-runner 0.01 --num-cpus-per-env-runner 1.0 \
      --no-spawn-curriculum --no-degraded-exit --stagger-first-episode \
      --iters 999999 --train-batch-size "$BATCH" \
      --checkpoint-dir "$TMP/g$i" --checkpoint-every 999999 --resume \
      --wandb-mode disabled $EXTRA > "slurm_outputs/confirm-g$i-$J.out" 2>&1 &
  pids="$pids $!"
done
echo "[confirm] $G trainings launched (total runners $((G*NPG)), envs $((G*NPG*2))), window ${WINDOW}s"
sleep "$WINDOW"

total=0
for ((i=0; i<G; i++)); do
  f="slurm_outputs/confirm-g$i-$J.out"
  s=$(grep -hoE 'sps=[0-9.]+' "$f" 2>/dev/null | sed 's/sps=//' \
      | awk 'NR>3{n++; sum+=$1} END{if(n>0) printf "%.1f", sum/n; else print "0"}')
  iters=$(grep -cE '^iter ' "$f" 2>/dev/null || echo 0)
  caps=$(grep -c 'captured CUDA graph' "$f" 2>/dev/null || echo 0)
  echo "[confirm] gpu$i steady_sps=$s iters=$iters graph_captures=$caps"
  total=$(awk -v t="$total" -v s="$s" 'BEGIN{printf "%.1f", t+s}')
done
echo "[confirm] SOTA node_sps=$total  (G=$G NPG=$NPG batch=$BATCH extra=[$EXTRA])"
kill $pids 2>/dev/null || true
pkill -9 -f "confirm-.*-$J" 2>/dev/null || true
echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
rm -rf "$TMP" 2>/dev/null || true
echo "[confirm] done"
