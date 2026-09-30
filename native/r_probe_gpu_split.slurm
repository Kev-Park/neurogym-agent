#!/bin/bash
#SBATCH --job-name=zl-gpusplit
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=00:20:00
#SBATCH --output=slurm_outputs/slurm-%x-%j.out
#
# Task 1: attribute the co-pane GPU time (RENDER vs DINO) at a realistic mesh
# load, single env / single process (no MPS) => the clean per-env FLOP split.
# Drives the trained v5 policy so mesh accumulation matches training.
#
# Usage: sbatch native/r_probe_gpu_split.slurm [state-pkl]
set -eu
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_outputs
PKL=${1:-/scratch/kp0374/checkpoints/zmaxleft-v5/ckpt_000490.pkl}

export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
# Same renderer LRU budgets as the v3/v5 training launcher so the render cost
# measured here matches production (cache-miss rate drives per-move decode).
export NGL_NATIVE_FETCH_WORKERS=2
export NGL_NATIVE_MESH_LRU_MB=128
export NGL_NATIVE_CHUNK_LRU_MB=256
export NGL_NATIVE_VAO_LRU_MB=128
export NGL_NATIVE_MESH_PREFETCH=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[gpu-split] node=$(hostname) pkl=$PKL"
uv run --no-sync python native/probe_gpu_split.py \
  --config configs/native_zmaxleft_v5.yaml \
  --state-pkl "$PKL" \
  --warmup 150 --measure 250
