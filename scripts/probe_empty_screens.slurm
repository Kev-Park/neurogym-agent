#!/bin/bash
#SBATCH --job-name=zl-empty
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=02:00:00
#SBATCH --output=slurm_outputs/zl-empty-%j.out

# Empty-pane / missed-click audit (scripts/probe_empty_screens.py).
# VAO_MB=128 reproduces the TRAINING launchers' GPU mesh slot pool (4 slots);
# unset = the eval/render default (2 GB, 64 slots).
set -eu
cd /scratch/kp0374/wt/neurogym-agent-zmax-left
export PYTHONUNBUFFERED=1
if [ -n "${VAO_MB:-}" ]; then export NGL_NATIVE_VAO_LRU_MB="$VAO_MB"; fi
PKL=${PKL:-/scratch/kp0374/checkpoints/zmaxleft-v7c/ckpt_000740.pkl}
CONFIG=${CONFIG:-configs/native_zmaxleft_v7_eval.yaml}
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
  uv run --no-sync python scripts/probe_empty_screens.py \
    --config "$CONFIG" --state-pkl "$PKL" \
    --holdout /scratch/kp0374/neurogym-agent/eval_zmaxleft_v1.parquet \
    --n "${N:-12}" --out "empty_screens_${SLURM_JOB_ID}.json"
