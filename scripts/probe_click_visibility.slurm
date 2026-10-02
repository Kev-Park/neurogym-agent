#!/bin/bash
#SBATCH --job-name=zl-clickvis
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
#SBATCH --output=slurm_outputs/zl-clickvis-%j.out

# Click-visibility audit (scripts/probe_click_visibility.py): can a 3D-pane
# right-click hit content the policy could not see? PKL/CONFIG literal at submit.
set -eu
cd /scratch/kp0374/wt/neurogym-agent-zmax-left
export PYTHONUNBUFFERED=1
PKL=${PKL:-/scratch/kp0374/checkpoints/zmaxleft-v7/ckpt_000450.pkl}
CONFIG=${CONFIG:-configs/native_zmaxleft_v7_eval.yaml}
N=${N:-12}
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
  uv run --no-sync python scripts/probe_click_visibility.py \
    --config "$CONFIG" --state-pkl "$PKL" \
    --holdout /scratch/kp0374/neurogym-agent/eval_zmaxleft_v1.parquet \
    --n "$N" --out "click_visibility_${SLURM_JOB_ID}.json"
