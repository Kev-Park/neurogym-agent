#!/bin/bash
#SBATCH --job-name=zl-video
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=01:30:00
#SBATCH --output=slurm_outputs/zl-video-%j.out

# Render zmax-left rollout MP4s for visual validation (6 holdout states by
# headroom strata). PKL is passed literally at submit time (bridge rule).
set -eu
cd /scratch/kp0374/wt/neurogym-agent-zmax-left
export PYTHONUNBUFFERED=1
PKL=${PKL:-/scratch/kp0374/checkpoints/zmaxleft-v2/ckpt_000200.pkl}

echo "=== node: $(hostname) pkl: $PKL ==="
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
  uv run --no-sync python scripts/eval_zmaxleft_video.py \
    --holdout /scratch/kp0374/neurogym-agent/eval_zmaxleft_v1.parquet \
    --state-pkl "$PKL" --out-dir eval_videos_zmaxleft
