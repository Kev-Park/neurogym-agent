#!/bin/bash
#SBATCH --job-name=evalvid
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-7,sarekl16-1,sarekl16-3,sarekl16-5,sarekl16-6,sarekl16-7,sarekl16-8
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=code@thiskevin.com
#SBATCH --output=slurm_outputs/evalvid-%j.out

# Rollout videos per length quartile (success + failure each) for a
# state-pickle checkpoint, stochastic protocol (REFINEMENT R11).
#   sbatch r_eval_video.slurm /path/to/ckpt.pkl out_subdir
set -e
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1

PKL=${1:?state pickle path}
OUT=${2:?output subdir under eval_videos/}

UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python scripts/eval_video.py \
    --eval-d0 eval_d0_v1.parquet --skeleton segment_positions.parquet \
    --state-pkl "$PKL" --out-dir "eval_videos/$OUT"
