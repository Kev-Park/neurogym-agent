#!/bin/bash
#SBATCH --job-name=evalvid
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# Rollout videos for a state-pickle checkpoint, rendered in CHROME (the
# deployment target) so frames show real Neuroglancer. Worktree variant of
# ../r_eval_video.slurm, which cds into the main repo instead.
#   sbatch native/r_eval_video_native.slurm <ckpt.pkl> <out-subdir> [extra]
set -e
cd /scratch/kp0374/wt/neurogym-agent-native
export PYTHONUNBUFFERED=1
PKL=${1:?state pickle path}
OUT=${2:?output subdir}
shift 2
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python scripts/eval_video.py \
  --config configs/ppo_zmax_navigate.yaml \
  --eval-d0 /scratch/kp0374/neurogym-agent/eval_d0_v1.parquet \
  --skeleton /scratch/kp0374/neurogym-agent/segment_positions.parquet \
  --state-pkl "$PKL" --out-dir "/scratch/kp0374/native_spike/eval_videos/$OUT" "$@"
