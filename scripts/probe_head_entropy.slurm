#!/bin/bash
#SBATCH --job-name=zl-heads
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=01:00:00
#SBATCH --output=slurm_outputs/zl-heads-%j.out

# Per-head entropy of v7-family checkpoints on identical states
# (scripts/probe_head_entropy.py). States come from the FIRST label's policy.
set -eu
cd /scratch/kp0374/wt/neurogym-agent-zmax-left
export PYTHONUNBUFFERED=1
C=/scratch/kp0374/checkpoints
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
  uv run --no-sync python scripts/probe_head_entropy.py \
    --config configs/native_zmaxleft_v7_eval.yaml \
    --holdout /scratch/kp0374/neurogym-agent/eval_zmaxleft_v1.parquet \
    --label v7=$C/zmaxleft-v7/ckpt_000450.pkl \
    --label v7b=$C/zmaxleft-v7b/ckpt_000800.pkl \
    --label v7c=$C/zmaxleft-v7c/ckpt_000800.pkl \
    --label A_v7d=$C/zmaxleft-v7d/ckpt_000800.pkl \
    --label B_v7e=$C/zmaxleft-v7e/ckpt_${B_CKPT:-000800}.pkl
