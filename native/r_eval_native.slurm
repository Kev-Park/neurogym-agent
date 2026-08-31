#!/bin/bash
#SBATCH --job-name=nativeeval
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Good vulkan/EGL nodes; exclude-style co-tenancy with running experiments.
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-7,sarekl16-1,sarekl16-3,sarekl16-5
# Smoke measured ~0.3-0.5 s/step cold-cache (clicks refetch EM/label tiles);
# 200 pairs x <=600 steps worst-cases past 4h, so give slack.
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# Stage D: frozen-policy eval on the NATIVE backend (configs/native_eval.yaml)
# via the standard eval_d0 harness — numbers compare 1:1 against the same
# checkpoint's browser-backend eval.
#
# Usage: sbatch r_eval_native.slurm <state-pkl> [extra eval_d0 args...]
# Env: EVAL_CONFIG picks the config -- right-pane-only policies need
# configs/native_v9test_noleft.yaml to match their observation space.
set -e
WT=/scratch/kp0374/wt/neurogym-agent-native
cd "$WT"
export PYTHONUNBUFFERED=1
PKL=$1; shift || true
uv run --no-sync python scripts/eval_d0.py \
  --config ${EVAL_CONFIG:-configs/native.yaml} \
  --eval-d0 /scratch/kp0374/neurogym-agent/eval_d0_v1.parquet \
  --skeleton /scratch/kp0374/neurogym-agent/segment_positions.parquet \
  --state-pkl "$PKL" \
  --obs dino \
  --output /scratch/kp0374/native_spike/eval_native_$SLURM_JOB_ID.json \
  "$@"
