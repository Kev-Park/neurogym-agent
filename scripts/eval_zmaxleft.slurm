#!/bin/bash
#SBATCH --job-name=zl-eval
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=04:00:00
#SBATCH --output=slurm_outputs/zl-eval-%j.out

# zmax-left holdout eval (paired delta-z protocol; see scripts/eval_zmaxleft.py).
# Submit with literal exports (bridge rule: no shell vars through wsl->ssh):
#   sbatch --export=ALL,POLICY_ARGS='--random-policy' scripts/eval_zmaxleft.slurm
#   sbatch --export=ALL,POLICY_ARGS='--state-pkl /scratch/kp0374/checkpoints/zmaxleft-v1/ckpt_200.pkl --stochastic' \
#          scripts/eval_zmaxleft.slurm
set -eu
cd /scratch/kp0374/wt/neurogym-agent-zmax-left
export PYTHONUNBUFFERED=1
HOLDOUT=${HOLDOUT:-/scratch/kp0374/neurogym-agent/eval_zmaxleft_v1.parquet}
OUTPUT=${OUTPUT:-eval_zmaxleft_${SLURM_JOB_ID}.json}
POLICY_ARGS=${POLICY_ARGS:---random-policy}

echo "=== node: $(hostname) policy: $POLICY_ARGS out: $OUTPUT ==="
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
  uv run --no-sync python scripts/eval_zmaxleft.py \
    --holdout "$HOLDOUT" --output "$OUTPUT" $POLICY_ARGS
