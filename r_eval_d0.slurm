#!/bin/bash
#SBATCH --job-name=evald0
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
# Known-good vulkan fleet minus the operational exclusions (15-5 CUDA-wedged,
# 15-8 flaky, 16-2 hang factory — CLAUDE.md "Operationally excluded").
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-7,sarekl16-1,sarekl16-3,sarekl16-5,sarekl16-6,sarekl16-7,sarekl16-8
#SBATCH --time=02:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=code@thiskevin.com
#SBATCH --output=slurm_outputs/evald0-%x-%j.out

# Frozen-d0 policy eval (REFINEMENT R9 follow-up: score coord-test-v7).
#   sbatch --job-name=evald0-v7  r_eval_d0.slurm pkl /path/to/ckpt.pkl v7_ckpt370
#   sbatch --job-name=evald0-rnd r_eval_d0.slurm random - random_baseline
# $1 = pkl|random, $2 = state pickle path (ignored for random), $3 = result tag.
# Runs a 3-pair smoke first so a policy-adapter failure aborts in minutes,
# then the full pool. Single env, sequential pairs (~20-35s/pair at
# capture_scale 0.5), 200 pairs ~= 1-2h.
set -e
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1

MODE=${1:?pkl|random}
PKL=${2:-}
TAG=${3:?result tag}
D0=eval_d0_v1.parquet
SKEL=segment_positions.parquet
mkdir -p eval_results

if [ "$MODE" = "random" ]; then
    POLICY_ARGS=(--random-policy)
else
    POLICY_ARGS=(--state-pkl "$PKL")
fi

echo "==== SMOKE (3 pairs) ===="
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python scripts/eval_d0.py \
    --eval-d0 "$D0" --skeleton "$SKEL" --obs dino --limit 3 \
    --output "eval_results/${TAG}_smoke.json" "${POLICY_ARGS[@]}"

echo "==== FULL POOL ===="
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python scripts/eval_d0.py \
    --eval-d0 "$D0" --skeleton "$SKEL" --obs dino \
    --output "eval_results/${TAG}.json" "${POLICY_ARGS[@]}"
