#!/bin/bash
#SBATCH --job-name=evalbshard
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%A_%a.out
#
# Sharded BROWSER eval: 8 x 25 pairs. One long browser eval loses the whole
# run to an unkillable playwright greenlet wedge (a watchdog kill followed
# by a browser restart hangs the main thread past SIGALRM's reach — two
# 200-pair runs died at pair ~37 on 2026-08-28). Sharding bounds the blast
# radius to 25 pairs, retries cheaply, and finishes ~8x sooner in
# wall-clock. Merge with native/merge_eval_shards.py.
#
# Usage: sbatch native/r_eval_browser_shards.slurm <state-pkl> <out-prefix>
set -e
WT=/scratch/kp0374/wt/neurogym-agent-native
cd "$WT"
export PYTHONUNBUFFERED=1
PKL=$1
PREFIX=$2
# Gap-filling reruns use smaller shards from a base offset, e.g.
#   sbatch --array=0-4 --export=ALL,SHARD_SIZE=5,OFFSET_BASE=25 ...
SHARD_SIZE=${SHARD_SIZE:-25}
OFFSET_BASE=${OFFSET_BASE:-0}
OFFSET=$((OFFSET_BASE + SLURM_ARRAY_TASK_ID * SHARD_SIZE))

uv run --no-sync python scripts/eval_d0.py \
  --config configs/ppo_zmax_navigate.yaml \
  --eval-d0 /scratch/kp0374/neurogym-agent/eval_d0_v1.parquet \
  --skeleton /scratch/kp0374/neurogym-agent/segment_positions.parquet \
  --state-pkl "$PKL" \
  --obs dino --stochastic \
  --offset "$OFFSET" --limit "$SHARD_SIZE" \
  --output "${PREFIX}_shard_off${OFFSET}.json"
