#!/bin/bash
#SBATCH --job-name=nativetrain
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=44
#SBATCH --mem=200G
# EXCLUDE, not --nodelist (2026-08-31): --gres=gpu:3090:1 already confines
# this to the sarekl15/16 fleet, so excluding the Mesa-only and
# behaviorally-bad nodes leaves exactly the good pool AND lets backfill
# take whichever frees first. Pinning a nodelist while those nodes are
# reserved for a higher-priority partition gave job 871493 a StartTime
# estimate NINE DAYS out (CLAUDE.md "Backfill pitfall").
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
# ~4h at 170-210 sps for 740 iters; over-requesting deprioritizes the job
# and keeps it out of short backfill windows.
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# v9-identical training on the NATIVE renderer — single node, NO coordinator.
# The coordinator existed to manage multi-node salloc + respawn for the
# browser fleet; a single-node native run needs only SLURM --requeue +
# train.py --resume.
#
# Topology from the density sweeps (jobs 870732..741): per-step Python is
# GIL-serialized, so ONE process caps ~10 sps at any thread count; the
# full K x M allocation grid at 32 envs picked FULL process isolation:
#   8x4 48.1 | 16x2 96.8 | 22x2 98.8 | 32x1 116.5 sps/node (browser
# ~35-40), 15.3G VRAM peak (learner is CPU-in-driver). Hence 32 runner
# processes x 1 env, 1 fetch worker each.
#
# Usage: sbatch native/r_train_native.slurm [run-name] [target-iters]
set -e
WT=/scratch/kp0374/wt/neurogym-agent-native
cd "$WT"
RUN=${1:-native-v9-test}
ITERS=${2:-740}
CKPT=/scratch/kp0374/checkpoints/${RUN}
mkdir -p "$CKPT"

export PYTHONUNBUFFERED=1
export RAY_NUM_CPUS=44
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
# 2 workers: reset-ahead mesh prefetch (~30s tasks) must not starve the
# per-click tile groups.
# FETCH_WORKERS=1 reproduces v9-test (it ran 1/runner; 2 landed 08-28 03:08).
export NGL_NATIVE_FETCH_WORKERS=${FETCH_WORKERS:-2}
# MeshStore decoded-mesh LRU is PER STORE and local mode builds one per env:
# 32 envs x 128MB = 4G. Unbounded (pre-2026-08-31) this reached 206 GiB RSS
# and the cgroup killed job 871069 at iteration 410 of 740 with no traceback.
export NGL_NATIVE_MESH_LRU_MB=128
# 2D-pane fill policy: atomic (default, the 92.5% behaviour) |
# progressive (measured -11pp, do not use) | concurrent | random.
export NGL_NATIVE_PANE_MODE=${PANE_MODE:-atomic}
# Step-anchored curriculum (see providers/flywire_skeleton.py).
export CURRICULUM_PROGRESS_FILE="$CKPT/meta.json"

uv run --no-sync python -m ngllib_agent.train \
  --config ${TRAIN_CONFIG:-configs/native.yaml} \
  --run-name "$RUN" \
  --num-env-runners 32 \
  --num-envs-per-env-runner 1 \
  --num-gpus-per-env-runner 0.02 \
  --vector threads \
  --iters "$ITERS" \
  --train-batch-size 4000 \
  --checkpoint-dir "$CKPT" --checkpoint-every 10 \
  --wandb-project neurogym-agent \
  --resume
