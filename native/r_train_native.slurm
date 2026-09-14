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
WT=${SLURM_SUBMIT_DIR:-$(pwd)}   # the checkout this was submitted from
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
# The other two per-process caches, sized the same way (2026-09-12 sweep:
# 64 runners x 2 envs hit ~108 GB host RAM per GPU step and filled 24 GB of
# VRAM within an hour; every bound was per process and nothing multiplied
# it by the process count). Rules of thumb for one node:
#   CHUNK_LRU_MB ~= host_RAM_MB / (processes x 10 volume handles)
#   VAO_LRU_MB   ~= (VRAM_MB - ~600 x processes_per_GPU) / processes_per_GPU
# 32x1 FW=2 (this file's default topology) is 96 processes: the 256 MB chunk
# default is ~240 GB worst case on a 376 GB node, fine; the VAO default of
# 2 GB x 32 processes on one 24 GB card is not, so cap it here.
export NGL_NATIVE_CHUNK_LRU_MB=${CHUNK_LRU_MB:-256}
export NGL_NATIVE_VAO_LRU_MB=${VAO_LRU_MB:-256}
# 2D-pane fill policy (env.pane_mode via --pane-mode): atomic (default, the
# 92.5% behaviour) | progressive (measured -11pp, do not use) | concurrent |
# random.
PANE_MODE=${PANE_MODE:-atomic}
# Topology knobs. ENVS_PER_RUNNER=2 is the production default since 2026-09-14:
# a renderer-only probe put 32x2 at 3.3x the sps of 32x1, but under a trained
# policy the second env only hides what little fetch latency remains, and in
# real training it measured +12% (129 -> 144, jobs 885623/885624) and +0.8%
# median / +5% mean (147 -> 148 / 150, job 885618) -- small, positive, free.
# 62-step fragments at batch 4000 are inside the range where fragment length
# was shown irrelevant. RLlib's loop, not the renderer, bounds training at
# this density: the per-iteration 'timers' log line says where the time goes.
RUNNERS=${RUNNERS:-32}
ENVS_PER_RUNNER=${ENVS_PER_RUNNER:-2}
# Step-anchored curriculum (see providers/flywire_skeleton.py).
export CURRICULUM_PROGRESS_FILE="$CKPT/meta.json"

uv run --no-sync python -m ngllib_agent.train \
  --config ${TRAIN_CONFIG:-configs/native.yaml} \
  --run-name "$RUN" --pane-mode "$PANE_MODE" \
  --num-env-runners "$RUNNERS" \
  --num-envs-per-env-runner "$ENVS_PER_RUNNER" \
  --num-gpus-per-env-runner 0.02 \
  --vector threads \
  --iters "$ITERS" \
  --train-batch-size 4000 \
  --checkpoint-dir "$CKPT" --checkpoint-every 10 \
  --wandb-project neurogym-agent \
  --resume
