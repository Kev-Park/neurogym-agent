#!/bin/bash
#SBATCH --job-name=gate6
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=110G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# Gate 6 (renderer_seam_plan.md section 9): a short training run per backend,
# sps and return curve compared pre/post refactor. Submit the SAME file from
# the pre-seam checkout (wt/neurogym-agent-native) and the refactor checkout
# (wt/neurogym-agent-refactor); it runs from whichever it was submitted from,
# so the two arms differ only in the code under test.
#
#   BACKEND=simulator  reproduces native-v9-pace740's recipe exactly:
#                      configs/native.yaml, 32 runners x 1 env, batch 4000,
#                      0.02 GPU/runner, FETCH_WORKERS=2, MESH_LRU 128 MB,
#                      atomic pane mode. pace740 logged 146-175 sps per iter.
#   BACKEND=chrome     configs/ppo_zmax_navigate.yaml at the pinned browser
#                      topology, 2 runners x 16 threaded envs (~35-40 sps/node).
#
#   BACKEND=simulator sbatch native/r_train_gate6.slurm <run-name> [iters=50]
set -e
WT=${SLURM_SUBMIT_DIR:-$(pwd)}   # the checkout this was submitted from
cd "$WT"
RUN=${1:?run-name}
ITERS=${2:-50}
BACKEND=${BACKEND:-simulator}
CKPT=/scratch/kp0374/checkpoints/${RUN}
mkdir -p "$CKPT"

export PYTHONUNBUFFERED=1
export RAY_NUM_CPUS=44
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NGL_NATIVE_FETCH_WORKERS=${FETCH_WORKERS:-2}
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
# The pre-seam checkout reads the pane mode from this variable; the refactor
# reads env.pane_mode (default atomic). Both arms therefore run atomic.
export NGL_NATIVE_PANE_MODE=atomic
export CURRICULUM_PROGRESS_FILE="$CKPT/meta.json"
echo "gate6 arm: checkout=$WT backend=$BACKEND commit=$(git rev-parse --short HEAD) iters=$ITERS"

if [ "$BACKEND" = "simulator" ]; then
  CFG=configs/native.yaml; RUNNERS=32; ENVS=1; GPU=0.02
else
  CFG=configs/ppo_zmax_navigate.yaml; RUNNERS=2; ENVS=16; GPU=0.5
fi

uv run --no-sync python -m ngllib_agent.train \
  --config "$CFG" \
  --run-name "$RUN" \
  --num-env-runners "$RUNNERS" \
  --num-envs-per-env-runner "$ENVS" \
  --num-gpus-per-env-runner "$GPU" \
  --vector threads \
  --iters "$ITERS" \
  --train-batch-size 4000 \
  --checkpoint-dir "$CKPT" --checkpoint-every 10 \
  --wandb-mode disabled \
  --resume
echo "GATE6-DONE $BACKEND $(git rev-parse --short HEAD)"
