#!/bin/bash
#SBATCH --job-name=nativetrain
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=44
#SBATCH --mem=200G
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-7,sarekl16-1,sarekl16-3,sarekl16-5
#SBATCH --time=24:00:00
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
export NGL_NATIVE_FETCH_WORKERS=1
# Wall-clock spawn-curriculum anchor must survive preemption requeues: first
# launch stamps it into the checkpoint dir; requeues reuse the stamp so the
# anneal doesn't replay the easy phase.
T0_FILE="$CKPT/curriculum_t0"
[ -f "$T0_FILE" ] || date +%s > "$T0_FILE"
export CURRICULUM_T0=$(cat "$T0_FILE")

uv run --no-sync python -m ngllib_agent.train \
  --config configs/native.yaml \
  --run-name "$RUN" \
  --num-env-runners 32 \
  --num-envs-per-env-runner 1 \
  --num-gpus-per-env-runner 0.02 \
  --learner-gpu \
  --vector threads \
  --iters "$ITERS" \
  --train-batch-size 4000 \
  --checkpoint-dir "$CKPT" --checkpoint-every 10 \
  --wandb-project neurogym-agent \
  --resume
