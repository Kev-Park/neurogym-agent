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
# Topology from the density sweeps (jobs 870732/34/37/38/39): per-step
# Python is GIL-serialized, so ONE process caps ~10 sps at any thread
# count; process count is the scaling axis (K=4 -> 28.8, K=8 -> 48.1
# sps/node vs browser ~35-40). Hence 8 EnvRunner processes x 4 threaded
# envs (32 envs/node), small per-runner fetch pools.
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
export NGL_NATIVE_FETCH_WORKERS=3
# Wall-clock spawn-curriculum anchor must survive preemption requeues: first
# launch stamps it into the checkpoint dir; requeues reuse the stamp so the
# anneal doesn't replay the easy phase.
T0_FILE="$CKPT/curriculum_t0"
[ -f "$T0_FILE" ] || date +%s > "$T0_FILE"
export CURRICULUM_T0=$(cat "$T0_FILE")

uv run --no-sync python -m ngllib_agent.train \
  --config configs/native.yaml \
  --run-name "$RUN" \
  --num-env-runners 8 \
  --num-envs-per-env-runner 4 \
  --num-gpus-per-env-runner 0.1 \
  --vector threads \
  --iters "$ITERS" \
  --train-batch-size 4000 \
  --checkpoint-dir "$CKPT" --checkpoint-every 10 \
  --wandb-project neurogym-agent \
  --resume
