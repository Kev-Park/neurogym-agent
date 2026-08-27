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
# train.py --resume. Topology mirrors one production node: 2 EnvRunner
# processes x 16 threaded envs sharing the GPU with the learner-in-driver.
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
# Wall-clock spawn-curriculum anchor must survive preemption requeues: first
# launch stamps it into the checkpoint dir; requeues reuse the stamp so the
# anneal doesn't replay the easy phase.
T0_FILE="$CKPT/curriculum_t0"
[ -f "$T0_FILE" ] || date +%s > "$T0_FILE"
export CURRICULUM_T0=$(cat "$T0_FILE")

uv run --no-sync python -m ngllib_agent.train \
  --config configs/native.yaml \
  --run-name "$RUN" \
  --num-env-runners 2 \
  --num-envs-per-env-runner 16 \
  --num-gpus-per-env-runner 0.5 \
  --vector threads \
  --iters "$ITERS" \
  --train-batch-size 4000 \
  --checkpoint-dir "$CKPT" --checkpoint-every 10 \
  --wandb-project neurogym-agent \
  --resume
