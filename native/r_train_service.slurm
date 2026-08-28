#!/bin/bash
#SBATCH --job-name=svctrain
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=44
#SBATCH --mem=200G
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-7,sarekl16-1,sarekl16-3,sarekl16-5
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# v9-identical training on the SERVICE architecture (per-node render+encode
# actor, envs as state-machine clients). Topology: clients are IO-bound
# (RPC waits), so few fat runners: 4 x 16 threaded envs = 64 clients
# (probe optimum), runners need NO GPU. Learner on GPU: only 2 CUDA
# contexts on the card now (service DINO + learner), so the 33-context
# arbitration that made --learner-gpu slower does not apply.
#
# Usage: sbatch native/r_train_service.slurm [run-name] [target-iters]
set -e
WT=/scratch/kp0374/wt/neurogym-agent-native
cd "$WT"
RUN=${1:-native-svc-v9}
ITERS=${2:-740}
CKPT=/scratch/kp0374/checkpoints/${RUN}
mkdir -p "$CKPT"

export PYTHONUNBUFFERED=1
export RAY_NUM_CPUS=44
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
T0_FILE="$CKPT/curriculum_t0"
[ -f "$T0_FILE" ] || date +%s > "$T0_FILE"
export CURRICULUM_T0=$(cat "$T0_FILE")

uv run --no-sync python -m ngllib_agent.train \
  --config configs/native_service.yaml \
  --run-name "$RUN" \
  --render-service \
  --num-env-runners 4 \
  --num-envs-per-env-runner 16 \
  --num-gpus-per-env-runner 0 \
  --learner-gpu \
  --vector threads \
  --iters "$ITERS" \
  --train-batch-size 4000 \
  --checkpoint-dir "$CKPT" --checkpoint-every 10 \
  --wandb-project neurogym-agent \
  --resume
