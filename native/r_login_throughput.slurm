#!/bin/bash
#SBATCH --job-name=loginthru
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
# Sized to BACKFILL: the fleet typically has ~24 idle CPUs per node, and a
# 48-CPU request waits for a full drain instead (job 967819 was quoted a start
# six days out). Chrome 2x16 needs two runner processes, not 48 cores.
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=01:10:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out
#
# V6: does serving the viewer off disk cost throughput?  Chrome at the pinned
# production topology (2 runners x 16 threaded envs), a short run per arm. Every
# context recycle now pulls ~15 assets through the Playwright driver into a
# Python handler instead of Chrome's own network stack, and at 16 envs/process
# that work is GIL-serialized -- so this is measured, not assumed.
#
# BOTH ARMS RUN IN ONE ALLOCATION, SEQUENTIALLY. Submitted as separate jobs they
# landed on different nodes (sarekl16-6 vs 16-1) and the 13% gap that appeared
# was indistinguishable from node variation; run concurrently, two Chrome fleets
# would contend for one GPU instead.
#
#   ITERS=12 RUN_PREFIX=v6ab \
#   REPO=/scratch/kp0374/neurogym-agent WT=/scratch/kp0374/wt/neurogym-agent-login \
#   NGLLIB=/scratch/kp0374/wt/neurogym-login/src sbatch native/r_login_throughput.slurm
set -e
REPO=${REPO:-/scratch/kp0374/neurogym-agent}
WT=${WT:-/scratch/kp0374/wt/neurogym-agent-login}
cd "$REPO"
export PYTHONUNBUFFERED=1
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-44}
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
[ -n "${NGLLIB:-}" ] && export PYTHONPATH="$NGLLIB:$WT/src:${PYTHONPATH:-}"
ITERS=${ITERS:-12}
ARMS=${ARMS:-"packaged:$WT/configs/ppo_zmax_navigate.yaml hosted:$WT/configs/ppo_zmax_navigate_hosted.yaml"}
echo "loginthru: node=$(hostname) iters=$ITERS ngllib=${NGLLIB:-installed}"

for arm in $ARMS; do
  name=${arm%%:*}
  cfg=${arm#*:}
  run=${RUN_PREFIX:-login-thru}-$name
  CKPT=/scratch/kp0374/checkpoints/$run
  mkdir -p "$CKPT"
  export CURRICULUM_PROGRESS_FILE="$CKPT/meta.json"
  echo
  echo "=== ARM $name  config=$cfg ==="
  grep -n 'viewer' "$cfg" | head -2 || true
  uv run --no-sync python -m ngllib_agent.train \
    --config "$cfg" \
    --run-name "$run" \
    --num-env-runners 2 \
    --num-envs-per-env-runner 16 \
    --num-gpus-per-env-runner 0.5 \
    --vector threads \
    --iters "$ITERS" \
    --train-batch-size 4000 \
    --checkpoint-dir "$CKPT" --checkpoint-every 100 \
    --wandb-mode disabled
  echo "ARM-DONE $name"
done
echo "LOGINTHRU-DONE"
