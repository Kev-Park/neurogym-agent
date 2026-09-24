#!/bin/bash
#SBATCH --job-name=loginthru
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
# MUST match the baseline arm's resources (job 923949: 48 cpus / 110G) or the
# comparison measures starvation, not the viewer. Sizing down to 20 cpus / 48G
# to chase backfill gave 15-22 sps against a 34-43 baseline and then an
# OUT_OF_MEMORY kill (job 967927): 32 Chromium instances plus DINO do not fit
# in 48G. Queue longer instead -- and see MEMORY: request RAM toward node max.
#SBATCH --cpus-per-task=48
#SBATCH --mem=110G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=02:30:00
# preempt partition: job 967905 was preempted 27s in and died because this
# wrapper lacked --requeue (the production training scripts have it).
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out
#
# V6: does serving the viewer locally cost throughput?  Chrome at the pinned
# production topology (2 runners x 16 threaded envs), a short run per arm.
# Serving through page.route made every viewer asset a Python callback on the
# step loop's own driver connection; the loopback server (default since
# 2026-09-24) removes that, and this says what it is worth at training scale.
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
set -o pipefail
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
  # Per-arm log, overwritten per attempt. --requeue with --open-mode=append
  # turned job 967927's single log into an interleave of partial passes that no
  # summarizer could read; scripts/summarize_thru.sh reads these instead.
  ARMLOG=/scratch/kp0374/native_spike/thru_${run}.log
  echo
  echo "=== ARM $name  config=$cfg  log=$ARMLOG ==="
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
    --wandb-mode disabled 2>&1 | tee "$ARMLOG"
  echo "ARM-DONE $name -> $ARMLOG"
done
echo "LOGINTHRU-DONE"
