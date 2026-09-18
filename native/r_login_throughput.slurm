#!/bin/bash
#SBATCH --job-name=loginthru
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=110G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out
#
# V6: does serving the viewer off disk cost throughput?  Chrome at the pinned
# production topology (2 runners x 16 threaded envs), short run, one arm per
# viewer source. Every context recycle now reads ~3.8 MB through the Playwright
# driver instead of Chrome's network stack + HTTP cache, so this is measured,
# not assumed.
#
#   CONFIG=<yaml> RUN=<name> ITERS=12 \
#   REPO=/scratch/kp0374/neurogym-agent WT=/scratch/kp0374/wt/neurogym-agent-login \
#   NGLLIB=/scratch/kp0374/wt/neurogym-login/src sbatch native/r_login_throughput.slurm
set -e
REPO=${REPO:-/scratch/kp0374/neurogym-agent}
WT=${WT:-/scratch/kp0374/wt/neurogym-agent-login}
cd "$REPO"
export PYTHONUNBUFFERED=1
export RAY_NUM_CPUS=44
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
[ -n "${NGLLIB:-}" ] && export PYTHONPATH="$NGLLIB:$WT/src:${PYTHONPATH:-}"
CONFIG=${CONFIG:-$WT/configs/ppo_zmax_navigate.yaml}
RUN=${RUN:-login-thru}
ITERS=${ITERS:-12}
CKPT=/scratch/kp0374/checkpoints/${RUN}
mkdir -p "$CKPT"
export CURRICULUM_PROGRESS_FILE="$CKPT/meta.json"
echo "loginthru: node=$(hostname) run=$RUN config=$CONFIG iters=$ITERS ngllib=${NGLLIB:-installed}"
grep -n 'viewer' "$CONFIG" | head -3 || true

uv run --no-sync python -m ngllib_agent.train \
  --config "$CONFIG" \
  --run-name "$RUN" \
  --num-env-runners 2 \
  --num-envs-per-env-runner 16 \
  --num-gpus-per-env-runner 0.5 \
  --vector threads \
  --iters "$ITERS" \
  --train-batch-size 4000 \
  --checkpoint-dir "$CKPT" --checkpoint-every 100 \
  --wandb-mode disabled
echo "LOGINTHRU-DONE $RUN"
