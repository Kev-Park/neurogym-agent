#!/bin/bash
#SBATCH --job-name=selparity
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Vulkan-good nodes MINUS the ones the running experiment holds; the parity
# probe must not contend with it.
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# Double-click (NG `select`) I/O parity, both backends in one job: the browser
# pass writes ground truth, the simulator pass replays it and scores agreement.
# Sequential, not concurrent — Chrome and the EGL context share the GPU.
#
# REUSE_JSONL=<path> skips the browser pass and replays existing ground truth.
# Chrome's answers do not depend on the simulator, so iterating on the pick
# only needs the (fast) native pass.
set -e
WT=/scratch/kp0374/wt/neurogym-agent-native
cd "$WT"
export PYTHONUNBUFFERED=1
FRAMES=/scratch/kp0374/native_spike/select_frames_$SLURM_JOB_ID
N=${N_STATES:-8}

if [ -n "$REUSE_JSONL" ]; then
  OUT="$REUSE_JSONL"
else
  OUT=/scratch/kp0374/native_spike/select_browser_$SLURM_JOB_ID.jsonl

  # What coordinate system is mouse_xy actually in? Read it off the DOM.
  uv run --no-sync python native/probe_select_parity.py --mode rects \
    --config configs/native_select.yaml \
    --pool /scratch/kp0374/neurogym-agent/eval_d0_v1.parquet --n-states 1

  uv run --no-sync python native/probe_select_parity.py --mode browser \
    --config configs/native_select.yaml \
    --pool /scratch/kp0374/neurogym-agent/eval_d0_v1.parquet \
    --n-states "$N" --out "$OUT" --frame-dir "$FRAMES"
fi

# Does the SHOW_ALL pane render at all, independent of streaming?
uv run --no-sync python native/probe_show_all.py \
  --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 3 \
  --out-dir /scratch/kp0374/native_spike/show_all_$SLURM_JOB_ID || true

uv run --no-sync python native/probe_select_parity.py --mode native \
  --config configs/native_select.yaml --browser-jsonl "$OUT" --diagnose \
  --frame-dir "$FRAMES"
