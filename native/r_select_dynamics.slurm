#!/bin/bash
#SBATCH --job-name=seldyn
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# Selection DYNAMICS: not just the settled frame but the frames in between.
# Both passes write into the same frame directory so the native pass can score
# frame-by-frame SSIM against the browser at matching step indices.
set -e
WT=${SLURM_SUBMIT_DIR:-$(pwd)}   # the checkout this was submitted from
cd "$WT"
export PYTHONUNBUFFERED=1
OUT=/scratch/kp0374/native_spike/dyn_browser_$SLURM_JOB_ID.jsonl
# FRAME_DIR=none skips the PNG dump, for long windows where the frame count
# would be in the thousands and only the timing numbers are wanted.
if [ "$FRAME_DIR" = "none" ]; then
  FRAMEARG=""
else
  FRAMEARG="--frame-dir /scratch/kp0374/native_spike/dyn_frames_$SLURM_JOB_ID"
fi
N=${N_STATES:-4}
STEPS=${STEPS:-8}
OBS=${OBS:-raw}

uv run --no-sync python native/probe_select_dynamics.py --mode browser \
  --config configs/native_select.yaml \
  --pool /scratch/kp0374/neurogym-agent/eval_d0_v1.parquet \
  --n-states "$N" --steps "$STEPS" --obs "$OBS" --out "$OUT" $FRAMEARG

uv run --no-sync python native/probe_select_dynamics.py --mode native \
  --config configs/native_select.yaml --browser-jsonl "$OUT" \
  --steps "$STEPS" --obs "$OBS" $FRAMEARG
