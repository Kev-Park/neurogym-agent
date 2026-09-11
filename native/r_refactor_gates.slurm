#!/bin/bash
#SBATCH --job-name=seamgates
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

# Renderer-seam validation gates 3-5 (renderer_seam_plan.md section 9), one
# GPU, sequential -- Chrome and the EGL context share it. Runs from the
# checkout it was submitted from.
#
#   gate 3  both backends build + step; identical obs/action spaces; the
#           shared zoom rule agrees                (probe_smoke_both)
#           full-stack DINO eval smoke, 3 pairs, both backends (eval_d0)
#   gate 4  input parity, Chrome ground truth -> simulator replay: 72/72
#           (probe_select_parity), plus the zoom-boundary measurement the
#           plan still owes                        (probe_zoom_boundary)
#   gate 5  pixel parity against the recorded baselines and the Chrome
#           self-consistency ceiling               (probe_visual_audit,
#                                                   probe_pane3d_parity)
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1
export NGL_MASK_UI=${NGL_MASK_UI:-1}     # same state on both sides (gate 5 note)
OUT=/scratch/kp0374/native_spike
POOL=/scratch/kp0374/neurogym-agent/eval_d0_v1.parquet
SKEL=/scratch/kp0374/neurogym-agent/segment_positions.parquet
rc=0
section() { echo; echo "################ $1 ################"; }

section "GATE 3a: smoke both backends"
uv run --no-sync python native/probe_smoke_both.py --config configs/native_select.yaml || rc=1

section "GATE 3b: eval_d0 DINO smoke, simulator"
uv run --no-sync python scripts/eval_d0.py --config configs/native.yaml \
  --eval-d0 "$POOL" --skeleton "$SKEL" --random-policy --obs dino \
  --limit 3 --max-steps 40 --report-budgets 40 \
  --output "$OUT/seam_smoke_sim_$SLURM_JOB_ID.json" || rc=1

section "GATE 3c: eval_d0 DINO smoke, Chrome"
uv run --no-sync python scripts/eval_d0.py --config configs/ppo_zmax_navigate.yaml \
  --eval-d0 "$POOL" --skeleton "$SKEL" --random-policy --obs dino \
  --limit 3 --max-steps 40 --report-budgets 40 \
  --output "$OUT/seam_smoke_chrome_$SLURM_JOB_ID.json" || rc=1

section "GATE 4a: zoom boundary (what NG does with a rejected zoom)"
uv run --no-sync python native/probe_zoom_boundary.py || true

section "GATE 4b: input parity (select), Chrome -> simulator"
JSONL=$OUT/seam_select_browser_$SLURM_JOB_ID.jsonl
FRAMES=$OUT/seam_select_frames_$SLURM_JOB_ID
uv run --no-sync python native/probe_select_parity.py --mode browser \
  --config configs/native_select.yaml --pool "$POOL" \
  --n-states "${N_STATES:-8}" --out "$JSONL" --frame-dir "$FRAMES" || rc=1
uv run --no-sync python native/probe_select_parity.py --mode native \
  --config configs/native_select.yaml --browser-jsonl "$JSONL" --diagnose \
  --frame-dir "$FRAMES" || rc=1

section "GATE 5a: 2D pixel parity vs ceiling"
uv run --no-sync python native/probe_visual_audit.py --pairs-dir "$OUT/pairs_v1" --limit 12 || rc=1

section "GATE 5b: 3D pane parity (whole / content / mesh IoU / plane IoU)"
uv run --no-sync python native/probe_pane3d_parity.py --pairs-dir "$OUT/pairs_v1" --limit 12 || rc=1

echo; echo "SEAM-GATES rc=$rc"
exit $rc
