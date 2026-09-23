#!/bin/bash
#SBATCH --job-name=logingates
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

# Gates for the refactor-login branch, run against a BRANCH checkout while the
# production checkouts stay on main:
#   REPO   holds the venv (main checkout)
#   WT     the agent worktree whose probes/configs are under test
#   NGLLIB the ngllib src to prepend (branch ngllib)
#
# 1. select parity (3D pick): the offset fix moved every 3D click 3 capture px,
#    so Chrome's answers are re-collected and replayed rather than reused.
# 2. zoom boundary: the shared rule both backends implement.
#
#   REPO=/scratch/kp0374/neurogym-agent \
#   WT=/scratch/kp0374/wt/neurogym-agent-login \
#   NGLLIB=/scratch/kp0374/wt/neurogym-login/src \
#   sbatch native/r_login_gates.slurm
set -e
REPO=${REPO:-/scratch/kp0374/neurogym-agent}
WT=${WT:-/scratch/kp0374/wt/neurogym-agent-login}
cd "$REPO"
export PYTHONUNBUFFERED=1
[ -n "${NGLLIB:-}" ] && export PYTHONPATH="$NGLLIB:$WT/src:${PYTHONPATH:-}"
OUT=/scratch/kp0374/native_spike
POOL=${POOL:-/scratch/kp0374/neurogym-agent/eval_d0_v1.parquet}
N=${N_STATES:-8}
JSONL=$OUT/login_select_browser_$SLURM_JOB_ID.jsonl
FRAMES=$OUT/login_select_frames_$SLURM_JOB_ID
echo "login gates: node=$(hostname) repo=$REPO wt=$WT ngllib=${NGLLIB:-installed}"

echo; echo "=== GATE: select parity, Chrome ground truth ==="
uv run --no-sync python "$WT/native/probe_select_parity.py" --mode browser \
  --config "$WT/configs/native_select.yaml" --pool "$POOL" \
  --n-states "$N" --out "$JSONL" --frame-dir "$FRAMES"

echo; echo "=== GATE: select parity, simulator replay (3D pick after TOOLBAR_3D) ==="
uv run --no-sync python "$WT/native/probe_select_parity.py" --mode native \
  --config "$WT/configs/native_select.yaml" --browser-jsonl "$JSONL" --diagnose \
  --frame-dir "$FRAMES"

echo; echo "=== GATE: zoom boundary ==="
uv run --no-sync python "$WT/native/probe_zoom_boundary.py" || true

echo "LOGINGATES-DONE"
