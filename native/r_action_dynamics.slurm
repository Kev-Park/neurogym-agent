#!/bin/bash
#SBATCH --job-name=actdyn
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# Per-verb step-by-step dynamics: every action type, both backends.
# select was measured alone and responds fetch-free; the verbs that MOVE the
# viewer still refetch tiles and are the ones that can diverge.
set -e
WT=/scratch/kp0374/wt/neurogym-agent-native
cd "$WT"
export PYTHONUNBUFFERED=1
OUT=/scratch/kp0374/native_spike/act_browser_$SLURM_JOB_ID.jsonl
N=${N_STATES:-3}
STEPS=${STEPS:-60}
OBS=${OBS:-dino}
SEQ=${SEQ:-0}

uv run --no-sync python native/probe_action_dynamics.py --mode browser \
  --config configs/native_select.yaml \
  --pool /scratch/kp0374/neurogym-agent/eval_d0_v1.parquet \
  --n-states "$N" --steps "$STEPS" --obs "$OBS" --sequence "$SEQ" --out "$OUT"

uv run --no-sync python native/probe_action_dynamics.py --mode native \
  --config configs/native_select.yaml --browser-jsonl "$OUT" \
  --steps "$STEPS" --obs "$OBS"
