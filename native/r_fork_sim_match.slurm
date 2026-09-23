#!/bin/bash
#SBATCH --job-name=forkmatch
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# How close is the simulator to the self-hosted Chrome fork, factually?
# Needs a GPU: Chromium/ANGLE for one arm, the simulator's EGL context for the
# other. NGLLIB prepends a branch checkout of ngllib so the production
# checkouts keep running main.
#   NGLLIB=/scratch/kp0374/wt/neurogym-login/src sbatch native/r_fork_sim_match.slurm
set -e
REPO=${REPO:-/scratch/kp0374/neurogym-agent}
cd "$REPO"
export PYTHONUNBUFFERED=1
[ -n "${NGLLIB:-}" ] && export PYTHONPATH="$NGLLIB:${PYTHONPATH:-}"
export NGL_NATIVE_FETCH_WORKERS=${FETCH_WORKERS:-4}
export NGL_NATIVE_MESH_LRU_MB=512
echo "forkmatch: node=$(hostname) repo=$REPO ngllib=${NGLLIB:-installed} commit=$(git rev-parse --short HEAD)"
uv run --no-sync python ${PROBE:-native/probe_fork_sim_match.py} "$@"
echo "FORKMATCH-DONE"
