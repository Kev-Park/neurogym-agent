#!/bin/bash
#SBATCH --job-name=forkparity
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# Is the self-hosted lab-fork build calibration-compatible with the appspot
# build ngllib.state was fitted against?  Runs the same states through both on
# a vulkan-good node (production rendering path) and diffs state readback +
# right-pane pixels.  sbatch native/r_fork_parity.slurm
# Submittable from a scratch dir so the production checkout keeps its branch:
# REPO holds the venv, PROBE the script (both must be on a shared filesystem,
# not the login node's /tmp).
set -e
REPO=${REPO:-/scratch/kp0374/neurogym-agent}
PROBE=${PROBE:-scripts/probe_fork_parity.py}
cd "$REPO"
export PYTHONUNBUFFERED=1
export NGL_DIST=${NGL_DIST:-/scratch/kp0374/ngl_fork_dist}
echo "fork parity: node=$(hostname) repo=$REPO probe=$PROBE dist=$NGL_DIST"
uv run --no-sync python "$PROBE"
echo "FORKPARITY-DONE"
