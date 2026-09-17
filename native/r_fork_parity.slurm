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
set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1
export NGL_DIST=${NGL_DIST:-/scratch/kp0374/ngl_fork_dist}
echo "fork parity: node=$(hostname) commit=$(git rev-parse --short HEAD) dist=$NGL_DIST"
uv run --no-sync python scripts/probe_fork_parity.py
echo "FORKPARITY-DONE"
