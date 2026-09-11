#!/bin/bash
#SBATCH --job-name=zoombound
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# Gate 4a on its own: what Neuroglancer does with a rejected (non-positive)
# projectionScale. Chrome only; see native/probe_zoom_boundary.py.
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1
uv run --no-sync python native/probe_zoom_boundary.py
