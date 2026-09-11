#!/bin/bash
#SBATCH --job-name=p3dpar
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

# 3D pane parity against Chrome, through the production render path.
set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1
uv run --no-sync python native/probe_pane3d_parity.py   --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 12
