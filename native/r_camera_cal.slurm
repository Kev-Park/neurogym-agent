#!/bin/bash
#SBATCH --job-name=camcal
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # the checkout this was submitted from
export PYTHONUNBUFFERED=1
uv run --no-sync python native/probe_camera_cal.py   --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 8
