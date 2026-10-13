#!/bin/bash
#SBATCH --job-name=dbl-counterfactual
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:3090:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_outputs/dbl-counterfactual-%j.out



#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5,sarekl16-6
set -eu
cd /scratch/kp0374/wt/neurogym-agent-zmax-left
export PYTHONUNBUFFERED=1
uv run --no-sync python scripts/probe_click_counterfactual.py "$@"
