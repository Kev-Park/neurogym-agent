#!/bin/bash
#SBATCH --job-name=node-compat
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:3090:1
#SBATCH --time=00:12:00
#SBATCH --output=slurm_outputs/node-compat-%j.out


set -eu
cd /scratch/kp0374/wt/neurogym-agent-zmax-left
export PYTHONUNBUFFERED=1
uv run --no-sync python scripts/probe_node_compat.py
