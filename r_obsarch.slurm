#!/usr/bin/env bash
#SBATCH --job-name=obsarch
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=code@thiskevin.com
#SBATCH --output=slurm_outputs/obsarch-%j.out

# Phase-1: per-env-CPU-decode+DINO vs batched-GPU-decode+DINO. See probe_obs_arch.py.
cd /scratch/kp0374/neurogym-agent
for M in 16 32; do
  UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python scripts/probe_obs_arch.py "$M" 8
done
