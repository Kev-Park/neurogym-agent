#!/usr/bin/env bash
#SBATCH --job-name=dino
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
#SBATCH --output=slurm_outputs/dino-%j.out

# DINO throughput: per-env (batch 2) vs batched, and threaded per-env ceiling.
# See scripts/probe_dino.py.
cd /scratch/kp0374/neurogym-agent
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python scripts/probe_dino.py
