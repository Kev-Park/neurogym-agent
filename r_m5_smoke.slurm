#!/bin/bash
#SBATCH --job-name=m5smoke
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=12
# Sized to backfill into co-tenanted nodes (holder leaves ~24cpu/47G free):
# one Chrome + the probe needs far less than the trainer's 48G.
#SBATCH --mem=24G
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-5,sarekl15-7,sarekl16-1,sarekl16-3,sarekl16-5,sarekl16-6,sarekl16-7,sarekl16-8
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8
#SBATCH --time=00:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=code@thiskevin.com
#SBATCH --output=slurm_outputs/m5smoke-%j.out

# M5 reset-ahead functional smoke: one browser, 4 episode boundaries, verify
# warm contexts are prepped in the background and adopted at reset (warm=true,
# navigate_ms~0). Small + short so it backfills ahead of the 3-node sweep jobs.
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python scripts/probe_reset_ahead.py
