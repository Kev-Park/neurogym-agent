#!/bin/bash
#SBATCH --job-name=levers
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-5,sarekl15-7,sarekl16-1,sarekl16-3,sarekl16-5,sarekl16-6,sarekl16-7,sarekl16-8
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=code@thiskevin.com
#SBATCH --output=slurm_outputs/levers-%j.out

# Cycle-time lever probe: capture_scale / cache-clear / footprint flags.
# Sized small (12cpu/24G) to backfill into co-tenanted nodes. Sample frames are
# copied to slurm_outputs/levers-frames-$SLURM_JOB_ID for visual QA.
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python scripts/probe_cycle_levers.py
RC=$?
mkdir -p "slurm_outputs/levers-frames-$SLURM_JOB_ID"
cp /tmp/cycle_levers/*.png "slurm_outputs/levers-frames-$SLURM_JOB_ID/" 2>/dev/null
exit $RC
