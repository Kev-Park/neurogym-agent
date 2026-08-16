#!/bin/bash
#SBATCH --job-name=levers2
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-5,sarekl15-7,sarekl16-1,sarekl16-3,sarekl16-5,sarekl16-6,sarekl16-7,sarekl16-8
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=code@thiskevin.com
#SBATCH --output=slurm_outputs/levers2-%j.out

# Aggregate confirm of capture_scale=0.5: M=16 threaded vector, scale 1.0 vs 0.5
# (same node/co-tenancy for a clean relative signal).
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
for S in 1.0 0.5; do
  UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python scripts/probe_throughput.py 16 120 "$S" 2>&1 | grep -E "^\[thru\] RESULT|warmup"
done
