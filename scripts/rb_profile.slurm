#!/bin/bash
#SBATCH --job-name=rb-profile
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/rb-profile-%j.out
#
# GL-render vs DINO-encode GPU-time split (informs disaggregation). Single
# process, no MPS needed. Runs a few batch sizes.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
export PYTHONUNBUFFERED=1
for B in 8 12 16; do
  echo "=== batch $B ==="
  PROF_BATCH=$B PROF_ITERS=200 UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
    uv run --no-sync python scripts/gl_dino_profile.py
done
echo "[profile] done"
