#!/usr/bin/env bash
#SBATCH --job-name=flood
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:25:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=code@thiskevin.com
#SBATCH --output=slurm_outputs/flood-%j.out

# Pure screenshot-readback ceiling on ONE GPU: sweep browser count and flood
# captureScreenshot. See scripts/probe_flood.py.
cd /scratch/kp0374/neurogym-agent
for M in 1 4 8 16 32; do
  UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python scripts/probe_flood.py "$M" 15
done
