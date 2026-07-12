#!/usr/bin/env bash
#SBATCH --job-name=decode
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
#SBATCH --output=slurm_outputs/decode-%j.out

# CPU-decode ceiling (memory-bandwidth confirmation) vs GPU nvJPEG decode.
# See scripts/probe_decode.py.
cd /scratch/kp0374/neurogym-agent
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python scripts/probe_decode.py
