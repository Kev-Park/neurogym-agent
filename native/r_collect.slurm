#!/bin/bash
#SBATCH --job-name=nativecollect
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
# Good vulkan nodes MINUS coord-v9's allocation (sarekl16-[6-8]).
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-7,sarekl16-1,sarekl16-3,sarekl16-5
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# Browser-side calibration pairs: (state, raw frame a/b) x N via the MAIN
# checkout's venv+configs; the script lives on the native-renderer worktree.
set -e
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python \
    /scratch/kp0374/wt/neurogym-agent-native/native/collect_pairs.py \
    --n-states 300 --out-dir /scratch/kp0374/native_spike/pairs_v1
