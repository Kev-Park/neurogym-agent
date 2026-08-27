#!/bin/bash
#SBATCH --job-name=nativedensity
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
# Full production-node CPU share so the density numbers reflect real
# contention (a node runs 2 runner procs x 16 envs on 48 cpus).
#SBATCH --cpus-per-task=44
#SBATCH --mem=100G
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-7,sarekl16-1,sarekl16-3,sarekl16-5
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# Density sweep: dino (production path) at M=4..32, plus a render-only
# ceiling (pos obs discards the image after render) at M=16.
set -e
WT=/scratch/kp0374/wt/neurogym-agent-native
cd "$WT"
export PYTHONUNBUFFERED=1
uv run --no-sync python native/probe_native_density.py \
  --config configs/native.yaml --threads-list 4,8,16,32 --obs dino --secs 120
uv run --no-sync python native/probe_native_density.py \
  --config configs/native.yaml --threads-list 16 --obs pos --secs 120
echo "DENSITY-SWEEP-COMPLETE"
