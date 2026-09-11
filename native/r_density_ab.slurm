#!/bin/bash
#SBATCH --job-name=densab
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# Throughput A/B for the parallel fetch split.
#
# Splitting the bundle into two parallel jobs halved the 2D pane's wait and
# took the move response from "7 of 8 states never respond in 41 steps" to
# "8 of 8, median 32.5". But it also DOUBLES pool dispatches per fetch, and
# the workers are shared across every env on the runner -- so the dynamics win
# could be paid for in aggregate steps/sec, which is the number this project
# actually optimises. Same node, same density, both arms.
set -e
WT=/scratch/kp0374/wt/neurogym-agent-native
cd "$WT"
export PYTHONUNBUFFERED=1
M=${THREADS:-16}
SECS=${SECS:-90}

echo "=== ARM: parallel parts (shipping) ==="
NGL_NATIVE_PARALLEL_PARTS=1 uv run --no-sync python native/probe_native_density.py \
  --config configs/native.yaml --threads-list "$M" --obs dino --secs "$SECS"

echo "=== ARM: single bundled job (previous behaviour) ==="
NGL_NATIVE_PARALLEL_PARTS=0 uv run --no-sync python native/probe_native_density.py \
  --config configs/native.yaml --threads-list "$M" --obs dino --secs "$SECS"

echo "DENSITY-AB-COMPLETE"
