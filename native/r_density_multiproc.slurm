#!/bin/bash
#SBATCH --job-name=nativemulti
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=44
#SBATCH --mem=150G
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-7,sarekl16-1,sarekl16-3,sarekl16-5
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# Multi-PROCESS density: K probe processes x M threaded envs each on one
# GPU. The single-process sweep plateaus ~10 sps regardless of M (per-step
# Python is GIL-serialized); node throughput should scale with K instead.
# Node sps = sum of the K DENSITY lines per round.
set -e
WT=/scratch/kp0374/wt/neurogym-agent-native
cd "$WT"
export PYTHONUNBUFFERED=1
export NGL_NATIVE_FETCH_WORKERS=3

for K in 4 8; do
  echo "ROUND K=$K M=4"
  pids=()
  for i in $(seq 1 $K); do
    uv run --no-sync python native/probe_native_density.py \
      --config configs/native.yaml --threads-list 4 --obs dino --secs 120 \
      2>&1 | sed "s/^/[p$i] /" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "ROUND-DONE K=$K"
done
echo "MULTIPROC-COMPLETE"
