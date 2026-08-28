#!/bin/bash
#SBATCH --job-name=encadd
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=44
#SBATCH --mem=150G
#SBATCH --exclude=sarekl15-2,sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# Coarse per-runner-batched shapes (4x8, 2x16) to complete the shape curve
# (8x4=211 > 16x2=130 sps; batch-size curve says coarser may win more).
set -e
cd /scratch/kp0374/wt/neurogym-agent-native
export PYTHONUNBUFFERED=1
export NGL_NATIVE_FETCH_WORKERS=1
for round in 4:8 2:16; do
  K=${round%%:*}; M=${round##*:}
  echo "ROUND per-runner K=$K M=$M"
  pids=()
  for i in $(seq 1 "$K"); do
    uv run --no-sync python native/probe_encode_strategies.py \
      --mode per-runner --m "$M" --secs 120 --seed "$i" 2>&1 | sed "s/^/[p$i] /" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "ROUND-DONE K=$K M=$M"
done
echo "ADDENDUM-COMPLETE"
