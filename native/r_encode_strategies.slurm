#!/bin/bash
#SBATCH --job-name=encstrat
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=44
#SBATCH --mem=150G
#SBATCH --nodelist=sarekl15-1,sarekl15-4,sarekl15-7,sarekl16-1,sarekl16-3,sarekl16-5
#SBATCH --time=01:15:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# Encoding-strategy shootout (experiment 1): raw encoder curve, then
# per-runner batched shapes (K procs x M envs, node totals = sum of
# PERRUNNER lines per round), then the per-node service.
set -e
WT=/scratch/kp0374/wt/neurogym-agent-native
cd "$WT"
export PYTHONUNBUFFERED=1

echo "ROUND bench"
uv run --no-sync python native/probe_encode_strategies.py --mode bench

for round in 8:4 16:2; do
  K=${round%%:*}; M=${round##*:}
  echo "ROUND per-runner K=$K M=$M"
  export NGL_NATIVE_FETCH_WORKERS=1
  pids=()
  for i in $(seq 1 "$K"); do
    uv run --no-sync python native/probe_encode_strategies.py \
      --mode per-runner --m "$M" --secs 120 --seed "$i" 2>&1 | sed "s/^/[p$i] /" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "ROUND-DONE per-runner K=$K M=$M"
done

echo "ROUND service K=32"
uv run --no-sync python native/probe_encode_strategies.py \
  --mode service --k 32 --secs 120
echo "ENCSTRAT-COMPLETE"
