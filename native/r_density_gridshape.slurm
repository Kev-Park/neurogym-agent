#!/bin/bash
#SBATCH --job-name=nativegrid
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=44
#SBATCH --mem=150G
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-7,sarekl16-1,sarekl16-3,sarekl16-5
#SBATCH --time=01:10:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# Node-allocation shape grid at constant 32 envs (plus one 48-env point):
# K processes x M threaded envs. Completes the K/M grid the first sweeps
# only sampled (K=1 x M=4..32; K=4,8 x M=4). K=32 x M=1 = fully
# process-isolated envs, one DINO + CUDA context each — VRAM column
# decides if that shape is even admissible.
# Node sps per round = sum of its DENSITY lines.
set -e
WT=/scratch/kp0374/wt/neurogym-agent-native
cd "$WT"
export PYTHONUNBUFFERED=1

# rounds: K:M:fetch-workers-per-runner
for round in 4:8:3 16:2:1 32:1:1 12:4:2; do
  K=${round%%:*}; rest=${round#*:}; M=${rest%%:*}; FW=${rest##*:}
  echo "ROUND K=$K M=$M FW=$FW total_envs=$((K * M))"
  export NGL_NATIVE_FETCH_WORKERS=$FW
  pids=()
  for i in $(seq 1 "$K"); do
    uv run --no-sync python native/probe_native_density.py \
      --config configs/native.yaml --threads-list "$M" --obs dino --secs 120 \
      2>&1 | sed "s/^/[p$i] /" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "ROUND-DONE K=$K M=$M"
done
echo "GRIDSHAPE-COMPLETE"
