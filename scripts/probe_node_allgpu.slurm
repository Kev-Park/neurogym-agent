#!/bin/bash
#SBATCH --job-name=node-allgpu
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --gres=gpu:3090:8
#SBATCH --time=00:45:00
#SBATCH --output=slurm_outputs/node-allgpu-%j.out


set -eu
cd /scratch/kp0374/wt/neurogym-agent-zmax-left
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version --format=csv
pids=()
for slot in $(seq 0 7); do
  # Step-level GPU cgroups also constrain Chromium, which ignores CVD.
  srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:3090:1 \
    uv run --no-sync python scripts/probe_node_compat.py --seconds 1800 --workers 1 \
    > slurm_outputs/node-allgpu-${SLURM_JOB_ID}-slot${slot}.out 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
nvidia-smi --query-gpu=index,uuid,temperature.gpu,memory.used --format=csv
echo "ALL_GPU_EXIT=$rc"
exit "$rc"
