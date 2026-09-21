#!/bin/bash
#SBATCH --job-name=disagg-netcheck
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --time=00:06:00
#SBATCH --output=slurm_outputs/disagg-netcheck-%j.out
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH hetjob
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#
# Firewall DIRECTIONALITY probe (GPU node <-> sarek-r27 CPU node). Runs a real TCP
# listener on each node, then connects from the OTHER node with a bounded /dev/tcp.
# CPU->GPU is known BLOCKED; the question is GPU->CPU. If GPU->CPU is OPEN, a
# GPU-INITIATED socket fetch (env-runners connect out to a fetch server on the CPU
# node) bypasses the firewall with NO login proxy and full node bandwidth.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
GPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
CPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1" | head -1)
GPU_IP=$(srun --het-group=0 --nodes=1 --ntasks=1 hostname -I | awk '{print $1}')
CPU_IP=$(srun --het-group=1 --nodes=1 --ntasks=1 hostname -I | awk '{print $1}')
PORT=$(( 30000 + SLURM_JOB_ID % 10000 ))
echo "[netcheck] gpu=$GPU_NODE($GPU_IP) cpu=$CPU_NODE($CPU_IP) port=$PORT"

# Listeners on both nodes (background, ~70s).
srun --het-group=0 --overlap --nodes=1 --ntasks=1 \
  uv run --no-sync python scripts/listen.py $PORT 70 &
srun --het-group=1 --overlap --nodes=1 --ntasks=1 \
  uv run --no-sync python scripts/listen.py $PORT 70 &
sleep 12

echo "=== CPU -> GPU (known blocked) ==="
srun --het-group=1 --overlap --nodes=1 --ntasks=1 bash scripts/net_probe.sh "$GPU_IP" "$PORT" || true
echo "=== GPU -> CPU (the one that matters) ==="
srun --het-group=0 --overlap --nodes=1 --ntasks=1 bash scripts/net_probe.sh "$CPU_IP" "$PORT" || true

wait 2>/dev/null || true
echo "[netcheck] done"
