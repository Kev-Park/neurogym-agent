#!/bin/bash
#SBATCH --job-name=mgpu-sock
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --time=00:45:00
#SBATCH --output=slurm_outputs/disagg-mgpu-sock-%j.out
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:3
#SBATCH --cpus-per-task=44
#SBATCH --mem=320G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH hetjob
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=200G
#
# Multi-GPU-per-node DISAGG: G independent trainings on the G-GPU node (one per GPU),
# each with SOCKET fetch to a shared fetch_server.py on the sarek-r27 CPU node. Frees
# the GPU node's cores from fetch -> should beat the local-fetch baseline (which is
# fetch-CPU-bound across GPUs). M=2. Aggregate sps = sum of the per-GPU logs.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
FPORT=$(( 40000 + SLURM_JOB_ID % 20000 ))
export PYTHONUNBUFFERED=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_memory_monitor_refresh_ms=0
export NGL_NATIVE_CHUNK_LRU_MB=120 NGL_NATIVE_VAO_LRU_MB=128 NGL_RENDER_BATCH_SIZE=${M:-2}
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
export MGPU_TAG=mgpu-sock

CPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1" | head -1)
GPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
CPU_IP=$(srun --het-group=1 --nodes=1 --ntasks=1 hostname -I | awk '{print $1}')
export NGL_FETCH_SERVER_HOST=$CPU_IP NGL_FETCH_SERVER_PORT=$FPORT
export NGL_NATIVE_FETCH_BACKEND=socket NGL_NATIVE_FETCH_WORKERS=${FW:-1}
export NGL_NATIVE_FETCH_AFFINITY=${AFFINITY:-1}
echo "[mgpu-sock] gpu=$GPU_NODE cpu=$CPU_NODE cpu_ip=$CPU_IP fport=$FPORT"

# Fetch server on the CPU node (forks a handler per connection; G*NRUN*FW total).
srun --het-group=1 --overlap --nodes=1 --ntasks=1 \
  uv run --no-sync python -m ngllib.simulator.fetch_server &
SRV_PID=$!
sleep 12

# Sanity: GPU node reaches the CPU fetch server (the firewall-allowed direction).
srun --het-group=0 --overlap --nodes=1 --ntasks=1 bash scripts/net_probe.sh "$CPU_IP" "$FPORT" || true

# MPS on the GPU node, then launch one training per GPU (socket fetch).
srun --het-group=0 --overlap --nodes=1 --ntasks=1 bash -c \
  "mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY; nvidia-cuda-mps-control -d && echo '[mgpu-sock] mps up'"
srun --het-group=0 --overlap --nodes=1 --ntasks=1 bash scripts/launch_mgpu_trainings.sh
EXIT=$?

kill "$SRV_PID" 2>/dev/null || true
srun --het-group=0 --overlap --nodes=1 --ntasks=1 bash -c "echo quit | nvidia-cuda-mps-control 2>&1 | tail -1" || true
wait 2>/dev/null || true
echo "[mgpu-sock] done exit=$EXIT ; per-GPU steady sps:"
grep -hE '^iter (8|9|1[0-2])' slurm_outputs/mgpu-sock-$J-g*.out 2>/dev/null | grep -oE 'sps=[0-9.]+' || true
