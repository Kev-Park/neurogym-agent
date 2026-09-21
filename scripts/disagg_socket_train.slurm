#!/bin/bash
#SBATCH --job-name=disagg-socket
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --time=00:45:00
#SBATCH --output=slurm_outputs/disagg-socket-%j.out
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=44
#SBATCH --mem=140G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH hetjob
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=200G
#
# Disaggregated fetch via SOCKET -- NO cross-node Ray. group0 = GPU node (render +
# DINO + learner, LOCAL ray); group1 = sarek-r27 CPU node running fetch_server.py.
# The GPU env-runners connect OUT to the CPU fetch server (GPU->CPU, the only
# firewall-allowed direction). 16 runners x 6 = 96 fetch worker procs on the CPU node
# (~96 decode cores vs the baseline's 22). A/B affinity: AFFINITY=1|0.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
AFFINITY=${AFFINITY:-1}
FPORT=$(( 40000 + SLURM_JOB_ID % 20000 ))
export PYTHONUNBUFFERED=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_memory_monitor_refresh_ms=0
export UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp
export NGL_NATIVE_CHUNK_LRU_MB=160 NGL_NATIVE_VAO_LRU_MB=128 NGL_RENDER_BATCH_SIZE=8
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J

CPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1" | head -1)
GPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
CPU_IP=$(srun --het-group=1 --nodes=1 --ntasks=1 hostname -I | awk '{print $1}')
echo "[socket] gpu=$GPU_NODE cpu=$CPU_NODE cpu_ip=$CPU_IP fport=$FPORT affinity=$AFFINITY"

# Fetch env inherited by all srun steps: server reads PORT; training reads BACKEND +
# HOST + PORT (env-runner actors are LOCAL ray on the GPU node -> inherit the driver).
export NGL_FETCH_SERVER_HOST=$CPU_IP NGL_FETCH_SERVER_PORT=$FPORT
export NGL_NATIVE_FETCH_BACKEND=socket NGL_NATIVE_FETCH_AFFINITY=$AFFINITY
export NGL_NATIVE_FETCH_WORKERS=${FW:-6}   # NRUN x FW connections <= CPU-node cores (~104)

# Fetch server on the CPU node (background; forks a handler per connection).
srun --het-group=1 --overlap --nodes=1 --ntasks=1 \
  uv run --no-sync python -m ngllib.simulator.fetch_server &
SRV_PID=$!
sleep 12

# Sanity: GPU node can reach the CPU fetch server (the allowed direction).
srun --het-group=0 --overlap --nodes=1 --ntasks=1 bash scripts/net_probe.sh "$CPU_IP" "$FPORT" || true

# MPS daemon on the GPU node (persists for the training's CUDA actors).
srun --het-group=0 --overlap --nodes=1 --ntasks=1 bash -c \
  "mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY; nvidia-cuda-mps-control -d && echo '[socket] mps up'"

echo "[socket] launching training on the GPU node (local ray + socket fetch)"
srun --het-group=0 --overlap --nodes=1 --ntasks=1 \
  uv run --no-sync python -m ngllib_agent.train \
      --config configs/native_rb_interop.yaml --run-name disagg-socket-$J-aff$AFFINITY \
      --no-spawn-curriculum --learner-gpu \
      --num-env-runners ${NRUN:-16} --num-envs-per-env-runner 8 \
      --num-gpus-per-env-runner ${GPUFRAC:-0.03} --num-cpus-per-env-runner 0.5 \
      --vector threads --iters 15 --train-batch-size 24000 \
      --checkpoint-every 999999 --no-degraded-exit --wandb-mode disabled
EXIT=$?

kill "$SRV_PID" 2>/dev/null || true
srun --het-group=0 --overlap --nodes=1 --ntasks=1 bash -c "echo quit | nvidia-cuda-mps-control 2>&1 | tail -1" || true
wait 2>/dev/null || true
echo "[socket] done exit=$EXIT (affinity=$AFFINITY)"
