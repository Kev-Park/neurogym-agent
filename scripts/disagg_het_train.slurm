#!/bin/bash
#SBATCH --job-name=disagg-het
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --time=00:45:00
#SBATCH --output=slurm_outputs/disagg-het-%j.out
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=22
#SBATCH --mem=120G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH hetjob
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=200G
#
# Disaggregated fetch, HET-JOB (the win-capable topology the fleet actually favors):
# group0 = a GPU node (render+DINO+learner, 22 cpu), group1 = a GPU-LESS CPU node
# (idle sarek-r27, 104 cpu) advertising fetch_cpu=100. 16 runners x 6 fetch workers =
# 96 _FetchActors pinned to the CPU node -> ~96 real decode cores vs the baseline's
# 22 -> should clearly beat the fetch-contended single node IF fetch-CPU-bound.
# A/B affinity: AFFINITY=1 (default) | 0.  Submit: sbatch --export=ALL,AFFINITY=1 ...
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
AFFINITY=${AFFINITY:-1}
RAY_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))   # unique per job (avoid 6379 collision
RAY_TMP=/tmp/ray_$J                               # when co-tenanting a node)
# Fixed inter-node ray ports (job-id-derived) so CROSS-PARTITION traffic uses KNOWN
# ports -- ray's default random node-manager/object-manager/worker ports are the
# likely thing firewalled between highpri<->preempt (the join on the GCS port worked
# but sustained GCS/object traffic timed out). Stable across a SLURM requeue (same
# job id -> same ports, free on the fresh node).
RAY_PORTS="--node-manager-port=$((RAY_PORT+1)) --object-manager-port=$((RAY_PORT+2)) --min-worker-port=$((RAY_PORT+100)) --max-worker-port=$((RAY_PORT+199)) --include-dashboard=false"
export PYTHONUNBUFFERED=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_memory_monitor_refresh_ms=0          # disable Ray OOM killer (misfires in
                                                # the SLURM cgroup -> raylet SIGKILL)
# FETCH_BACKEND=ray set ONLY in the head ray-start (below) so env-runner ACTORS use
# ray while the DRIVER's RLlib spec env keeps the process backend (a ray spec env on
# the driver hangs build_algo).
export NGL_NATIVE_CHUNK_LRU_MB=160 NGL_NATIVE_VAO_LRU_MB=128 NGL_RENDER_BATCH_SIZE=8
export UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp   # NOT `srun env VAR=..`: broken
                                                      # ~/.local/bin/env shim (exit 13)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J

GPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -1)
CPU_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_1" | head -1)
HEAD_IP=$(srun --het-group=0 --nodes=1 --ntasks=1 hostname -I | awk '{print $1}')
echo "[het] gpu=$GPU_NODE cpu=$CPU_NODE head_ip=$HEAD_IP affinity=$AFFINITY"

# Ray head on the GPU node (MPS for render+DINO+learner contention).
srun --het-group=0 --overlap --nodes=1 --ntasks=1 bash -c "
  export NGL_NATIVE_FETCH_BACKEND=ray NGL_NATIVE_FETCH_RESOURCE=fetch_cpu
  export NGL_NATIVE_FETCH_AFFINITY=$AFFINITY NGL_NATIVE_FETCH_WORKERS=6
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d && echo '[het] mps up on gpu node'
  uv run --no-sync ray start --head --node-ip-address=$HEAD_IP --port=$RAY_PORT \
      $RAY_PORTS --temp-dir=$RAY_TMP --num-cpus=20 --num-gpus=1 --block" &
sleep 20
# Ray worker on the CPU node: no GPU, big fetch_cpu pool.
srun --het-group=1 --overlap --nodes=1 --ntasks=1 \
  uv run --no-sync ray start --address="$HEAD_IP:$RAY_PORT" $RAY_PORTS --temp-dir=$RAY_TMP \
      --num-cpus=104 --num-gpus=0 --resources='{"fetch_cpu": 100}' --block &
sleep 20

# Reachability from the CPU node to the head: FAIL ping => routing; ping OK + tcp
# BLOCKED => host firewall on the port; both OK but ray fails => deeper.
srun --het-group=1 --overlap --nodes=1 --ntasks=1 bash scripts/net_probe.sh "$HEAD_IP" "$RAY_PORT" || true

echo "[het] launching training (driver on gpu node)"
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
srun --het-group=0 --overlap --nodes=1 --ntasks=1 \
  uv run --no-sync python -m ngllib_agent.train \
      --config configs/native_rb_interop.yaml --run-name disagg-het-$J-aff$AFFINITY \
      --no-spawn-curriculum --learner-gpu \
      --num-env-runners 16 --num-envs-per-env-runner 8 \
      --num-gpus-per-env-runner 0.03 --num-cpus-per-env-runner 1.0 \
      --vector threads --iters 15 --train-batch-size 24000 \
      --checkpoint-every 999999 --no-degraded-exit --wandb-mode disabled
EXIT=$?

echo "[het] stopping ray"
srun --het-group=0 --overlap --nodes=1 --ntasks=1 uv run --no-sync ray stop --force 2>&1 | tail -1 || true
srun --het-group=1 --overlap --nodes=1 --ntasks=1 uv run --no-sync ray stop --force 2>&1 | tail -1 || true
srun --het-group=0 --overlap --nodes=1 --ntasks=1 bash -c "echo quit | nvidia-cuda-mps-control 2>&1 | tail -1" || true
echo "[het] done exit=$EXIT (affinity=$AFFINITY)"
