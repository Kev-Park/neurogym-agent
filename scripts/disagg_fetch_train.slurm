#!/bin/bash
#SBATCH --job-name=disagg-fetch
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=22
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/disagg-fetch-%j.out
#
# Disaggregated-fetch SPS test (2-GPU-node proof; the 2nd node's GPU is unused).
# node0 = render + DINO + learner (Ray head, 1 GPU). node1 = fetch actors (Ray
# worker, 0 GPU, advertises the fetch_cpu resource). The env-runners on node0 build
# their fetch pools as Ray _FetchActors pinned to node1 via NGL_NATIVE_FETCH_RESOURCE,
# so CloudVolume read+Draco decode runs on node1's cores and node0's cores are freed
# for render/step -- attacking the chunk-decompress CPU wall (SPS recipe §10).
#
# A/B affinity:  AFFINITY=1 (env->actor sticky, default) vs AFFINITY=0 (round-robin).
#   sbatch scripts/disagg_fetch_train.slurm            # affinity on
#   AFFINITY=0 sbatch scripts/disagg_fetch_train.slurm # affinity off
# Baseline for comparison = single-node process backend at the same 16x8 density.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
AFFINITY=${AFFINITY:-1}
RAY_PORT=$(( 20000 + (SLURM_JOB_ID % 20000) ))   # unique per job: two of my jobs can
RAY_TMP=/tmp/ray_$J                               # co-tenant a node without colliding
                                                  # on 6379 / /tmp/ray (session clash)

# --- env inherited by ALL srun steps (raylets + their actors read these) ---
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0          # shared /scratch; no CWD upload
export RAY_memory_monitor_refresh_ms=0          # disable Ray OOM killer: it misfires
                                                # in the SLURM cgroup ("negative used
                                                # memory for cgroup -1" -> raylet SIGKILL)
# NB: NGL_NATIVE_FETCH_BACKEND=ray is set ONLY in the head ray-start below, so the
# env-runner ACTORS (spawned by that raylet) use the ray backend, while the DRIVER's
# RLlib spec-inference env (this srun's env) keeps the default PROCESS backend -- a
# ray-backend spec env on the driver hung build_algo (0 actors, 0 ray usage, silent).
export NGL_NATIVE_CHUNK_LRU_MB=160 NGL_NATIVE_VAO_LRU_MB=128
export NGL_RENDER_BATCH_SIZE=8
export UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp   # NB: never `srun env VAR=..` -
                                                      # a broken ~/.local/bin/env shim
                                                      # shadows /usr/bin/env (exit 13)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J

NODES=($(scontrol show hostnames "$SLURM_NODELIST"))
HEAD=${NODES[0]}; FETCH=${NODES[1]}
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD" hostname -I | awk '{print $1}')
echo "[disagg] head(gpu)=$HEAD ($HEAD_IP)  fetch(cpu)=$FETCH  affinity=$AFFINITY"

# Ray head on the GPU node: start MPS first so env-runner actors (spawned by this
# raylet) inherit CUDA_MPS_PIPE_DIRECTORY and co-schedule render+DINO+learner.
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" bash -c "
  export NGL_NATIVE_FETCH_BACKEND=ray NGL_NATIVE_FETCH_RESOURCE=fetch_cpu
  export NGL_NATIVE_FETCH_AFFINITY=$AFFINITY NGL_NATIVE_FETCH_WORKERS=1
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d && echo '[disagg] mps up on head'
  uv run --no-sync ray start --head --node-ip-address=$HEAD_IP --port=$RAY_PORT \
      --temp-dir=$RAY_TMP --num-cpus=20 --num-gpus=1 --block" &
HEAD_PID=$!
sleep 20

# Ray worker on the fetch node: no GPU, advertises fetch_cpu for the actors.
srun --overlap --nodes=1 --ntasks=1 -w "$FETCH" \
  uv run --no-sync ray start --address="$HEAD_IP:$RAY_PORT" --temp-dir=$RAY_TMP \
      --num-cpus=20 --num-gpus=0 --resources='{"fetch_cpu": 20}' --block &
FETCH_PID=$!
sleep 20

echo "[disagg] launching training (driver on head)"
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
  uv run --no-sync python -m ngllib_agent.train \
      --config configs/native_rb_interop.yaml --run-name disagg-$J-aff$AFFINITY \
      --no-spawn-curriculum --learner-gpu \
      --num-env-runners 16 --num-envs-per-env-runner 8 \
      --num-gpus-per-env-runner 0.03 --num-cpus-per-env-runner 1.0 \
      --vector threads --iters 15 --train-batch-size 24000 \
      --checkpoint-every 999999 --no-degraded-exit --wandb-mode disabled
EXIT=$?

echo "[disagg] stopping ray"
for n in "$HEAD" "$FETCH"; do
  srun --overlap --nodes=1 --ntasks=1 -w "$n" uv run --no-sync ray stop --force 2>&1 | tail -1 || true
done
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" bash -c "echo quit | nvidia-cuda-mps-control 2>&1 | tail -1" || true
kill "$HEAD_PID" "$FETCH_PID" 2>/dev/null || true
wait 2>/dev/null || true
echo "[disagg] done exit=$EXIT (affinity=$AFFINITY)"
