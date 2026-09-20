#!/bin/bash
#SBATCH --job-name=rb-bench
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/sps-bench-%x-%j.out
#
# throughput-scaling render-batching bench. Right-pane-only (3D pane), in-process
# per-process DINO, under a per-job CUDA MPS daemon (so CUDA switching is present,
# which is what makes the render-context reduction show). Real PPO.
#   CONFIG=configs/native_rb_base.yaml     -> per-env GL contexts (baseline)
#   CONFIG=configs/native_rb_readback.yaml -> batched atlas, one readback/batch
#   CONFIG=configs/native_rb_interop.yaml  -> batched atlas, all-VRAM interop
# Sweep RUNNERS x ENVS (ENVS = envs/runner = the atlas batch size).
#
# Submit, e.g.:
#   sbatch --job-name=rb-base-r16e2 --export=ALL,CONFIG=configs/native_rb_base.yaml,\
#     RUNNERS=16,ENVS=2 scripts/rb_bench.slurm
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
export PYTHONUNBUFFERED=1

CONFIG=${CONFIG:?set CONFIG}
RUNNERS=${RUNNERS:-16}
ENVS=${ENVS:-2}
ITERS=${ITERS:-20}
GPU_PER_RUNNER=${GPU_PER_RUNNER:-0.02}
CPUS_PER_RUNNER=${CPUS_PER_RUNNER:-1.0}
# fragment-125 parity: 125 steps/env/iter -> train_batch = 125 * runners * envs.
BATCH=${BATCH:-$((125 * RUNNERS * ENVS))}
RUN=${RUN:-${SLURM_JOB_NAME}-${SLURM_JOB_ID}}

# Atlas cell count = envs per runner. The render service sizes its atlas from this.
export NGL_RENDER_BATCH_SIZE="$ENVS"

# M: DINO-server instance count (only used by the *_srv_* configs). Empty = use
# the config's value.
DINO_ARGS=""
[ -n "${M:-}" ] && DINO_ARGS="--dino-server-instances $M"

export NGL_NATIVE_FETCH_WORKERS="${NGL_NATIVE_FETCH_WORKERS:-2}"
export NGL_NATIVE_MESH_LRU_MB="${NGL_NATIVE_MESH_LRU_MB:-128}"
export NGL_NATIVE_CHUNK_LRU_MB="${NGL_NATIVE_CHUNK_LRU_MB:-256}"
# Shared per-process mesh pool now (one MeshRenderer/process), so size it for the
# process's whole env fleet, not per-env.
export NGL_NATIVE_VAO_LRU_MB="${NGL_NATIVE_VAO_LRU_MB:-512}"

# --- per-job CUDA MPS daemon (kept on for all arms; exported before ray) ---
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${SLURM_JOB_ID}
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-${SLURM_JOB_ID}
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
echo "[mps] node=$(hostname) starting nvidia-cuda-mps-control -d"
nvidia-cuda-mps-control -d && echo "[mps] daemon start rc=$?" || echo "[mps] daemon FAILED (continuing)"
sleep 3

echo "[rb] config=$CONFIG runners=$RUNNERS envs=$ENVS batch=$BATCH batch_size=$NGL_RENDER_BATCH_SIZE"
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python -m ngllib_agent.train \
  --config "$CONFIG" --run-name "$RUN" \
  --no-spawn-curriculum --learner-gpu \
  --num-env-runners "$RUNNERS" --num-envs-per-env-runner "$ENVS" \
  --num-gpus-per-env-runner "$GPU_PER_RUNNER" --num-cpus-per-env-runner "$CPUS_PER_RUNNER" \
  --vector threads --iters "$ITERS" --train-batch-size "$BATCH" \
  --checkpoint-every 999999 --no-degraded-exit --wandb-mode disabled \
  $DINO_ARGS

echo quit | nvidia-cuda-mps-control 2>&1 | sed 's/^/[mps] /' || true
echo "[rb] done"
