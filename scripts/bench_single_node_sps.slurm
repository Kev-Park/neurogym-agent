#!/bin/bash
#SBATCH --job-name=sps-bench
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
# Single-node TRAINING throughput measurement (not the coordinator; a plain
# 1-node job so the sps number is unambiguous). Runs real RLlib PPO on the
# chosen backend's production config for a brief window and prints per-iter
# sps. --no-spawn-curriculum makes every reset spawn at full distance from
# step 0, so the measured sps is the SUSTAINED (post-anneal) training rate --
# the curriculum's early near-target episodes reset ~2x more often and
# understate it. --obs is left to default (dino) to include the real
# per-step DINO encode cost, exactly as production runs do.
#
# Submit (one backend at a time):
#   Simulator: sbatch --job-name=sps-sim --export=ALL,\
#     CONFIG=configs/native.yaml,RUNNERS=32,ENVS=3,BATCH=12000,\
#     GPU_PER_RUNNER=0.02,CPUS_PER_RUNNER=1.2,LEARNER_GPU=--learner-gpu,ITERS=25 \
#     scripts/bench_single_node_sps.slurm
#   Chrome:    sbatch --job-name=sps-chrome --export=ALL,\
#     CONFIG=configs/ppo_zmax_navigate.yaml,RUNNERS=2,ENVS=16,BATCH=4000,\
#     GPU_PER_RUNNER=0.5,CPUS_PER_RUNNER=1.0,LEARNER_GPU=,ITERS=20 \
#     scripts/bench_single_node_sps.slurm
set -u
# dino-server worktree: run the worktree's code + .venv (NOT main).
cd /scratch/kp0374/wt/neurogym-agent-dino-server
export PYTHONUNBUFFERED=1

CONFIG=${CONFIG:?set CONFIG}
RUNNERS=${RUNNERS:-32}
ENVS=${ENVS:-3}
BATCH=${BATCH:-12000}
GPU_PER_RUNNER=${GPU_PER_RUNNER:-0.02}
CPUS_PER_RUNNER=${CPUS_PER_RUNNER:-1.2}
LEARNER_GPU=${LEARNER_GPU:-}
ITERS=${ITERS:-25}
RUN=${RUN:-${SLURM_JOB_NAME}-${SLURM_JOB_ID}}

# DINO-server sweep overrides (empty = use config). M sweeps instances;
# SRV_MAX_BATCH=2 SRV_MAX_DELAY=0 gives no-batch (dispatch as arrivals come).
DINO_ARGS=""
[ -n "${M:-}" ] && DINO_ARGS="$DINO_ARGS --dino-server-instances $M"
[ -n "${SRV_MAX_BATCH:-}" ] && DINO_ARGS="$DINO_ARGS --dino-server-max-batch $SRV_MAX_BATCH"
[ -n "${SRV_MAX_DELAY:-}" ] && DINO_ARGS="$DINO_ARGS --dino-server-max-delay-ms $SRV_MAX_DELAY"

# Simulator per-process cache env (verbatim from launch_coord_local.sh; the
# Chrome backend ignores all of them). Sized for the 32x3 single-node default.
export NGL_NATIVE_FETCH_WORKERS="${NGL_NATIVE_FETCH_WORKERS:-2}"
export NGL_NATIVE_MESH_LRU_MB="${NGL_NATIVE_MESH_LRU_MB:-128}"
export NGL_NATIVE_CHUNK_LRU_MB="${NGL_NATIVE_CHUNK_LRU_MB:-256}"
export NGL_NATIVE_VAO_LRU_MB="${NGL_NATIVE_VAO_LRU_MB:-256}"

echo "[bench] node=$(hostname) config=$CONFIG runners=$RUNNERS envs=$ENVS "\
"batch=$BATCH gpu/runner=$GPU_PER_RUNNER learner_gpu='$LEARNER_GPU' iters=$ITERS"

UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python -m ngllib_agent.train \
  --config "$CONFIG" \
  --run-name "$RUN" \
  --no-spawn-curriculum \
  $LEARNER_GPU \
  --num-env-runners "$RUNNERS" --num-envs-per-env-runner "$ENVS" \
  --num-gpus-per-env-runner "$GPU_PER_RUNNER" --num-cpus-per-env-runner "$CPUS_PER_RUNNER" \
  --vector threads \
  --iters "$ITERS" --train-batch-size "$BATCH" \
  --checkpoint-every 999999 --no-degraded-exit \
  --wandb-mode disabled \
  $DINO_ARGS
