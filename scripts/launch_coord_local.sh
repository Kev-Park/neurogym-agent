#!/usr/bin/env bash
# Multi-node SIMULATOR launcher: N renderer nodes (every runner carries its own
# GL context + DINO) + learner on the head GPU, via the production-validated
# coordinator (re-salloc/respawn). Runs from whichever checkout it lives in.
#
#   bash scripts/launch_coord_local.sh [run-name] [renderer-nodes] [target-iters]
# Env: SIM_CONFIG (default configs/native.yaml), TRAIN_BATCH, LR,
# RUNNERS, ENVS_PER_RUNNER, SALLOC_CPUS, SALLOC_MEM, CHECKPOINT_EVERY,
# GPU_PER_RUNNER (default 0.02), MESH_LRU_MB_LOCAL (default 128).
# Drop CHECKPOINT_EVERY to 2 on a heavily-preempted fleet: at the default
# 10, any allocation window shorter than 10 iterations banks NOTHING and
# the run livelocks, replaying the same iterations forever.
# Shrink SALLOC_CPUS/SALLOC_MEM to fit a partially-allocated fleet:
# to fit alongside other users: a whole-node 48cpu/200G request cannot
# schedule when the fleet is partially allocated (2026-09-02: every 3090
# node had only 24 cpus / 48G free, giving a 6-day StartTime estimate).
#
# Learning characteristics are v9-identical (configs/native.yaml); only the
# plumbing differs. The former per-node render+encode SERVICE was deleted in
# the renderer-seam refactor: local mode measured ~2x its throughput at matched
# topology (renderer_seam_plan.md 7.0).
set -u
WORKDIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$WORKDIR"

RUN=${1:-sim-mn}
RENDERERS=${2:-2}
TARGET_ITERS=${3:-250}   # 250 x 12000 = 3.0M steps (v9-equivalent)
NODES=$((RENDERERS + 1))
# ROLLOUT FRAGMENT (corrected 2026-09-02). rollout_fragment_length=auto is
# train_batch / TOTAL envs, and total envs = RUNNERS x ENVS_PER_RUNNER --
# NOT per node. The old default (96 runners x 3 = 288 envs @ batch 12000)
# gave 42-step fragments, not the 125 this comment used to claim: it
# assumed 96 envs. 42 truncates GAE far more often than the ~17-step
# effective horizon (gamma .99, lambda .95) and leaves most fragments with
# no episode end; the two fragment-length experiments that ran short
# scored 75.5% and 60.3% on Chrome.
#
# The constraint is fixed by arithmetic: fragment = batch / envs and
# total_steps = batch x iters, so at fragment 125 and ~3.0M steps (v9-test
# parity) envs x iters = 24,000. More parallelism therefore BUYS FEWER PPO
# iterations at constant experience, and PPO's clipped objective bounds
# policy movement PER ITERATION -- 288 envs would allow only 83.
#   288 envs -> 83 iters | 192 -> 125 | 96 -> 250 | 32 (v9-test) -> 740
# Default picks 96 envs (32 runners x 3 envs) @ batch 12000 -> 125-step
# fragments, 250 iterations, 3.0M steps. Override with RUNNERS /
# ENVS_PER_RUNNER / TRAIN_BATCH, keeping batch = 125 x RUNNERS x ENVS.
NUM_ENV_RUNNERS=${RUNNERS:-32}
ENVS_PER_RUNNER=${ENVS_PER_RUNNER:-3}
CKPT=/scratch/kp0374/checkpoints/${RUN}
STATE_DIR=/scratch/kp0374/coord-state
mkdir -p "$STATE_DIR" "$CKPT"

export RAY_NUM_CPUS=44
# Fetch workers are PER RUNNER: at 32 runners/node the ngllib default (6)
# spawns 192 fetch procs on 44 cores and throttled a plane-on run to 72 sps.
# 2 lets a runner's canvas+plane job overlap the next without thrashing.
export NGL_NATIVE_FETCH_WORKERS=2
# MeshStore is per ENV in local mode, so the node's mesh cache is
# envs_per_node x this. 96 envs x 128 MB ~ 12 GB/node; an unbounded version
# once reached 206 GiB and was cgroup-killed (r_train_native.slurm).
export NGL_NATIVE_MESH_LRU_MB="${MESH_LRU_MB_LOCAL:-128}"
export CURRICULUM_PROGRESS_FILE="${CKPT}/meta.json"
export COORD_WORKDIR="$WORKDIR"
export RAY_HEAD_ENDPOINT_FILE="${STATE_DIR}/ray_head_endpoint-${RUN}.txt"
export NUM_RENDERERS="$RENDERERS"
export SAMPLE_TIMEOUT_S="${SAMPLE_TIMEOUT_S:-600}"
# Every runner carries its own GL context and DINO, so each needs a GPU slice;
# VRAM caps density near 32-40 runners/node (15.3G at 32x1 on a 24G 3090).
GPU_PER_RUNNER="${GPU_PER_RUNNER:-0.02}"
export WORKLOAD_CMD="uv run --no-sync python -m ngllib_agent.train \
  --config ${SIM_CONFIG:-configs/native.yaml} \
  --run-name ${RUN} --learner-gpu \
  --num-env-runners ${NUM_ENV_RUNNERS} --num-envs-per-env-runner ${ENVS_PER_RUNNER} \
  --num-gpus-per-env-runner ${GPU_PER_RUNNER} --num-cpus-per-env-runner 1.2 \
  --vector threads \
  --iters ${TARGET_ITERS} --train-batch-size ${TRAIN_BATCH:-12000} \
  --lr ${LR:-5.0e-4} \
  --sample-timeout-s ${SAMPLE_TIMEOUT_S} \
  --checkpoint-dir ${CKPT} --checkpoint-every ${CHECKPOINT_EVERY:-10} \
  --wandb-project neurogym-agent --resume"

LOG="${STATE_DIR}/coord-${RUN}-$(date +%Y%m%d-%H%M%S).log"
SUP="${STATE_DIR}/supervisor-${RUN}.sh"
cat > "$SUP" <<SUPEOF
#!/bin/bash
cd "${WORKDIR}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS}"
export CURRICULUM_PROGRESS_FILE="${CURRICULUM_PROGRESS_FILE}"
export NGL_NATIVE_FETCH_WORKERS="${NGL_NATIVE_FETCH_WORKERS}"
export NGL_NATIVE_MESH_LRU_MB="${NGL_NATIVE_MESH_LRU_MB}"
export COORD_WORKDIR="${COORD_WORKDIR}"
export RAY_HEAD_ENDPOINT_FILE="${RAY_HEAD_ENDPOINT_FILE}"
export NUM_RENDERERS="${NUM_RENDERERS}"
export SAMPLE_TIMEOUT_S="${SAMPLE_TIMEOUT_S}"
export WORKLOAD_CMD="${WORKLOAD_CMD}"
for attempt in 1 2 3 4 5; do
  echo "[supervisor] attempt \${attempt} \$(date -Is)"
  uv run --no-sync python -m ngllib_agent.distributed.coordinator \\
    --run-id "${RUN}" \\
    --state-file "${STATE_DIR}/state-${RUN}.json" \\
    --renderers "${RENDERERS}" \\
    --learner-cmd "bash scripts/coord_learner.sh" \\
    --renderer-cmd "bash scripts/coord_renderer.sh" \\
    --partition "${PARTITION:-preempt}" \\
    --salloc-time "${SALLOC_TIME:-12:00:00}" \\
    --salloc-gres gpu:3090:1 \\
    --salloc-cpus-per-node "${SALLOC_CPUS:-48}" \\
    --salloc-mem "${SALLOC_MEM:-200G}" \\
    --exclude ${EXCLUDE_NODES:-sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5} \\
    --target-iterations "${TARGET_ITERS}" \\
    --progress-file "${CKPT}/meta.json" \\
    --progress-stall-timeout-s 1800 \\
    --worker-log-dir "${STATE_DIR}/logs-${RUN}"
  rc=\$?
  echo "[supervisor] coordinator exited rc=\${rc} (attempt \${attempt}) \$(date -Is)"
  [ "\${rc}" -eq 0 ] && break
  sleep 30
done
echo "[supervisor] done \$(date -Is)"
SUPEOF
chmod +x "$SUP"
nohup bash "$SUP" > "$LOG" 2>&1 &
echo "supervisor launched (pid $!): ${NODES} nodes, ${NUM_ENV_RUNNERS} runners"
ln -sf "$LOG" "${STATE_DIR}/coord-${RUN}.log"
echo "log: $LOG"
