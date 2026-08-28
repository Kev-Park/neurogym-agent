#!/usr/bin/env bash
# Multi-node SERVICE-architecture launcher (stress test 2): N renderer-packed
# nodes (per-node render+encode service + state-machine clients) + learner on
# the head GPU, via the production-validated coordinator (re-salloc/respawn).
#
#   bash scripts/launch_coord_svc.sh [run-name] [renderer-nodes] [target-iters]
#
# Learning characteristics are v9-identical (configs/native_service.yaml);
# only the plumbing differs. Per renderer node: 1 service actor (GPU) + its
# share of the client runners. num_cpus_per_env_runner=2.5 forces runner
# SPREAD (44 cpus/node => ~17 runners max per node; we place 16/node).
set -u
cd /scratch/kp0374/wt/neurogym-agent-native

RUN=${1:-native-svc-mn}
RENDERERS=${2:-2}
TARGET_ITERS=${3:-740}
NODES=$((RENDERERS + 1))
NUM_ENV_RUNNERS=$((NODES * 16))
CKPT=/scratch/kp0374/checkpoints/${RUN}
STATE_DIR=/scratch/kp0374/coord-state
mkdir -p "$STATE_DIR" "$CKPT"

export RAY_NUM_CPUS=44
T0_FILE="$CKPT/curriculum_t0"
[ -f "$T0_FILE" ] || date +%s > "$T0_FILE"
export CURRICULUM_T0=$(cat "$T0_FILE")
export RAY_HEAD_ENDPOINT_FILE="${STATE_DIR}/ray_head_endpoint-${RUN}.txt"
export NUM_RENDERERS="$RENDERERS"
export SAMPLE_TIMEOUT_S="${SAMPLE_TIMEOUT_S:-600}"
export WORKLOAD_CMD="uv run --no-sync python -m ngllib_agent.train \
  --config configs/native_service.yaml \
  --run-name ${RUN} --render-service --learner-gpu \
  --num-env-runners ${NUM_ENV_RUNNERS} --num-envs-per-env-runner 4 \
  --num-gpus-per-env-runner 0 --num-cpus-per-env-runner 2.5 \
  --vector threads \
  --iters ${TARGET_ITERS} --train-batch-size 4000 \
  --sample-timeout-s ${SAMPLE_TIMEOUT_S} \
  --checkpoint-dir ${CKPT} --checkpoint-every 10 \
  --wandb-project neurogym-agent --resume"

LOG="${STATE_DIR}/coord-${RUN}-$(date +%Y%m%d-%H%M%S).log"
SUP="${STATE_DIR}/supervisor-${RUN}.sh"
cat > "$SUP" <<SUPEOF
#!/bin/bash
cd /scratch/kp0374/wt/neurogym-agent-native
export RAY_NUM_CPUS="${RAY_NUM_CPUS}"
export CURRICULUM_T0="${CURRICULUM_T0}"
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
    --salloc-cpus-per-node 48 \\
    --salloc-mem 200G \\
    --exclude sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5 \\
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
