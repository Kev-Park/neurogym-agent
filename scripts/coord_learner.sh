#!/bin/bash
# Learner role for coordinator-driven distributed PPO.
#
# Runs INSIDE a `srun --overlap` step spawned by the M4 coordinator.
# Responsibilities:
#   1. Start Ray head on this node.
#   2. Write "IP:PORT" to $RAY_HEAD_ENDPOINT_FILE (renderers poll for it).
#   3. Wait for renderers to join (short poll on ray status).
#   4. Run ppo_smoke.py with RAY_ADDRESS set — it connects to the local head.
#   5. Print ray status snapshot before PPO (diagnostic #3: verify multi-node
#      cluster + per-node actor placement).
#   6. On PPO completion, exec `sleep` so the process stays alive until the
#      coordinator SIGTERMs us. Prevents the coordinator from interpreting the
#      clean exit as a death event and respawning (which would fail because
#      Ray head is already torn down).
#
# Env vars honored:
#   RAY_HEAD_ENDPOINT_FILE  path to endpoint dropfile (default:
#                           /scratch/kp0374/coord-state/ray_head_endpoint.txt)
#   NUM_RENDERERS           expected renderer count (used to wait for joins)
#   NUM_ITERS               PPO iters (default 3 — smoke)
#   TRAIN_BATCH             PPO train_batch_size (default 512)
#   BROWSER_RESTART_EVERY   override for env.browser_restart_every (unset -> ngllib default)

set -e
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
# Skip Ray auto-CWD upload (see scripts/ppo_smoke.py comment).
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

ENDPOINT_FILE=${RAY_HEAD_ENDPOINT_FILE:-/scratch/kp0374/coord-state/ray_head_endpoint.txt}
NUM_RENDERERS=${NUM_RENDERERS:-1}
NUM_ITERS=${NUM_ITERS:-3}
TRAIN_BATCH=${TRAIN_BATCH:-512}
# Rollout / sample tuning (see scripts/ppo_smoke.py comments): these avoid the
# "No samples returned from remote workers" nan cascade at ~0.5s/step.
SAMPLE_TIMEOUT_S=${SAMPLE_TIMEOUT_S:-600}
ROLLOUT_FRAGMENT_LENGTH=${ROLLOUT_FRAGMENT_LENGTH:-auto}
EXPECTED_NODES=$((NUM_RENDERERS + 1))

HEAD_IP=$(hostname -I | awk '{print $1}')
echo "[learner] $(hostname) ($HEAD_IP)  expecting $EXPECTED_NODES total nodes"

# Start Ray head. It daemonizes and returns; no --block.
uv run --no-sync ray start --head \
    --node-ip-address="$HEAD_IP" \
    --port=6379 \
    --num-cpus=6 \
    --num-gpus=1

# Publish endpoint atomically (tmp + rename on same fs).
mkdir -p "$(dirname "$ENDPOINT_FILE")"
echo "$HEAD_IP:6379" > "${ENDPOINT_FILE}.tmp"
mv "${ENDPOINT_FILE}.tmp" "$ENDPOINT_FILE"
echo "[learner] endpoint published: $HEAD_IP:6379 -> $ENDPOINT_FILE"

# Wait for renderers to join.
export RAY_ADDRESS="$HEAD_IP:6379"
for i in $(seq 1 30); do
    NODES=$(uv run --no-sync ray status --address="$RAY_ADDRESS" 2>/dev/null \
            | grep -c '^ [0-9]* node_' || echo 0)
    if [ "$NODES" -ge "$EXPECTED_NODES" ]; then
        echo "[learner] all $EXPECTED_NODES nodes joined (iter $i)"
        break
    fi
    sleep 2
done

# Diagnostic #3: cluster snapshot before PPO.
echo "[learner] ==== ray status pre-PPO ===="
uv run --no-sync ray status --address="$RAY_ADDRESS" || true
echo "[learner] ==== end ray status ===="

# Build ppo_smoke.py args.
PPO_ARGS=(
    --num-env-runners "$NUM_RENDERERS"
    --iters "$NUM_ITERS"
    --train-batch-size "$TRAIN_BATCH"
    --sample-timeout-s "$SAMPLE_TIMEOUT_S"
    --rollout-fragment-length "$ROLLOUT_FRAGMENT_LENGTH"
)
if [ -n "${BROWSER_RESTART_EVERY:-}" ]; then
    PPO_ARGS+=(--browser-restart-every "$BROWSER_RESTART_EVERY")
fi

if [ -n "${WORKLOAD_CMD:-}" ]; then
    # Real-training path: run the given command verbatim (e.g. ngllib_agent.train).
    # Inherits RAY_ADDRESS and connects to the cluster built above.
    echo "[learner] running workload: $WORKLOAD_CMD"
    bash -c "$WORKLOAD_CMD" || echo "[learner] WARN: workload exited $?"
else
    echo "[learner] running ppo_smoke.py ${PPO_ARGS[*]}"
    uv run --no-sync python scripts/ppo_smoke.py "${PPO_ARGS[@]}" \
        || echo "[learner] WARN: ppo_smoke exited $?"
fi

echo "[learner] PPO complete; sleeping until coordinator SIGTERM"
# Do NOT ray stop — coord's teardown will scancel the allocation, which
# takes everything down cleanly. Keep this process alive so coord's respawn
# logic doesn't fire.
exec sleep 3600
