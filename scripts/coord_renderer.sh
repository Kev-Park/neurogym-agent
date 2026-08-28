#!/bin/bash
# Renderer role for coordinator-driven distributed PPO.
#
# Runs INSIDE a `srun --overlap` step spawned by the M4 coordinator on a
# non-head node. Responsibilities:
#   1. Poll $RAY_HEAD_ENDPOINT_FILE for the learner's Ray head address.
#   2. `ray start --address=<endpoint> --block` to join the cluster and stay up.
#
# `--block` keeps this script (and thus the srun step) alive until Ray shuts
# down or the coordinator sends SIGTERM. The coordinator's cleanup handles the
# actual scancel.

set -e
cd "${COORD_WORKDIR:-/scratch/kp0374/neurogym-agent}"
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

ENDPOINT_FILE=${RAY_HEAD_ENDPOINT_FILE:-/scratch/kp0374/coord-state/ray_head_endpoint.txt}
ENDPOINT=""

echo "[renderer] $(hostname) waiting for $ENDPOINT_FILE"
for i in $(seq 1 60); do
    if [ -s "$ENDPOINT_FILE" ]; then
        ENDPOINT=$(cat "$ENDPOINT_FILE")
        echo "[renderer] found endpoint after ${i}x2s: $ENDPOINT"
        break
    fi
    sleep 2
done
if [ -z "$ENDPOINT" ]; then
    echo "[renderer] FATAL: no endpoint after 120s"
    exit 1
fi

echo "[renderer] joining Ray at $ENDPOINT"
# RAY_NUM_CPUS: advertise enough cores for 2 EnvRunner procs (32 browsers) on
# this renderer node under the pinned 2x16 topology (was hardcoded 6). Env-override.
exec uv run --no-sync ray start \
    --address="$ENDPOINT" \
    --num-cpus="${RAY_NUM_CPUS:-44}" \
    --num-gpus=1 \
    --block
