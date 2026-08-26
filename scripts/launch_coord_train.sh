#!/usr/bin/env bash
# Ready-to-run coordinator launcher for the PINNED 2x16 topology — a HANDS-OFF
# long run with auto re-salloc + respawn across preemption AND the salloc wall
# (removes the manual timeout-resubmit toil of the plain sbatch path).
#
# PRODUCTION-VALIDATED end-to-end by coord-test-v7 (2026-08-18, REFINEMENT R9):
# 370/370 iters autonomous incl. a live salloc-wall re-salloc handoff. This is
# the DEFAULT path for real RL runs; plain sbatch remains for throwaway probes.
#
#   bash scripts/launch_coord_train.sh [run-name] [renderer-nodes] [target-iters]
#
# Topology: salloc (renderers+1) nodes; every node's GPU hosts 2 EnvRunner procs
# x 16 threaded envs (learner is num_learners=0, CPU-in-driver on the head), so
#   num_env_runners = 2 * (renderers + 1)
# RAY_NUM_CPUS=44 lets Ray place both runners per node (coord_learner/renderer).
# The coordinator tears down at --target-iterations (reads train's meta.json);
# v7 measured steps/iter ~= 4080 at --train-batch-size 4000, so N steps needs
# target-iters ~= N/4080 (v7's 370 = 1.6M, not the 2M an older estimate said).
set -u
cd /scratch/kp0374/neurogym-agent

RUN=${1:-val-2x16-coord}
RENDERERS=${2:-2}
TARGET_ITERS=${3:-370}                       # ~2M steps at ~5450 steps/iter
NODES=$((RENDERERS + 1))
NUM_ENV_RUNNERS=$((NODES * 2))
CKPT=/scratch/kp0374/checkpoints/${RUN}
STATE_DIR=/scratch/kp0374/coord-state
mkdir -p "$STATE_DIR" "$CKPT"

export RAY_NUM_CPUS=44
export RAY_HEAD_ENDPOINT_FILE="${STATE_DIR}/ray_head_endpoint-${RUN}.txt"
export NUM_RENDERERS="$RENDERERS"
# Long sample timeout ON PURPOSE (2026-08-17): with the watchdog tree-kill
# bounding real hangs, a recovery just prices ONE long iteration and the
# rounds re-sync. A TIGHT timeout (90s) instead creates a terminal
# phase-lock: timed-out sample() calls stay in flight, new calls queue
# behind them, and every runner stays one-call-behind forever (v2 282s /
# v3 552s locks, reproducible, never self-recovered).
export SAMPLE_TIMEOUT_S="${SAMPLE_TIMEOUT_S:-600}"
# WORKLOAD_CMD: coord_learner.sh runs this verbatim after the Ray cluster is up
# (inherits RAY_ADDRESS). --iters is huge; the coordinator stops it at
# --target-iterations. train.py writes meta.json in --checkpoint-dir each iter.
export WORKLOAD_CMD="uv run --no-sync python -m ngllib_agent.train \
  --run-name ${RUN} --num-env-runners ${NUM_ENV_RUNNERS} \
  --iters 100000 --train-batch-size 4000 --sample-timeout-s ${SAMPLE_TIMEOUT_S} \
  --checkpoint-dir ${CKPT} --checkpoint-every 10 \
  --wandb-project neurogym-agent --resume"

# PARTITION=highpri when the preempt partition is starved (established practice:
# no preemption of running jobs, we queue for the next opening). SALLOC_TIME
# should be ~run-length + margin, not a blanket 24h hold.
#
# Supervisor + timestamped logs (2026-08-26): an earlier coordinator died
# unexplained AND the relaunch truncated the only log, destroying the
# forensics. Every launch now gets its own log file (coord-${RUN}.log is a
# symlink to the latest), and a supervisor restarts the coordinator on
# nonzero exit (up to 5 attempts). Exit 0 = clean/target-reached = stop.
# Deep forensics: ${STATE_DIR}/flight-${RUN}.jsonl + /tmp/coordflight-${RUN}.jsonl
# (heartbeats, exceptions, exit reasons) and /tmp/coordfault-${RUN}.log
# (faulthandler, hard crashes).
LOG="${STATE_DIR}/coord-${RUN}-$(date +%Y%m%d-%H%M%S).log"
SUP="${STATE_DIR}/supervisor-${RUN}.sh"
cat > "$SUP" <<SUPEOF
#!/bin/bash
cd /scratch/kp0374/neurogym-agent
export RAY_NUM_CPUS="${RAY_NUM_CPUS}"
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
    --salloc-time "${SALLOC_TIME:-24:00:00}" \\
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

echo "supervisor launched (pid $!): ${NODES} nodes, num_env_runners=${NUM_ENV_RUNNERS}"
ln -sf "$LOG" "${STATE_DIR}/coord-${RUN}.log"
echo "log: $LOG (symlinked at ${STATE_DIR}/coord-${RUN}.log)"
