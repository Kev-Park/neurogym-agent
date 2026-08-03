#!/bin/bash
#SBATCH --job-name=storm-trial
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --gres=gpu:3090:1
#SBATCH --nodelist=sarekl15-2,sarekl15-4,sarekl15-5,sarekl15-7,sarekl15-8,sarekl16-1,sarekl16-3,sarekl16-5,sarekl16-6,sarekl16-7,sarekl16-8
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4
#SBATCH --time=05:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=code@thiskevin.com
#SBATCH --output=slurm_outputs/storm-trial-%j.out

# Heavily-instrumented 2x16 escalate trial to capture what triggers the FIRST
# reset before a glitch storm. Emits per-env reset/glitch JSONL (NGLLIB_EVENT_LOG)
# + per-node 1Hz GPU/load CSV + train.py iter/checkpoint MARK lines. escalate
# baseline (the config that storms). NOTHING is mitigated here — data only.
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

RUN=storm-trial-$SLURM_JOB_ID
EVDIR=/scratch/kp0374/neurogym-agent/event_logs/$RUN
mkdir -p "$EVDIR"
# EnvRunner actors inherit this from their node's raylet (srun propagates env).
export NGLLIB_EVENT_LOG="$EVDIR/ev-{host}-{pid}.jsonl"
echo "=== instrumentation -> $EVDIR ==="

NODES=($(scontrol show hostnames "$SLURM_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
echo "nodes: ${NODES[*]}"; echo "head : $HEAD_NODE ($HEAD_IP)"

# --- per-node 1Hz resource samplers (background) ---
SAMPLER_PIDS=()
for n in "${NODES[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "$n" \
        bash scripts/node_sampler.sh "$EVDIR/gpu-$n.csv" &
    SAMPLER_PIDS+=($!)
done

# --- ray cluster ---
# Export NGLLIB_EVENT_LOG INSIDE each srun's shell so every node's raylet (and
# thus its EnvRunner actors) inherits it — relying on srun env-propagation alone
# missed a worker node last run (only 4/6 runners logged).
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -c \
    "export NGLLIB_EVENT_LOG='$NGLLIB_EVENT_LOG'; \
     uv run --no-sync ray start --head --node-ip-address='$HEAD_IP' --port=6379 \
        --num-cpus=44 --num-gpus=1 --block" &
HEAD_PID=$!
sleep 20
WORKER_PIDS=()
for ((i=1; i<${#NODES[@]}; i++)); do
    srun --overlap --nodes=1 --ntasks=1 -w "${NODES[$i]}" bash -c \
        "export NGLLIB_EVENT_LOG='$NGLLIB_EVENT_LOG'; \
         uv run --no-sync ray start --address='$HEAD_IP:6379' \
            --num-cpus=44 --num-gpus=1 --block" &
    WORKER_PIDS+=($!)
done
sleep 20

echo "=== ray status ==="
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    uv run --no-sync ray status --address="$HEAD_IP:6379" 2>&1 | head -40 || true

set +e
export UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp RAY_ADDRESS="$HEAD_IP:6379"
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    uv run --no-sync python -m ngllib_agent.train \
        --run-name "$RUN" \
        --iters 120 \
        --num-env-runners 6 \
        --train-batch-size 4000 \
        --rollout-fragment-length auto \
        --sample-timeout-s 600 \
        --recovery-mode escalate \
        --checkpoint-dir "/scratch/kp0374/checkpoints/$RUN" \
        --checkpoint-every 10 \
        --wandb-mode offline
PPO_EXIT=$?

echo "=== stopping ray + samplers ==="
for n in "${NODES[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "$n" uv run --no-sync ray stop --force 2>&1 | tail -1 || true
    srun --overlap --nodes=1 --ntasks=1 -w "$n" pkill -f node_sampler.sh 2>/dev/null || true
done
kill "$HEAD_PID" "${WORKER_PIDS[@]}" "${SAMPLER_PIDS[@]}" 2>/dev/null || true
wait 2>/dev/null || true
echo "=== storm-trial done (exit=${PPO_EXIT}); events in $EVDIR ==="
exit "$PPO_EXIT"
