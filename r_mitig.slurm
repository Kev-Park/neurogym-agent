#!/bin/bash
#SBATCH --job-name=mitig
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --gres=gpu:3090:1
# M4: sarekl15-8 EXCLUDED (2x-slower runs, event-log propagation failures,
# prior No-CUDA actor deaths). sarekl15-1 added (probed GOOD 2026-07-08).
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-5,sarekl15-7,sarekl16-1,sarekl16-3,sarekl16-5,sarekl16-6,sarekl16-7,sarekl16-8
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8
#SBATCH --time=05:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=code@thiskevin.com
#SBATCH --output=slurm_outputs/mitig-%x-%j.out

# Mitigation sweep (2026-08): pull unlucky runs up to the peak by killing
# straggler causes measured in the storm trials. Arms (submit with --export):
#   base : auto/4000/600, no stagger        (fair baseline w/ M4 node pool)
#   m1   : + --stagger-first-episode        (M1a: kill synchronized reset waves)
#   m1m2 : m1 + FRAG=8 BATCH=5376 TIMEOUT=60 (+ tuned small-round sampler)
# All arms run with NGLLIB_EVENT_LOG instrumentation (M3 per-step timing rides
# along) + 1Hz node samplers, so wave-death and F4 attribution come for free.
ARM="${ARM:-base}"
SEED="${SEED:-1}"
ITERS="${ITERS:-150}"
RUN_NAME="mitig-${ARM}-s${SEED}"

STAGGER_FLAG=""; FRAG="auto"; BATCH="4000"; TIMEOUT="600"
case "$ARM" in
  base) ;;
  m1)   STAGGER_FLAG="--stagger-first-episode" ;;
  m1m2) STAGGER_FLAG="--stagger-first-episode"; FRAG=8; BATCH=5376; TIMEOUT=60 ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac

cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
CKPT_DIR="/scratch/kp0374/checkpoints/${RUN_NAME}"
EVDIR="/scratch/kp0374/neurogym-agent/event_logs/${RUN_NAME}"
mkdir -p "$EVDIR"
export NGLLIB_EVENT_LOG="$EVDIR/ev-{host}-{pid}.jsonl"

echo "=== ARM: ${RUN_NAME} stagger='${STAGGER_FLAG}' frag=${FRAG} batch=${BATCH} timeout=${TIMEOUT} iters=${ITERS} ==="
echo "=== instrumentation -> $EVDIR ==="

NODES=($(scontrol show hostnames "$SLURM_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
echo "nodes: ${NODES[*]}"; echo "head : $HEAD_NODE ($HEAD_IP)"

# 1Hz per-node resource samplers (background srun steps).
SAMPLER_PIDS=()
for n in "${NODES[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "$n" \
        bash scripts/node_sampler.sh "$EVDIR/gpu-$n.csv" &
    SAMPLER_PIDS+=($!)
done

# Export NGLLIB_EVENT_LOG INSIDE each srun's shell so every node's raylet (and
# thus its EnvRunner actors) inherits it (plain srun propagation missed a node).
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
    uv run --no-sync ray status --address="$HEAD_IP:6379" 2>&1 | head -30 || true

set +e
export UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp RAY_ADDRESS="$HEAD_IP:6379"
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -c \
    "export NGLLIB_EVENT_LOG='$NGLLIB_EVENT_LOG'; \
     uv run --no-sync python -m ngllib_agent.train \
        --run-name '$RUN_NAME' \
        --iters '$ITERS' \
        --num-env-runners 6 \
        --train-batch-size '$BATCH' \
        --rollout-fragment-length '$FRAG' \
        --sample-timeout-s '$TIMEOUT' \
        $STAGGER_FLAG \
        --checkpoint-dir '$CKPT_DIR' \
        --checkpoint-every 25 \
        --wandb-mode offline"
PPO_EXIT=$?

echo "=== stopping ray + samplers ==="
for n in "${NODES[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "$n" uv run --no-sync ray stop --force 2>&1 | tail -1 || true
done
kill "$HEAD_PID" 2>/dev/null || true
for pid in "${WORKER_PIDS[@]}" "${SAMPLER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
wait 2>/dev/null || true

echo "=== mitig ${RUN_NAME} done (train exit=${PPO_EXIT}) ==="
exit "$PPO_EXIT"
