#!/bin/bash
#SBATCH --job-name=val2x16v2
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
#SBATCH --time=07:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=code@thiskevin.com
#SBATCH --output=slurm_outputs/val2x16v2-%j.out

# v2 2M run: same multi-node PINNED 2x16 topology, but with the jittered-backoff
# reset recovery (resilient wrapper) that ports the legacy anti-thundering-herd
# behavior. Fresh run-name/ckpt-dir so it starts at iter 0 (clean glitch-rate
# comparison vs val-2x16-mn). --resume auto-set so a relaunch continues.
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

CKPT_DIR=/scratch/kp0374/checkpoints/val-2x16-v2

NODES=($(scontrol show hostnames "$SLURM_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
echo "nodes: ${NODES[*]}"
echo "head : $HEAD_NODE ($HEAD_IP)"

srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    uv run --no-sync ray start --head --node-ip-address="$HEAD_IP" --port=6379 \
        --num-cpus=44 --num-gpus=1 --block &
HEAD_PID=$!
sleep 20

WORKER_PIDS=()
for ((i=1; i<${#NODES[@]}; i++)); do
    srun --overlap --nodes=1 --ntasks=1 -w "${NODES[$i]}" \
        uv run --no-sync ray start --address="$HEAD_IP:6379" \
            --num-cpus=44 --num-gpus=1 --block &
    WORKER_PIDS+=($!)
done
sleep 20

echo "=== ray status ==="
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    uv run --no-sync ray status --address="$HEAD_IP:6379" 2>&1 | head -40 || true

RESUME_FLAG=""
if [ -n "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then RESUME_FLAG="--resume"; fi
echo "=== train.py (resume='${RESUME_FLAG}') ==="

set +e
export UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp RAY_ADDRESS="$HEAD_IP:6379"
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    uv run --no-sync python -m ngllib_agent.train \
        --run-name val-2x16-v2 \
        --iters 500 \
        --num-env-runners 6 \
        --train-batch-size 4000 \
        --rollout-fragment-length auto \
        --sample-timeout-s 600 \
        --checkpoint-dir "$CKPT_DIR" \
        --checkpoint-every 10 \
        --wandb-project neurogym-agent \
        $RESUME_FLAG
PPO_EXIT=$?

echo "=== stopping ray ==="
for n in "${NODES[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "$n" uv run --no-sync ray stop --force 2>&1 | tail -1 || true
done
kill "$HEAD_PID" 2>/dev/null || true
for pid in "${WORKER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
wait 2>/dev/null || true

echo "=== val2x16v2 done (train exit=${PPO_EXIT}) ==="
exit "$PPO_EXIT"
