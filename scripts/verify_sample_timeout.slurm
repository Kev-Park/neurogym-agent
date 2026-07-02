#!/bin/bash
#SBATCH --job-name=verify-sample-timeout
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:3090:1
#SBATCH --nodelist=sarekl15-2,sarekl15-4,sarekl15-5,sarekl15-7,sarekl15-8,sarekl16-1,sarekl16-3,sarekl16-5,sarekl16-6,sarekl16-7,sarekl16-8
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4
#SBATCH --time=01:00:00
#SBATCH --output=slurm_outputs/verify-sample-timeout-%j.out

# Verify the sample_timeout_s fix: 30 iters at rollout-fragment-length=128 with
# sample-timeout-s=600. If the fix works we should see valid num_steps + real
# loss for MOST iters (a few nan on browser restarts is expected). Also
# captures wall time per iter for throughput comparison against legacy.
set -e
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

NODES=($(scontrol show hostnames "$SLURM_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
echo "=== allocation ==="
echo "nodes: ${NODES[*]}"
echo "head : $HEAD_NODE ($HEAD_IP)"

srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    uv run --no-sync ray start --head \
        --node-ip-address="$HEAD_IP" --port=6379 \
        --num-cpus=6 --num-gpus=1 --block &
HEAD_PID=$!
sleep 15

WORKER_PIDS=()
for ((i=1; i<${#NODES[@]}; i++)); do
    srun --overlap --nodes=1 --ntasks=1 -w "${NODES[$i]}" \
        uv run --no-sync ray start \
            --address="$HEAD_IP:6379" --num-cpus=6 --num-gpus=1 --block &
    WORKER_PIDS+=($!)
done
sleep 15

echo "=== running verification (30 iters, rfl=128, timeout=600) ==="
export RAY_ADDRESS="$HEAD_IP:6379"
T0=$SECONDS
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    uv run --no-sync python scripts/ppo_smoke.py \
        --num-env-runners 2 \
        --iters 30 \
        --train-batch-size 512 \
        --sample-timeout-s 600 \
        --rollout-fragment-length 128
PPO_EXIT=$?
T_ELAPSED=$((SECONDS - T0))
echo "=== PPO wallclock: ${T_ELAPSED}s ==="

for n in "${NODES[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "$n" uv run --no-sync ray stop --force 2>&1 | tail -2 || true
done

kill "$HEAD_PID" 2>/dev/null || true
for pid in "${WORKER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
wait 2>/dev/null || true

echo "=== verify done (exit=$PPO_EXIT, elapsed=${T_ELAPSED}s) ==="
exit $PPO_EXIT
