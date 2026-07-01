#!/bin/bash
#SBATCH --job-name=m3-ext-ppo
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:3090:1
#SBATCH --nodelist=sarekl15-2,sarekl15-4,sarekl15-5,sarekl15-7,sarekl15-8,sarekl16-1,sarekl16-3,sarekl16-5,sarekl16-6,sarekl16-7,sarekl16-8
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4
#SBATCH --time=00:45:00
#SBATCH --output=slurm_outputs/m3-ext-ppo-%j.out

# Milestone 3 extended: same shape as m3_real_ppo.slurm but longer horizon
# with browser_restart_every=5, so the Playwright refresh mechanism fires
# repeatedly during training. Verifies Chrome is actively cycling (not a
# stale one-time launch) and training survives multiple restarts per env_runner.
# Expected: ~3 restarts per env_runner over 20 iterations.
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
        --node-ip-address="$HEAD_IP" \
        --port=6379 \
        --num-cpus=6 \
        --num-gpus=1 \
        --block &
HEAD_PID=$!
sleep 15

WORKER_PIDS=()
for ((i=1; i<${#NODES[@]}; i++)); do
    srun --overlap --nodes=1 --ntasks=1 -w "${NODES[$i]}" \
        uv run --no-sync ray start \
            --address="$HEAD_IP:6379" \
            --num-cpus=6 \
            --num-gpus=1 \
            --block &
    WORKER_PIDS+=($!)
done
sleep 15

echo "=== ray cluster status ==="
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    uv run --no-sync ray status --address="$HEAD_IP:6379" 2>&1 | head -30 || true

echo "=== running extended ppo_smoke (20 iters, browser_restart_every=5) ==="
export RAY_ADDRESS="$HEAD_IP:6379"
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    uv run --no-sync python scripts/ppo_smoke.py \
        --num-env-runners 2 \
        --iters 20 \
        --train-batch-size 512 \
        --browser-restart-every 5
PPO_EXIT=$?

echo "=== stopping ray on all nodes ==="
for n in "${NODES[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "$n" uv run --no-sync ray stop --force 2>&1 | tail -2 || true
done

kill "$HEAD_PID" 2>/dev/null || true
for pid in "${WORKER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
wait 2>/dev/null || true

echo "=== m3 extended done (ppo exit=$PPO_EXIT) ==="
exit $PPO_EXIT
