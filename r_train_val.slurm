#!/bin/bash
#SBATCH --job-name=val2x16
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --gres=gpu:3090:1
# Vulkan-good allowlist (CLAUDE.md "Node quirks"); SLURM picks 3 free ones.
#SBATCH --nodelist=sarekl15-2,sarekl15-4,sarekl15-5,sarekl15-7,sarekl15-8,sarekl16-1,sarekl16-3,sarekl16-5,sarekl16-6,sarekl16-7,sarekl16-8
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4
#SBATCH --time=07:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=code@thiskevin.com
#SBATCH --output=slurm_outputs/val2x16-%j.out

# Systems-validation run: multi-node PINNED 2x16 topology (2 EnvRunner procs x 16
# threaded envs per GPU node), 3 nodes -> 6 runners x 16 = 96 envs, 2M env-steps.
# Learner is num_learners=0 (CPU, in driver); DINO is env-side. --resume auto-set
# so a relaunch after preemption/bug continues from the latest checkpoint.
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

CKPT_DIR=/scratch/kp0374/checkpoints/val-2x16-mn

NODES=($(scontrol show hostnames "$SLURM_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
echo "nodes: ${NODES[*]}"
echo "head : $HEAD_NODE ($HEAD_IP)"

# Ray head on node 0.
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    uv run --no-sync ray start --head --node-ip-address="$HEAD_IP" --port=6379 \
        --num-cpus=44 --num-gpus=1 --block &
HEAD_PID=$!
sleep 20

# Ray workers on the rest.
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
# Export env for the train step (srun inherits via --export=ALL). Do NOT use the
# `env` command — uv installs a non-executable ~/.local/bin/env that shadows
# /usr/bin/env on PATH and fails execve.
export UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp RAY_ADDRESS="$HEAD_IP:6379"
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    uv run --no-sync python -m ngllib_agent.train \
        --run-name val-2x16-mn \
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

echo "=== val2x16 done (train exit=${PPO_EXIT}) ==="
exit "$PPO_EXIT"
