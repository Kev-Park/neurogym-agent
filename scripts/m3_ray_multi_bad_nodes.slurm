#!/bin/bash
#SBATCH --job-name=m3-ray-badnodes
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:3090:1                  # any GPU; we're not rendering
#SBATCH --nodelist=sarekl15-3,sarekl15-6   # known-bad for HW rendering, fine for Ray
#SBATCH --time=00:15:00
#SBATCH --output=slurm_outputs/m3-ray-badnodes-%j.out

# Milestone 3 preview: form a 2-node Ray cluster and verify cross-node
# membership + object store. Uses Mesa-only nodes intentionally — no HW
# rendering needed, and vulkan-capable nodes are congested by highpri jobs.
# Follows the standard Ray-on-SLURM pattern (head on first node, workers
# join by address).
set -e
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
# Ray >=2.43 under `uv run` auto-uploads CWD as runtime_env (1.2 GB here > 512 MB
# limit). All nodes see /scratch/ over shared FS, so no upload is needed. Matches
# the pattern in scripts/ppo_smoke.py.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

NODES=($(scontrol show hostnames "$SLURM_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
echo "=== allocation ==="
echo "nodes: ${NODES[*]}"
echo "head : $HEAD_NODE ($HEAD_IP)"

# Start head daemon on the first node. --block keeps the srun alive so the
# daemon isn't reaped when srun exits; backgrounded so we can launch workers.
# --overlap on ALL subsequent sruns so they can share the same node allocations
# (default srun steps are exclusive within a job — Ray blocks would otherwise
# lock the nodes and any probe srun would fail with "Requested nodes are busy").
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    uv run --no-sync ray start --head \
        --node-ip-address="$HEAD_IP" \
        --port=6379 \
        --num-cpus=4 \
        --num-gpus=0 \
        --block &
HEAD_PID=$!
sleep 15

# Start worker daemons on the other nodes.
WORKER_PIDS=()
for ((i=1; i<${#NODES[@]}; i++)); do
    srun --overlap --nodes=1 --ntasks=1 -w "${NODES[$i]}" \
        uv run --no-sync ray start \
            --address="$HEAD_IP:6379" \
            --num-cpus=4 \
            --num-gpus=0 \
            --block &
    WORKER_PIDS+=($!)
done
sleep 15

echo "=== running probe on head ==="
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    uv run --no-sync python scripts/ray_multi_node_probe.py \
        --head-ip "$HEAD_IP" \
        --expected-nodes ${#NODES[@]}
PROBE_EXIT=$?

echo "=== stopping ray on all nodes ==="
for n in "${NODES[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "$n" uv run --no-sync ray stop --force 2>&1 | tail -2 || true
done

# Reap the backgrounded ray-start sruns.
kill "$HEAD_PID" 2>/dev/null || true
for pid in "${WORKER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
wait 2>/dev/null || true

echo "=== m3 badnodes done (probe exit=$PROBE_EXIT) ==="
exit $PROBE_EXIT
