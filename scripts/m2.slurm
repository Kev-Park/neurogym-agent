#!/bin/bash
#SBATCH --job-name=m2-multi-runner
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # driver + 2 remote env_runners × ~2 cpu each + headroom
#SBATCH --mem=48G                  # 2 Chromes × ~10G + Ray + Python + margin
#SBATCH --gres=gpu:3090:1
#SBATCH --nodelist=sarekl15-4      # vulkan-capable; see CLAUDE.md "Node quirks"
#SBATCH --time=00:30:00
#SBATCH --output=slurm_outputs/m2-multi-runner-%j.out

# Milestone 2: single-node multi-worker PPO. Verifies that 2 remote Ray-actor
# env runners (each with its own Chrome) sample in parallel, the driver-side
# learner collects + trains + broadcasts, and PPO loss moves.
set -e
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

echo "=== ppo_smoke --num-env-runners 2 (5 iters, 512-sample batch) ==="
uv run --no-sync python scripts/ppo_smoke.py \
    --num-env-runners 2 \
    --iters 5 \
    --train-batch-size 512

echo "=== m2 done ==="
