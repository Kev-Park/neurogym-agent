#!/bin/bash
#SBATCH --job-name=m1-smoke
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:3090:1
#SBATCH --nodelist=sarekl15-4          # vulkan-capable (HW-confirmed); see CLAUDE.md "Node quirks"
#SBATCH --time=00:30:00
#SBATCH --output=slurm_outputs/m1-smoke-%j.out

# Milestone 1 browser-in-the-loop smoke: sanity env loop, then single-process PPO.
# Run from the repo root on the cluster after `uv sync`.
set -e
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

echo "=== [1/2] sanity_env_loop (real Dict obs, 10 steps) ==="
uv run --no-sync python scripts/sanity_env_loop.py --steps 10 --seed 0

echo "=== [2/2] ppo_smoke (new RLlib API, 2 iters, tiny batch) ==="
uv run --no-sync python scripts/ppo_smoke.py --iters 2 --train-batch-size 128

echo "=== smoke done ==="
