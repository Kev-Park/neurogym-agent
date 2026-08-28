#!/bin/bash
#SBATCH --job-name=rendersvc
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=44
#SBATCH --mem=100G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# Experiment 3: per-node render+encode service (batched renderer + per-node
# DINO), K state-machine clients over tiny-message queues. K=32 and K=64.
set -e
cd /scratch/kp0374/wt/neurogym-agent-native
export PYTHONUNBUFFERED=1
uv run --no-sync python native/probe_render_service.py --k 32 --secs 90
uv run --no-sync python native/probe_render_service.py --k 64 --secs 90
echo "RENDERSVC-COMPLETE"
