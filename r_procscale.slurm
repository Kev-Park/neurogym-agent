#!/usr/bin/env bash
#SBATCH --job-name=procscale
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --nodelist=sarekl15-4
#SBATCH --gres=gpu:3090:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=150G
#SBATCH --time=0:25:00
#SBATCH --output=slurm_outputs/procscale-%j.out

# Per-node process-scaling + GPU-scaling sweep for the screenshot-readback
# ceiling. Short --time so the scheduler can backfill it past the highpri
# reservation holding the good fleet. Runs on a HW-confirmed vulkan-good node.
cd /scratch/kp0374/neurogym-agent
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python scripts/probe_procscale.py
