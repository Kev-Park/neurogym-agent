#!/bin/bash
#SBATCH --job-name=zmaxleft-parity
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=00:50:00
#SBATCH --output=slurm_outputs/zmaxleft-parity-%j.out

# zmax-left gate 1: Chrome vs simulator LIGHT parity probe for left-pane
# interaction SEQUENCES (neighbor select / toggle / SHOW_ALL / hop move).
# Needs a vulkan-good node (Chrome HW rendering) — exclude list per CLAUDE.md.
set -u
cd /scratch/kp0374/wt/neurogym-agent-zmax-left
export PYTHONUNBUFFERED=1
mkdir -p zmaxleft_parity_out

echo "=== node: $(hostname) ==="
EM_OUT=zmaxleft_parity_out N_STATES=6 SEED=7 \
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
  uv run --no-sync python scripts/zmaxleft_select_parity.py
echo "rc=$?"
