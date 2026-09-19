#!/bin/bash
#SBATCH --job-name=rb-parity
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/rb-parity-%j.out
#
# Parity gate for the batched atlas renderer: render_batch_probe compares each
# batched-atlas cell to the per-env MeshRenderer.render for identical scenes
# (readback + interop). Must PASS (cells pixel-match) before the SPS sweep.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
export PYTHONUNBUFFERED=1
mkdir -p slurm_outputs rb_out
RB_OUT=rb_out UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
  uv run --no-sync python scripts/render_batch_probe.py
