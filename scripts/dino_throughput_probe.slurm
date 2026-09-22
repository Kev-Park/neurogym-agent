#!/bin/bash
#SBATCH --job-name=dino-dcap
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-dcap-%j.out
#
# D_cap upper bound: max DINO encode img/s on ONE dedicated GPU (single process,
# so CUDA-graph capture works), sweeping batch, eager vs graph. If best <= ~382
# img/s (=317/0.83), dedicated-DINO-GPU disaggregation cannot beat shared 317.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
export PYTHONUNBUFFERED=1
mkdir -p slurm_outputs
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
  uv run --no-sync python scripts/dino_throughput_probe.py
