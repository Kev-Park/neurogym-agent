#!/bin/bash
#SBATCH --job-name=dino-graph-parity
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
#SBATCH --output=slurm_outputs/dino-graph-parity-%j.out
#
# DINO-GRAPH-PARITY gate: the CUDA-graph encode_gpu path must be numerically
# identical to eager across distinct frames (and must re-read new pixels, not
# replay a stale capture) before any graph SPS run. Prints eager-vs-graph timing.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
export PYTHONUNBUFFERED=1
mkdir -p slurm_outputs
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
  uv run --no-sync python scripts/dino_graph_probe.py
