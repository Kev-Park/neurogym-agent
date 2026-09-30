#!/bin/bash
#SBATCH --job-name=zl-encgraph
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=00:10:00
#SBATCH --output=slurm_outputs/slurm-%x-%j.out
#
# Validate + bench the ported encode() CUDA-graph path (parity gate before sweeps).
set -eu
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_outputs
export PYTHONUNBUFFERED=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
echo "[enc-graph] node=$(hostname)"
uv run --no-sync python native/probe_encoder_graph.py
