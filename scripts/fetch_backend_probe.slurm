#!/bin/bash
#SBATCH --job-name=fetch-probe
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_outputs/fetch-probe-%j.out
#
# CPU-ONLY (no --gres) validation of the Ray fetch backend: parity (actor decode ==
# process decode, tile + mesh) + affinity (same actor reuses chunk LRU). Runs in a
# clean SLURM allocation (login node can't run Ray workers) on an idle CPU node
# (e.g. the 144-core sarek-r27 boxes), so it schedules without a GPU.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
export PYTHONUNBUFFERED=1
# Ray>=2.43 under `uv run` auto-uploads the 1.2GB CWD as runtime_env (stalls actor
# startup, >512MB limit). Shared /scratch => no upload needed. (see m3_ray_multi.)
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
echo "[fetch-probe] node=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
  uv run --no-sync python /scratch/kp0374/wt/neurogym-throughput-scaling/scripts/fetch_backend_probe.py
echo "[fetch-probe] done"
