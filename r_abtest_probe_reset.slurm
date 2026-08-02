#!/bin/bash
#SBATCH --job-name=abtest-probe-reset
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:3090:1
#SBATCH --nodelist=sarekl15-2,sarekl15-4,sarekl15-5,sarekl15-7,sarekl15-8,sarekl16-1,sarekl16-3,sarekl16-5,sarekl16-6,sarekl16-7,sarekl16-8
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=code@thiskevin.com
#SBATCH --output=slurm_outputs/abtest-probe-reset-%j.out

# Q2 RESET-PATH probe A/B (the variant): hammers _navigate_with_retry — the path
# recovery_mode actually changes — where the step-path probe saw nothing.
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp

for M in 16 32; do
  for MODE in escalate in_place; do
    echo "===== RESET-PROBE M=$M mode=$MODE ====="
    uv run --no-sync python scripts/probe_recovery_reset.py "$M" 60 "$MODE" 2>&1 | tail -40
    echo
  done
done
echo "=== abtest-probe-reset done ==="
