#!/bin/bash
#SBATCH --job-name=abtest-probe
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
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=code@thiskevin.com
#SBATCH --output=slurm_outputs/abtest-probe-%j.out

# Q2 pure-env glitch-recovery A/B (no Ray/PPO): one GPU, sweep M x recovery_mode.
# A ThreadedVectorEnv joins all M threads per step, so a glitchy sub-env stalls
# the step (within-runner barrier analog). Compare sps + stall tax between
# escalate (current) and in_place (cheap context recycle at the source).
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp

for M in 16 32; do
  for MODE in escalate in_place; do
    echo "===== PROBE M=$M mode=$MODE ====="
    uv run --no-sync python scripts/probe_recovery.py "$M" 300 "$MODE" 2>&1 | tail -30
    echo
  done
done
echo "=== abtest-probe done ==="
