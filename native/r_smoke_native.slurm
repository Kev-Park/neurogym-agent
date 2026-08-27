#!/bin/bash
#SBATCH --job-name=nativesmoke
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-7,sarekl16-1,sarekl16-3,sarekl16-5
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# NativeEnvironment end-to-end smoke: random policy, a few pairs, dino obs —
# exercises build_env(backend=native) + full wrapper stack + eval loop.
set -e
WT=/scratch/kp0374/wt/neurogym-agent-native
cd "$WT"
export PYTHONUNBUFFERED=1
uv run --no-sync python scripts/eval_d0.py \
  --config configs/native_eval.yaml \
  --eval-d0 /scratch/kp0374/neurogym-agent/eval_d0_v1.parquet \
  --skeleton /scratch/kp0374/neurogym-agent/segment_positions.parquet \
  --random-policy \
  --obs dino \
  --limit 3 --max-steps 40 --report-budgets 40 \
  --output /scratch/kp0374/native_spike/smoke_native_$SLURM_JOB_ID.json
