#!/usr/bin/env bash
#SBATCH --job-name=e2e
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=0:40:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=code@thiskevin.com
#SBATCH --output=slurm_outputs/e2e-%j.out

# Phase-2 end-to-end: per-env DINO (baseline) vs batched DINO (proto), single
# process AND process-packing sweep. See probe_e2e_batched.py / run_e2e_sweep.sh.
cd /scratch/kp0374/neurogym-agent
echo "===== single-process baseline vs proto ====="
bash scripts/run_e2e_sweep.sh 1 32 baseline
bash scripts/run_e2e_sweep.sh 1 32 proto
echo "===== 2 procs x 16 (the pinned config) ====="
bash scripts/run_e2e_sweep.sh 2 16 baseline
bash scripts/run_e2e_sweep.sh 2 16 proto
echo "===== process sweep on proto (total 32 browsers/GPU) ====="
bash scripts/run_e2e_sweep.sh 4 8 proto
