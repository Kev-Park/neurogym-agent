#!/bin/bash
#SBATCH --job-name=ngl-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # one per environment
#SBATCH --mem=200G               # Chrome grows ~8-10G/env over long runs; 120G was hitting SLURM cap
#SBATCH --gres=gpu:3090:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=kp0374@princeton.edu
##SBATCH --nodelist=sarekl16-8

export PYTHONUNBUFFERED=1

uv run src/neurogym-agent/train.py \
  --segment_positions segment_positions.parquet \
  --n_envs 16 \
  --total_timesteps 4000000 \
  --wandb_mode online \
  --run_name hierarchical_v1 \
  --batch_size 512