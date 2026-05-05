#!/bin/bash
#SBATCH --job-name=ngl-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # one per environment
#SBATCH --mem=200G               # Chrome grows ~8-10G/env over long runs; 120G was hitting SLURM cap
#SBATCH --gres=gpu:3090:1
#SBATCH --time=96:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=kp0374@princeton.edu
#SBATCH --nodelist=sarekl15-2

export PYTHONUNBUFFERED=1

uv run src/neurogym-agent/train.py \
  --segment_positions segment_positions.parquet \
  --wandb_mode online \
  --run_name hierarchical_v2 \
  --n_envs 16 \
  --total_timesteps 8000000 \
  --n_steps 2048 \
  --batch_size 32768 \
  --n_epochs 4 \
  --gamma 0.99 \
  --gae_lambda 0.95 \
  --clip_range 0.2 \
  --learning_rate 3e-4 \
  --ent_coef_initial 0.05 \
  --ent_coef_final 0.001 \
  --vf_coef 0.5 \
  --max_grad_norm 0.5 \
  --seed 0 \
  --device cuda