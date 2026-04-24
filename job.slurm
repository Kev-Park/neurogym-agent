#!/bin/bash
#SBATCH --job-name=ngl-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # one per environment
#SBATCH --mem=80G                # ~5G per Chrome env × 12 + DINO model + headroom
#SBATCH --gres=gpu:3090:1
#SBATCH --time=48:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=kp0374@princeton.edu
##SBATCH --nodelist=sarekl16-8

xvfb-run -a --server-args="-screen 0 1920x1080x24" \
  uv run src/neurogym-agent/train.py \
  --segment_positions segment_positions.csv \
  --n_envs 12 \
  --total_timesteps 1000000 \
  --wandb_mode online \
  --run_name v3
