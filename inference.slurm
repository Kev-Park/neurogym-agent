#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # one per environment
#SBATCH --mem=80G                # ~5G per Chrome env × 12 + DINO model + headroom
#SBATCH --gres=gpu:3090:1
#SBATCH --time=1:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=kp0374@princeton.edu
##SBATCH --nodelist=sarekl16-8

uv run src/neurogym-agent/rollout.py ./checkpoints/0hwtbu19 \
  --rollouts 5 

## --deterministic option