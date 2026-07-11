#!/usr/bin/env bash
#SBATCH --job-name=procscale2
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
# No node pin; exclude the 3 Mesa-only nodes so backfill uses any good 3090 node.
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4
#SBATCH --gres=gpu:3090:3
#SBATCH --mem=150G
#SBATCH --time=0:25:00
#SBATCH --output=slurm_outputs/procscale2-%j.out

# Corrected GPU-scaling sweep: 3 tasks so each srun --gres=gpu:1 step gets its
# own cgroup-isolated GPU (see run_procscale2.sh for the why).
cd /scratch/kp0374/neurogym-agent
bash scripts/run_procscale2.sh
