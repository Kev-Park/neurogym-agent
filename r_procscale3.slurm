#!/usr/bin/env bash
#SBATCH --job-name=procscale3
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4
#SBATCH --gres=gpu:3090:8
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=0:45:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=code@thiskevin.com
#SBATCH --output=slurm_outputs/procscale3-%j.out

# Exclusive good 3090 node so each GPU-process gets abundant CPU + no co-tenant.
# See run_procscale3.sh for the hypothesis being tested.
cd /scratch/kp0374/neurogym-agent
bash scripts/run_procscale3.sh
