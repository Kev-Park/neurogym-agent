#!/bin/bash
#SBATCH --job-name=nativeprobe
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# Good vulkan/EGL nodes MINUS coord-v9's allocation (sarekl16-[6-8]) — the
# native-renderer spike must not contend with the running experiment.
#SBATCH --nodelist=sarekl15-1,sarekl15-2,sarekl15-4,sarekl15-7,sarekl16-1,sarekl16-3,sarekl16-5
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# EGL + CloudVolume feasibility probe (spike venv, native-renderer worktree).
set -e
WT=/scratch/kp0374/wt/neurogym-agent-native
VENV=/scratch/kp0374/native_spike/.venv
export PYTHONUNBUFFERED=1
"$VENV/bin/python" "$WT/native/probe_native_stack.py"
