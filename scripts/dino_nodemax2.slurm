#!/bin/bash
#SBATCH --job-name=dino-nodemax2
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:3090:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=01:20:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-nodemax2-%j.out
#
# NODE-MAX confirm: sweep 1 revealed aggregate RISES with GPU count at fixed ~60
# runners (G3=930, G4=960, G6=~1250 noisy) -- spreading runners over more GPUs cuts
# per-GPU contention. Confirm cleanly with MORE iters (stable warm) at G=4/6/8, and
# test G=8 (untested). ITERS=12. Finds the true per-node max + whether it plateaus by 8.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0 ITERS=12
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_CHUNK_LRU_MB=140 NGL_NATIVE_VAO_LRU_MB=140
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[nodemax2] mps up on $(hostname) job=$J cores=$(nproc) gpus=$(nvidia-smi -L | wc -l)"

bash scripts/node_max_arm.sh 4 15     # 60 runners (ref)
bash scripts/node_max_arm.sh 6 10     # 60 runners
bash scripts/node_max_arm.sh 8 8      # 64 runners -- does the 8th GPU still help?

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[nodemax2] all arms done"
