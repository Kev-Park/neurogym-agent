#!/bin/bash
#SBATCH --job-name=dino-cpu
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=60
#SBATCH --mem=340G
#SBATCH --time=00:55:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-cpu-%j.out
#
# CPU-cores lever: bottleneck is GIL/process-parallelism (job 966211), and earlier
# "baseline peaks at NRUN~16-24" was a CPU-ALLOCATION cap (jobs had only 20-24 cpus;
# 24 runners saturated them). Request ~all node cores (60/64) + generous RAM, then
# scale runners: M=2, NRUN {24,36,48}. If SPS climbs past ~340 as NRUN grows, the
# cores were the cap and process-parallelism is the lever (no numeric change).
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0 ITERS=6
export NGL_NATIVE_CHUNK_LRU_MB=200 NGL_NATIVE_VAO_LRU_MB=192
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[dino-cpu] mps up on $(hostname) job=$J cores=$(nproc)"

for NR in 24 36 48; do
  echo "[dino-cpu] === M=2 NRUN=$NR start $(date +%H:%M:%S) ==="
  bash scripts/dino_lever_arm.sh baseline 48 "$NR" 2 > "slurm_outputs/dino-cpu-n$NR-$J.out" 2>&1
  echo "[dino-cpu] === M=2 NRUN=$NR done exit=$? ==="
  sleep 12
done

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[dino-cpu] all done; sps per NRUN in slurm_outputs/dino-cpu-n*-$J.out"
