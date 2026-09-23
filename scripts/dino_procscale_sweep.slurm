#!/bin/bash
#SBATCH --job-name=dino-procscale
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=340G
#SBATCH --time=00:55:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-procscale-%j.out
#
# Q1 gap: with graphs (GIL off DINO) + more cores, does N slightly above 24 scale
# past 360 before the GL-init wall? Test M=2 graphs at N=24 (ref), 28, 32 on 48 cores
# with startup stagger. Finds both (a) whether GIL-removal+cores buys a few more procs
# and (b) exactly where the ~30ish context wall starts (24 works, 36 hung).
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0 ITERS=8
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_CHUNK_LRU_MB=200 NGL_NATIVE_VAO_LRU_MB=192
export NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J.lock
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[procscale] mps up on $(hostname) job=$J cores=$(nproc)"

for NR in 24 28 32; do
  echo "[procscale] === N=$NR M=2 graphs start $(date +%H:%M:%S) ==="
  bash scripts/dino_lever_arm.sh baseline 48 "$NR" 2 > "slurm_outputs/dino-procscale-n$NR-$J.out" 2>&1
  echo "[procscale] === N=$NR done exit=$? ==="
  echo "[procscale] N=$NR sps:"; grep -hE '^iter ' "slurm_outputs/dino-procscale-n$NR-$J.out" 2>/dev/null | grep -oE 'sps=[0-9.]+|H=[0-9]+/[0-9]+' | tail -8
  sleep 15
done

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[procscale] done"
