#!/bin/bash
#SBATCH --job-name=dino-stagger
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=340G
#SBATCH --time=01:00:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-stagger-%j.out
#
# TASK 2: can staggered runner startup (NGL_RUNNER_STAGGER_SEC) launch MORE runners
# on a many-core node (48 cpus) and scale SPS past ~360? N=36 hung before (36 GL
# contexts init at once); stagger spreads that. Test N=36 then N=48 (M=2, graphs on).
# If they START cleanly AND SPS > 360 -> more processes = more throughput (cores were
# the cap); if they start but plateau ~360 -> contention plateau (as hypothesized).
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0 ITERS=8
export NGL_RUNNER_STAGGER_SEC=3          # runner widx waits widx*3s before env init
export NGL_NATIVE_CHUNK_LRU_MB=200 NGL_NATIVE_VAO_LRU_MB=192
export NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J.lock
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[stagger] mps up on $(hostname) job=$J cores=$(nproc)"

for NR in 36 48; do
  echo "[stagger] === N=$NR (stagger=${NGL_RUNNER_STAGGER_SEC}s) start $(date +%H:%M:%S) ==="
  bash scripts/dino_lever_arm.sh baseline 48 "$NR" 2 > "slurm_outputs/dino-stagger-n$NR-$J.out" 2>&1
  echo "[stagger] === N=$NR done exit=$? ==="
  sleep 15
done

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[stagger] SPS per N:"
grep -hE '^iter ' slurm_outputs/dino-stagger-n36-$J.out slurm_outputs/dino-stagger-n48-$J.out 2>/dev/null | grep -oE 'sps=[0-9.]+' | tail -20
