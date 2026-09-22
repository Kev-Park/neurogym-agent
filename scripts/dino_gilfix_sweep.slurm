#!/bin/bash
#SBATCH --job-name=dino-gilfix
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=01:05:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-gilfix-%j.out
#
# Does freeing the GIL from DINO's forward dispatch (CUDA graphs, now reliable via
# flock) REOPEN the batch lever? Without graphs the batch sweep FELL with M
# (M=8=248, M=16=165) because big M = fewer GILs saturated by DINO dispatch. With
# graphs the dispatch no longer holds the GIL, so bigger batches (fewer procs) may
# now scale. Test graph @ M=8/N=6 and M=16/N=3 vs those no-graph numbers. Plus
# torch.compile @ M=2/N=24 (user's ask; adopt as SOTA if it beats ~360 and parity
# holds -- note compile numerics DRIFT, dynamics-gate before science use).
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0 ITERS=8
export NGL_NATIVE_CHUNK_LRU_MB=300 NGL_NATIVE_VAO_LRU_MB=256
export NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J.lock
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[gilfix] mps up on $(hostname) job=$J"

run() {  # $1=arm $2=N $3=M label=arm-mM-nN
  echo "[gilfix] === $1 M=$3 N=$2 start $(date +%H:%M:%S) ==="
  bash scripts/dino_lever_arm.sh "$1" 48 "$2" "$3" > "slurm_outputs/dino-gilfix-$1-m$3-n$2-$J.out" 2>&1
  echo "[gilfix] === $1 M=$3 N=$2 done exit=$? ==="
  sleep 12
}
run graph 6 8          # graph + big batch: does GIL-free reopen M=8? (no-graph was 248)
run graph 3 16         # graph + bigger batch: M=16 (no-graph was 165)
run compile 24 2       # torch.compile @ M=2/N=24 (vs graph 360 / baseline 340)

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[gilfix] done; per-arm files slurm_outputs/dino-gilfix-*-$J.out"
