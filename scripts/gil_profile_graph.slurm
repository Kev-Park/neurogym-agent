#!/bin/bash
#SBATCH --job-name=gil-prof-graph
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/gil-prof-graph-%j.out
#
# TASK 1: re-profile the GIL WITH CUDA graphs on (native_rb_interop.yaml now has
# cuda_graph:true) at the SOTA point M=2/N=24. Did graphs push DINO's forward
# dispatch OFF the GIL, and what's the new top holder? Compare gil-flamegil-graphmany
# to the eager gil-flamegil-stress-966211 (DINO dispatch was ~78% there).
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0
export NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J.lock
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[gil-prof-graph] mps up on $(hostname) job=$J"

bash scripts/gil_profile.sh 2 24 graphmany

echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[gil-prof-graph] === GIL flamegraph top frames (WITH graphs) ==="
cd slurm_outputs
grep -oE '<title>[^<]*</title>' gil-flamegil-graphmany-$J.svg 2>/dev/null | sed 's/<[^>]*>//g' | grep -iE 'samples' | sort -t'(' -k2 -rn | head -22
