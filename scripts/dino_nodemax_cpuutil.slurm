#!/bin/bash
#SBATCH --job-name=dino-cpuutil
#SBATCH --partition=highpri
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:3090:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=360G
#SBATCH --time=00:30:00
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --output=slurm_outputs/dino-cpuutil-%j.out
#
# Close the loop: confirm the full-node 64-core CPU utilization at the node max.
# Launch the node-max config (G=8 trainings x N=8 runners, graphs, unpinned), warm to
# steady sampling, then mpstat -P ALL all 64 cores. Theory predicts ~all 64 near 100%
# busy doing REAL %nice compute (not %sys spin). Summary computed here (native awk) so
# the read is a plain tail.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
J=$SLURM_JOB_ID
export PYTHONUNBUFFERED=1 RAY_memory_monitor_refresh_ms=0
export NGL_RUNNER_STAGGER_SEC=2
export NGL_NATIVE_CHUNK_LRU_MB=140 NGL_NATIVE_VAO_LRU_MB=140
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$J CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-$J
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" slurm_outputs
nvidia-cuda-mps-control -d && echo "[cpuutil] mps up on $(hostname) job=$J cores=$(nproc) gpus=$(nvidia-smi -L | wc -l)"

pids=""
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
    NGL_DINO_CAPTURE_LOCK=/tmp/ngl_dino_cudagraph-$J-$i.lock \
    uv run --no-sync python -m ngllib_agent.train \
      --config configs/native_rb_interop.yaml --run-name "cpuutil-gpu$i-$J" \
      --no-spawn-curriculum --learner-gpu --num-env-runners 8 --num-envs-per-env-runner 2 \
      --num-gpus-per-env-runner 0.03 --num-cpus-per-env-runner 0.5 --vector threads \
      --iters 12 --train-batch-size 24000 --checkpoint-every 999999 --no-degraded-exit \
      --wandb-mode disabled > "slurm_outputs/cpuutil-gpu$i-$J.out" 2>&1 &
  pids="$pids $!"
done
echo "[cpuutil] 8 trainings launched; warming 240s to steady state"
sleep 240

echo "[cpuutil] === mpstat -P ALL (all cores, 45s) ==="
mpstat -P ALL 3 15 > "slurm_outputs/cpuutil-mpstat-$J.txt" 2>&1
# Summary from the 'Average' block (mpstat cols: CPU %usr %nice %sys ... %idle).
grep -E '^Average:  *[0-9]' "slurm_outputs/cpuutil-mpstat-$J.txt" | awk '
  { busy=100-$NF; sumb+=busy; usr+=$3; nice+=$4; sys+=$5; n++; if(busy>90)o90++; if(busy>75)o75++ }
  END { printf "[cpuutil] CORES=%d  mean_busy=%.1f%%  mean_usr=%.1f  mean_nice=%.1f  mean_sys=%.1f  cores>90%%=%d  cores>75%%=%d\n", n, sumb/n, usr/n, nice/n, sys/n, o90, o75 }'
echo "[cpuutil] per-core busy% (100-idle), sorted:"
grep -E '^Average:  *[0-9]' "slurm_outputs/cpuutil-mpstat-$J.txt" | awk '{printf "%.0f ", 100-$NF}' | tr ' ' '\n' | sort -n | tr '\n' ' '; echo

echo "[cpuutil] per-GPU sps (context):"
for i in 0 1 2 3 4 5 6 7; do
  printf "gpu%s: " "$i"; grep -hE '^iter ' "slurm_outputs/cpuutil-gpu$i-$J.out" 2>/dev/null | grep -oE 'sps=[0-9.]+' | tail -1
done

kill $pids 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
echo quit | nvidia-cuda-mps-control 2>&1 | tail -1 || true
echo "[cpuutil] done"
