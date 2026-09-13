#!/bin/bash
#SBATCH --job-name=nodesps
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:4
#SBATCH --cpus-per-task=44
#SBATCH --mem=180G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# RENDERER-ONLY node throughput, by topology shape: K sampling processes x M
# envs each, spread over G GPUs, no learner, no Ray -- the sps a pure renderer
# node contributes. Each process is its own `probe_throughput.py` (a
# ThreadedVectorEnv of M envs with its own DINO). One `srun --exact --gres=gpu:1`
# STEP per GPU (G concurrent steps, each cgroup-pinned to one GPU, because
# Chrome ignores CUDA_VISIBLE_DEVICES -- REFINEMENT.md R4-frontier), with K/G
# processes sharing that GPU inside the step, the way Ray runners share one.
# Random actions, DINO obs: the production per-step cost minus the policy.
#
#   BACKEND=chrome    sbatch native/r_node_sps.slurm   # capture 0.5
#   BACKEND=simulator sbatch native/r_node_sps.slurm
#   SHAPES="2x16x1 4x16x2" overrides the list; each entry is KxMxG (K % G == 0):
#   K runner processes IN TOTAL (K/G per GPU), M envs each, over G GPUs.
#   Give the job --cpus-per-task divisible by 2G so every step gets whole cores.
set -u
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
BACKEND=${BACKEND:-chrome}
SECS=${SECS:-150}
NCPU=${SLURM_CPUS_PER_TASK:-44}
# Host RAM shared across the G GPU steps. The default matches --mem above;
# an hour-long simulator run at >=24 runners/GPU needs a node's worth
# (sbatch --mem=340G MEM_G=330): 48x2x2 and 64x2x2 both reached ~106 GB per
# GPU step within an hour and were cgroup-killed under the 85 GB default.
MEM_G=${MEM_G:-170}
if [ "$BACKEND" = simulator ]; then
  export THRU_CONFIG=configs/native.yaml
  export NGL_NATIVE_FETCH_WORKERS=${FETCH_WORKERS:-2}
  export NGL_NATIVE_MESH_LRU_MB=128
  # Per-process cache budgets (ngllib 0baa9e2). Unset = the library defaults
  # (256 MB per chunk handle, 2 GB of mesh VAOs per process), which are sized
  # for one renderer, not 24-32 per GPU -- see r_train_native.slurm for the
  # per-node sizing rules.
  [ -n "${CHUNK_LRU_MB:-}" ] && export NGL_NATIVE_CHUNK_LRU_MB=$CHUNK_LRU_MB
  [ -n "${VAO_LRU_MB:-}" ] && export NGL_NATIVE_VAO_LRU_MB=$VAO_LRU_MB
  # production = 32 procs x 1 env on 1 GPU (15.3 G VRAM); then denser packings
  # on the one GPU, and a second GPU's worth of runners.
  SHAPES=${SHAPES:-"32x1x1 16x2x1 8x4x1 32x2x1 64x1x2"}
else
  export THRU_CONFIG=configs/ppo_zmax_navigate.yaml
  # production = 2 procs x 16 browsers on 1 GPU. ~0.6 G VRAM per Chrome caps a
  # 3090 near 32-40 browsers, so denser shapes need more GPUs. The July sweep
  # (capture 1.0) found a ~35 sps node ceiling and flat multi-GPU scaling.
  SHAPES=${SHAPES:-"2x16x1 4x8x1 1x32x1 4x16x2 8x8x2 8x16x4"}
fi
echo "nodesps backend=$BACKEND node=$(hostname) gpus=$CUDA_VISIBLE_DEVICES cpus=$NCPU commit=$(git rev-parse --short HEAD) secs=$SECS"
# SAMPLE_VRAM=1: log every GPU's memory/util and the node's summed RSS every
# 10 s from the batch step, which already owns all the job's GPUs. (An extra
# srun --overlap --gres step for this blocked one of the job's own GPU steps
# for 66 min on 2026-09-12 -- --overlap does not share gres with --exact.)
if [ "${SAMPLE_VRAM:-0}" = 1 ]; then
  VRAM_CSV=/scratch/kp0374/native_spike/nodesps_logs/vram_${SLURM_JOB_ID}.csv
  echo "ts,gpu,mem_used_mib,util_pct" > "$VRAM_CSV"
  ( while true; do t=$(date +%s)
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | sed "s/^/$t,/"
      ps -u "$USER" -o rss= | awk -v t="$t" '{s+=$1} END{printf "%s,host,%d,%d\n", t, s/1024, NR}'
      sleep 10; done >> "$VRAM_CSV" 2>/dev/null ) &
  SAMPLER_PID=$!
  trap 'kill $SAMPLER_PID 2>/dev/null' EXIT
fi
echo "shapes: $SHAPES"
LOGDIR=/scratch/kp0374/native_spike/nodesps_logs; mkdir -p "$LOGDIR"

for shape in $SHAPES; do
  K=${shape%%x*}; rest=${shape#*x}; M=${rest%%x*}; G=${rest#*x}
  per=$(( K / G ))
  # Whole CORES per step: an odd thread count rounds UP to a core on these
  # 2-thread/core nodes, so 44/4 = 11 -> 12 per step and the 4th step waited
  # for the first three to FINISH ("Requested nodes are busy", every 4-GPU
  # job on 2026-09-12 -- their aggregates summed 3 concurrent + 1 solo).
  cpus=$(( NCPU / G / 2 * 2 )); mem=$(( MEM_G / G ))G
  logs=""; pids=""
  for g in $(seq 1 "$G"); do
    log=$LOGDIR/${SLURM_JOB_ID}_${shape}_g${g}.out
    # Inside the step: K/G probe processes on this one GPU, staggered so K
    # browsers do not thundering-herd at t=0; wait for all of them.
    srun --exact --ntasks=1 --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem \
      bash -lc "cd $PWD; echo STEP gpu=\$CUDA_VISIBLE_DEVICES cpus=\$(nproc); \
        for i in \$(seq 1 $per); do THRU_SECS=$SECS timeout -k 30 $((SECS + 900)) uv run --no-sync python scripts/probe_throughput.py $M 100000 0.5 & sleep 3; done; wait" \
      > "$log" 2>&1 &
    pids="$pids $!"; logs="$logs $log"
  done
  for p in $pids; do wait "$p"; done
  total=0; ok=0; detail=""
  for l in $logs; do
    for s in $(grep -oE 'RESULT.*sps=[0-9.]+' "$l" | grep -oE 'sps=[0-9.]+' | cut -d= -f2); do
      total=$(awk "BEGIN{print $total+$s}"); ok=$((ok+1)); detail="$detail $s"
    done
  done
  echo "[nodesps] $BACKEND shape=${K}x${M} gpus=$G procs_ok=$ok/$K AGGREGATE=${total} sps per-proc:${detail}"
  grep -h '^STEP' $logs | tr '\n' ';'; echo
  grep -h 'Traceback\|sps=FAILED' $logs | head -3
done
echo NODESPS-DONE
