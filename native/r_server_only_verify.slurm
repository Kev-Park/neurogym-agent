#!/bin/bash
#SBATCH --job-name=srvonly
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out
#
# Does removing the page.route viewer transport break anything?  The loopback
# server was already the default, so this is a regression check across the
# paths that touch a viewer or a renderer at all:
#
#   1. gate-3 smoke, BOTH backends: observation/action spaces identical, the
#      shared zoom rule, and every click verb -- which exercises Chrome
#      navigation and the simulator's mesh loading and 2D fetch.
#   2. pixel parity, packaged viewer vs hosted vs simulator (PNG): proves the
#      served bundle renders what it used to, meshes included.
#   3. a short TRAINING run per backend under RLlib, where the viewer is
#      served inside env-runner processes rather than a probe.
#
#   REPO=/scratch/kp0374/neurogym-agent WT=/scratch/kp0374/wt/neurogym-agent-login \
#   NGLLIB=/scratch/kp0374/wt/neurogym-login/src sbatch native/r_server_only_verify.slurm
set -e
set -o pipefail
REPO=${REPO:-/scratch/kp0374/neurogym-agent}
WT=${WT:-/scratch/kp0374/wt/neurogym-agent-login}
cd "$REPO"
export PYTHONUNBUFFERED=1
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-30}
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
[ -n "${NGLLIB:-}" ] && export PYTHONPATH="$NGLLIB:$WT/src:${PYTHONPATH:-}"
export NGL_NATIVE_FETCH_WORKERS=${FETCH_WORKERS:-2}
export NGL_NATIVE_MESH_LRU_MB=256
export NGL_NATIVE_CHUNK_LRU_MB=256
export NGL_NATIVE_VAO_LRU_MB=256
OUT=/scratch/kp0374/server_only
mkdir -p "$OUT"
echo "srvonly: node=$(hostname) repo=$REPO wt=$WT ngllib=${NGLLIB:-installed}"

echo; echo "=== 1. gate-3 smoke, both backends ==="
uv run --no-sync python "$WT/native/probe_smoke_both.py" \
  --config "$WT/configs/native_select.yaml" --steps 12 2>&1 | tee "$OUT/smoke.log"

echo; echo "=== 2. pixel parity: packaged viewer vs hosted vs simulator ==="
uv run --no-sync python "$WT/native/probe_fork_sim_match.py" \
  --out-dir "$OUT/match" --appspot 2>&1 | tee "$OUT/match.log"

for arm in "chrome:$WT/configs/ppo_zmax_navigate.yaml:1:8" \
           "simulator:$WT/configs/native.yaml:8:1"; do
  name=${arm%%:*}; rest=${arm#*:}
  cfg=${rest%%:*}; rest=${rest#*:}
  runners=${rest%%:*}; envs=${rest#*:}
  run=srvonly-$name
  CKPT=/scratch/kp0374/checkpoints/$run
  mkdir -p "$CKPT"
  export CURRICULUM_PROGRESS_FILE="$CKPT/meta.json"
  echo; echo "=== 3. training smoke, $name ($runners runners x $envs envs) ==="
  uv run --no-sync python -m ngllib_agent.train \
    --config "$cfg" \
    --run-name "$run" \
    --num-env-runners "$runners" \
    --num-envs-per-env-runner "$envs" \
    --num-gpus-per-env-runner 0.1 \
    --vector threads \
    --iters 3 \
    --train-batch-size 1200 \
    --checkpoint-dir "$CKPT" --checkpoint-every 100 \
    --wandb-mode disabled 2>&1 | tee "$OUT/train_$name.log"
  echo "TRAIN-OK $name"
done
echo "SRVONLY-DONE"
