#!/bin/bash
# Launch one training per visible GPU (CUDA_VISIBLE_DEVICES=i), each reading the
# NGL_* fetch env already exported by the caller (process fetch for the baseline,
# socket fetch to a CPU server for disagg). Waits for all; per-GPU logs. Shared by
# disagg_mgpu_baseline.slurm and disagg_mgpu_socket.slurm so the ONLY difference
# between the arms is the fetch backend. bash-native VAR=val (never `env` -- broken
# ~/.local/bin/env shim).
set -u
J=${SLURM_JOB_ID}
TAG=${MGPU_TAG:-mgpu}
N=$(nvidia-smi -L | wc -l)
echo "[mgpu] node=$(hostname) N=$N tag=$TAG backend=${NGL_NATIVE_FETCH_BACKEND:-process} NRUN=${NRUN:-24} M=${M:-2}"
pids=""
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$i UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
    uv run --no-sync python -m ngllib_agent.train \
      --config configs/native_rb_interop.yaml --run-name "$TAG-$J-g$i" \
      --no-spawn-curriculum --learner-gpu \
      --num-env-runners "${NRUN:-24}" --num-envs-per-env-runner "${M:-2}" \
      --num-gpus-per-env-runner "${GPUFRAC:-0.03}" --num-cpus-per-env-runner 0.5 \
      --vector threads --iters "${ITERS:-12}" --train-batch-size 24000 \
      --checkpoint-every 999999 --no-degraded-exit --wandb-mode disabled \
      > "slurm_outputs/$TAG-$J-g$i.out" 2>&1 &
  pids="$pids $!"
  echo "[mgpu] launched GPU $i pid $!"
done
wait $pids
echo "[mgpu] all $N trainings done"
