#!/bin/bash
#SBATCH --job-name=zmax-compat
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/kp0374/neurogym-agent/slurm_outputs/zmax-compat-%j.out
set -euo pipefail
ROOT=${ROOT:-/scratch/kp0374/integration_20261105}
cd "$ROOT/neurogym-agent"
PY=/scratch/kp0374/neurogym-agent/.venv/bin/python
export PYTHONPATH="$ROOT/neurogym/src:$PWD/src"
export PYTHONUNBUFFERED=1 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NGL_NATIVE_FETCH_WORKERS=2 NGL_NATIVE_MESH_WORKERS=2
export NGL_NATIVE_MESH_LRU_MB=256 NGL_NATIVE_CHUNK_LRU_MB=256 NGL_NATIVE_VAO_LRU_MB=256
OUT=/scratch/kp0374/integration_validation_$SLURM_JOB_ID
mkdir -p "$OUT"
echo "VALIDATION node=$(hostname) out=$OUT"
git rev-parse HEAD
git -C ../neurogym rev-parse HEAD
"$PY" -m pytest ../neurogym/tests tests -q > "$OUT/tests.log" 2>&1 || { tail -100 "$OUT/tests.log"; exit 1; }
tail -5 "$OUT/tests.log"
echo TESTS-OK
"$PY" native/probe_smoke_both.py --config configs/native_select.yaml --steps 12 2>&1 | tee "$OUT/backends.log"
common=(--set env.parquet_path=/scratch/kp0374/neurogym-agent/segment_positions.parquet
        --train-batch-size 128 --set ppo.sgd_minibatch_size=32 --set ppo.num_sgd_iter=1
        --num-env-runners 1 --num-envs-per-env-runner 1 --num-cpus-per-env-runner 4
        --num-gpus-per-env-runner 0.5 --learner-gpu --object-store-gb 2 --checkpoint-every 1)
"$PY" -m ngllib_agent.train "${common[@]}" --run-name compat-single --iters 1 --checkpoint-dir "$OUT/single" 2>&1 | tee "$OUT/single.log"
echo TRAIN-SINGLE-OK
# No --resume: an identical task must continue the previous checkpoint.
"$PY" -m ngllib_agent.train "${common[@]}" --run-name compat-single --iters 2 --checkpoint-dir "$OUT/single" 2>&1 | tee "$OUT/resume.log"
echo AUTO-RESUME-OK
"$PY" -m ngllib_agent.train --config configs/native_zmaxleft_v7.yaml "${common[@]}" --run-name compat-multi --iters 1 --checkpoint-dir "$OUT/multi" 2>&1 | tee "$OUT/multi.log"
echo TRAIN-MULTI-OK
"$PY" "$ROOT/neurogym/scripts/probe_label_resolution.py" capture --out "$OUT/labels" --cases /scratch/kp0374/zoom_resolution_probe_20261101/validation/cases.json
"$PY" "$ROOT/neurogym/scripts/probe_label_resolution.py" simulator --out "$OUT/labels" --tag after
echo INTEGRATION-DONE
