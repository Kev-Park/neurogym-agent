#!/bin/bash
#SBATCH --job-name=toposweep
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=44
#SBATCH --mem=200G
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4,sarekl15-8,sarekl16-2,sarekl15-5
#SBATCH --time=02:45:00
#SBATCH --output=/scratch/kp0374/native_spike/slurm-%x-%j.out

# In-RLlib service-mode topology sweep: runners x envs per node, incl.
# past-64 client counts. Each round = a FRESH 12-iteration training run
# (fresh curriculum -> reset-heavy easy phase = the conservative regime);
# compare mean t of iters 3-12. cpu reservation ~= 40/K.
set -e
cd /scratch/kp0374/wt/neurogym-agent-native
export PYTHONUNBUFFERED=1
export RAY_NUM_CPUS=44
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_ADDRESS=local

for round in 16:4:2.5 32:2:1.2 64:1:0.6 24:4:1.7 48:2:0.8 32:3:1.2; do
  K=${round%%:*}; rest=${round#*:}; M=${rest%%:*}; C=${rest##*:}
  RUN="swp-${K}x${M}"
  CKPT="/scratch/kp0374/checkpoints/sweep/${RUN}"
  rm -rf "$CKPT"; mkdir -p "$CKPT"
  export CURRICULUM_PROGRESS_FILE="$CKPT/meta.json"
  echo "SWEEP-ROUND K=$K M=$M clients=$((K * M))"
  timeout 1500 uv run --no-sync python -m ngllib_agent.train \
    --config configs/native_service.yaml \
    --run-name "$RUN" --render-service --learner-gpu \
    --num-env-runners "$K" --num-envs-per-env-runner "$M" \
    --num-gpus-per-env-runner 0 --num-cpus-per-env-runner "$C" \
    --vector threads --iters 12 --train-batch-size 4000 \
    --checkpoint-dir "$CKPT" --checkpoint-every 100 \
    --wandb-mode disabled 2>&1 | grep -E "^iter |Traceback" | tail -14 \
    || echo "ROUND-FAILED K=$K M=$M"
  echo "SWEEP-ROUND-DONE K=$K M=$M"
done
rm -rf /scratch/kp0374/checkpoints/sweep
echo "TOPOSWEEP-COMPLETE"
