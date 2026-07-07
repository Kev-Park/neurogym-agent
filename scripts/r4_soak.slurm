#!/bin/bash
#SBATCH --job-name=r4-soak
#SBATCH --partition=preempt
#SBATCH --account=pni
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:3090:1
#SBATCH --nodelist=sarekl15-2,sarekl15-4,sarekl15-5,sarekl15-7,sarekl15-8,sarekl16-1,sarekl16-3,sarekl16-5,sarekl16-6,sarekl16-7,sarekl16-8
#SBATCH --exclude=sarekl15-3,sarekl15-6,sarekl16-4
#SBATCH --time=03:30:00
#SBATCH --output=slurm_outputs/r4-soak-%j.out

# REFINEMENT R4: threaded long-run soak.
#   - SOAK_M envs (default 16) in ONE process via ThreadedVectorEnv.
#   - --browser-restart-every 10 episodes: forces frequent FULL browser
#     restarts under threads (restart mechanisms under load), on top of the
#     per-episode context recycles.
#   - Verdict vs legacy precedent: sustained >= ~30 sps/node (legacy mode
#     29-30 @ 32 procs; peak 32-36). Summary compares first-10 vs last-10
#     iters for flatness and prints PASS/FAIL lines.
set -e
cd /scratch/kp0374/neurogym-agent
export PYTHONUNBUFFERED=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

M=${SOAK_M:-16}
ITERS=${SOAK_ITERS:-150}
echo "=== r4-soak: M=$M iters=$ITERS node=$(hostname) ==="

uv run --no-sync python scripts/ppo_smoke.py \
    --obs dino --num-env-runners 1 --num-gpus-per-env-runner 0.6 \
    --num-envs-per-env-runner "$M" --vector threads \
    --iters "$ITERS" --train-batch-size $((128 * M)) \
    --rollout-fragment-length 128 \
    --browser-restart-every 10 \
  2>&1 | grep -E "iter [0-9]+:|glitch|Traceback|browser restart|complete" \
  | tee /tmp/soak_iters.txt | grep -vE "glitch" | tail -40

echo "########## SOAK SUMMARY ##########"
grep -E "iter [0-9]+:" /tmp/soak_iters.txt | \
awk -v target=30 '
  {for(i=1;i<=NF;i++) if($i ~ /^steps_per_s=/){split($i,a,"="); sps[++n]=a[2]}}
  END{
    if (n < 20) {print "INSUFFICIENT ITERS: " n; exit 1}
    f=0; for(i=1;i<=10;i++) f+=sps[i]; f/=10
    l=0; for(i=n-9;i<=n;i++) l+=sps[i]; l/=10
    all=0; for(i=1;i<=n;i++) all+=sps[i]; all/=n
    printf "iters=%d  mean_sps=%.1f  first10=%.1f  last10=%.1f  drift=%.0f%%\n", n, all, f, l, (l-f)/f*100
    print (l >= f*0.85) ? "FLATNESS: PASS (last10 >= 85% of first10)" : "FLATNESS: FAIL"
    print (all >= target) ? "THROUGHPUT: PASS (mean >= " target ")" : "THROUGHPUT: FAIL (mean < " target ")"
  }'
echo "glitch_total=$(grep -c glitch /tmp/soak_iters.txt || true)"
echo "restart_total=$(grep -c "browser restart" /tmp/soak_iters.txt || true)"
echo "=== r4-soak done ==="
