#!/usr/bin/env bash
# Gates 6 + 7 of the renderer-seam refactor (renderer_seam_plan.md section 9),
# pre-seam arm vs refactor arm, both backends. Run on the login node:
#
#   bash native/launch_gates67.sh [ckpt.pkl]
#
# Pre arm  = wt/neurogym-agent-native  (native-renderer, the commit the seam forked from)
# Post arm = wt/neurogym-agent-refactor (simulator-refactor)
#
# Gate 7: native-v9-pace740 ckpt_000740 evaluated k=5 on the 200-pair holdout,
#         simulator (one job per arm) and Chrome (20 x 10-pair shards per arm,
#         because Chrome evals wedge; merge with scripts/merge_eval_shards.py),
#         then scripts/eval_compare.py pre vs post per backend.
# Gate 6: 50 PPO iterations per backend per arm with pace740's exact recipe
#         (native/r_train_gate6.slurm); compare sps + return curves.
set -u
PKL=${1:-/scratch/kp0374/checkpoints/native-v9-pace740/ckpt_000740.pkl}
OUT=/scratch/kp0374/native_spike
PRE=/scratch/kp0374/wt/neurogym-agent-native
POST=/scratch/kp0374/wt/neurogym-agent-refactor
GATE6=$POST/native/r_train_gate6.slurm
LOG=$OUT/gates67_jobs.txt
: > "$LOG"
sub() { local tag=$1; shift; local jid; jid=$(sbatch --parsable "$@") && echo "$tag $jid" | tee -a "$LOG"; }

for arm in pre post; do
  if [ "$arm" = pre ]; then cd "$PRE"; else cd "$POST"; fi
  echo "== $arm: $(pwd) @ $(git rev-parse --short HEAD)"
  # gate 7, simulator, k=5 in one job
  sub "g7-sim-$arm" native/r_eval_native.slurm "$PKL" --repeats 5 \
      --output "$OUT/gate7_sim_${arm}.json"
  # gate 7, Chrome, k=5 sharded 20 x 10 pairs
  sub "g7-chrome-$arm" --time=05:00:00 --array=0-19 \
      --export=ALL,SHARD_SIZE=10,REPEATS=5 \
      native/r_eval_browser_shards.slurm "$PKL" "$OUT/gate7_chrome_${arm}"
  # gate 6, both backends
  BACKEND=simulator sub "g6-sim-$arm" "$GATE6" "gate6-sim-$arm" 50
  BACKEND=chrome sub "g6-chrome-$arm" "$GATE6" "gate6-chrome-$arm" 50
done
echo "submitted; ids in $LOG"
squeue -u kp0374 -h -o '%i %j %T %M %R' | head -60
