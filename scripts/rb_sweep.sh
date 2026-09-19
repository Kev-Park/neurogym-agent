#!/bin/bash
# Submit the render-batching SPS sweep: {baseline, batched-readback, batched-interop}
# x (runners, envs/runner). ENVS = the atlas batch size, so the batched arms should
# pull further ahead of baseline as ENVS grows (more per-env GL contexts collapsed
# into one). All under MPS, real PPO, right-pane-only. Run from the cluster worktree:
#   bash scripts/rb_sweep.sh            # default grid
#   GRID="16:2 16:4 16:8 32:2" bash scripts/rb_sweep.sh
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling

# runners:envs pairs. Default sweeps batch size (envs/runner) at fixed runners,
# plus one higher-runner point. Keep small first — the fleet is co-tenanted.
GRID=${GRID:-"16:2 16:4 16:8"}

declare -A CFG=(
  [base]=configs/native_rb_base.yaml
  [rb]=configs/native_rb_readback.yaml
  [ip]=configs/native_rb_interop.yaml
)
ARMS=${ARMS:-"base rb ip"}

for pair in $GRID; do
  R=${pair%%:*}; E=${pair##*:}
  for arm in $ARMS; do
    cfg=${CFG[$arm]}
    name="rb-${arm}-r${R}e${E}"
    sbatch --job-name="$name" \
      --export=ALL,CONFIG=$cfg,RUNNERS=$R,ENVS=$E \
      scripts/rb_bench.slurm
  done
done
echo "=== queued ==="
squeue -u kp0374 -o '%.10i %.16j %.8T %.10M %R'
