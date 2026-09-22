#!/bin/bash
# One arm of the DINO-lever single-GPU A/B (24 runners x 2 envs, MPS). Command
# logic lives here (committed, literal) so the .slurm wrappers pass only a literal
# arm name -- no parameterized-submit quoting through the Win10->wsl->ssh bridge.
#
#   bash scripts/dino_lever_arm.sh <baseline|server|graph> [max_batch]
#
# baseline = in-process DINO per runner (SOTA, batch=M=2).
# server   = 1 shared DinoServer/GPU, dynamic micro-batching (Exp 1, big batch).
# graph    = in-process DINO + CUDA-graph replay of encode_gpu (Exp 2).
# All write slurm_outputs/dino-<arm>-<jobid>.out; per-GPU sps = the `iter N` lines.
set -u
cd /scratch/kp0374/wt/neurogym-agent-throughput-scaling
ARM="$1"
MAXB="${2:-48}"
NRUN="${3:-24}"                         # env-runners (concurrency sweep knob)
MENV="${4:-2}"                          # envs per runner = render+DINO batch size
J="${SLURM_JOB_ID:-manual}"
COMMON="--no-spawn-curriculum --learner-gpu --num-env-runners $NRUN --num-envs-per-env-runner $MENV \
  --num-cpus-per-env-runner 0.5 --vector threads --iters ${ITERS:-8} --train-batch-size 24000 \
  --checkpoint-every 999999 --no-degraded-exit --wandb-mode disabled"

case "$ARM" in
  baseline)
    CFG=configs/native_rb_interop.yaml
    EXTRA="--num-gpus-per-env-runner 0.03" ;;
  graph)
    CFG=configs/native_rb_interop_graph.yaml
    EXTRA="--num-gpus-per-env-runner 0.03" ;;
  noop)
    CFG=configs/native_rb_interop_noop.yaml   # R_cap: render+interop, ViT skipped
    EXTRA="--num-gpus-per-env-runner 0.03" ;;
  server)
    CFG=configs/native_rb_srv_interop.yaml
    EXTRA="--num-gpus-per-env-runner 0 --dino-server-instances 1 --dino-server-max-batch $MAXB --dino-server-max-delay-ms 3" ;;
  *)
    echo "unknown arm: $ARM"; exit 2 ;;
esac

echo "[dino-lever] arm=$ARM cfg=$CFG maxb=$MAXB job=$J node=$(hostname)"
UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp \
  uv run --no-sync python -m ngllib_agent.train \
    --config "$CFG" --run-name "dino-$ARM-$J" $COMMON $EXTRA
echo "[dino-lever] arm=$ARM done exit=$?"
