#!/usr/bin/env bash
# Supervised eval shard: survive playwright greenlet wedges.
#
# eval_d0 runs ONE env in-process, so its SIGALRM cannot fire when the main
# thread is wedged inside playwright (the caller greenlet waits forever in
# select() below the interpreter loop). Only an external SIGKILL clears it —
# which is exactly what training gets for free from Ray actor restarts.
# This wraps a shard in that missing layer: run eval_d0 under `timeout`,
# count what it flushed (eval_d0 flushes EVERY pair for short runs), and
# relaunch for the remainder until the range is done.
#
# Each attempt writes its own offset-addressed file, so merge_eval_shards.py
# stitches them together and de-dupes by pair_idx with no extra logic.
#
#   eval_shard_supervised.sh <pkl> <out-prefix> <offset> <size> [config]
set -u
PKL=$1
PREFIX=$2
OFFSET=$3
SIZE=$4
CONFIG=${5:-configs/ppo_zmax_navigate.yaml}

# Budget per attempt: generous per-pair allowance + browser/DINO startup.
# A wedge burns this once, then the next attempt resumes from the flush.
PER_PAIR_S=${PER_PAIR_S:-90}
# Tier-2 evals run REPEATS rollouts per pair (paired signed-rank/t
# test); budget and resume accounting both scale with it.
REPEATS=${REPEATS:-1}
STARTUP_S=${STARTUP_S:-240}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-8}

cur=$OFFSET
remain=$SIZE
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  [ "$remain" -le 0 ] && break
  out="${PREFIX}_shard_off${cur}.json"
  budget=$((STARTUP_S + PER_PAIR_S * remain * REPEATS))
  echo "[sup] attempt ${attempt}: offset=${cur} remain=${remain} budget=${budget}s"
  timeout -k 30 "$budget" uv run --no-sync python scripts/eval_d0.py \
    --config "$CONFIG" \
    --eval-d0 /scratch/kp0374/neurogym-agent/eval_d0_v1.parquet \
    --skeleton /scratch/kp0374/neurogym-agent/segment_positions.parquet \
    --state-pkl "$PKL" \
    --obs dino --stochastic \
    --repeats "$REPEATS" \
    --offset "$cur" --limit "$remain" \
    --output "$out"
  rc=$?
  # Count only pairs with ALL their repeats flushed: a pair cut
  # off mid-repeat is redone in full, else resume drops rollouts.
  n=$(python3 - "$out" "$REPEATS" <<'PY'
import json, sys
from collections import Counter
try:
    rows = json.load(open(sys.argv[1]))['per_pair']
    c = Counter(r['pair_idx'] for r in rows)
    print(sum(1 for v in c.values() if v >= int(sys.argv[2])))
except Exception:
    print(0)
PY
)
  echo "[sup] attempt ${attempt} rc=${rc} pairs_written=${n}"
  cur=$((cur + n))
  remain=$((remain - n))
  # A wedge (or instant death) writes nothing: pause briefly so a hard
  # failure can't spin the retry loop.
  [ "$n" -eq 0 ] && sleep 10
done

if [ "$remain" -gt 0 ]; then
  echo "[sup] INCOMPLETE: ${remain} pairs still missing from offset ${cur}"
  exit 1
fi
echo "[sup] SHARD-COMPLETE offset=${OFFSET} size=${SIZE}"
