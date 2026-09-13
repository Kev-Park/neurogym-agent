#!/usr/bin/env bash
# One line per training job: queue state, iterations logged, last sps, and
# the terminal markers. Sentinel first, so empty output means the connection
# failed. Usage: bash scripts/probe_train_jobs.sh <jobid> [<jobid> ...]
OUT=/scratch/kp0374/native_spike
echo "TRAINJOBS $(date +%H:%M)"
for jid in "$@"; do
  q=$(squeue -j "$jid" -h -o '%T' 2>/dev/null | tr -d '\n')
  [ -z "$q" ] && q="gone:$(sacct -j "$jid" -n -X -o State | tr -d ' \n')"
  f=$OUT/slurm-gate6-$jid.out
  if [ -f "$f" ]; then
    n=$(grep -c '^iter ' "$f" || true)
    last=$(grep '^iter ' "$f" | tail -1 | grep -o 'sps=[0-9.]*' || true)
    done_=$(grep -o 'GATE6-DONE.*' "$f" | tail -1 || true)
    tb=$(grep -c Traceback "$f" || true)
    deg=$(grep -c degraded_exit "$f" || true)
    echo "$jid [$q] iters=$n $last $done_ tb=$tb degraded=$deg"
  else
    echo "$jid [$q] no-out"
  fi
done
