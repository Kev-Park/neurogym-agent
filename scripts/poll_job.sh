#!/usr/bin/env bash
# Poll a SLURM job until it leaves the queue, then print its output tail.
# Prints a sentinel line every cycle so a dropped connection (empty output)
# is distinguishable from "still pending".
#
#   bash scripts/poll_job.sh <jobid> [output-glob]
set -u
JID="$1"
GLOB="${2:-slurm_outputs/procscale-${JID}.out}"
cd /scratch/kp0374/neurogym-agent
for i in $(seq 1 160); do
  st=$(squeue -j "$JID" -h -o '%T' 2>/dev/null)
  if [ -z "$st" ]; then
    echo "POLL $i: job ${JID} left queue"
    f=$(ls -1 ${GLOB} 2>/dev/null | head -1)
    if [ -n "$f" ]; then echo "=== ${f} ==="; cat "$f"; else echo "NO-OUTPUT ${GLOB}"; fi
    exit 0
  fi
  echo "POLL $i: state=${st}"
  sleep 30
done
echo "POLL: gave up, job ${JID} still ${st}"
