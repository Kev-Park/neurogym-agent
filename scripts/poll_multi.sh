#!/bin/bash
# Lean multi-job SPS poller for Monitor/background use. Args: pairs of
# "<jobid>:<result.out>". Every ~30s prints one TICK line with the live job
# states (sentinel: empty output => ssh/VPN dropped, not "nothing new"); when a
# job leaves the queue prints one DONE line with its last iter sps + exit state;
# exits when all jobs are gone. Lean stdout so it is safe to stream as events.
set -u
PAIRS=("$@")
declare -A OUT
IDS=""
for p in "${PAIRS[@]}"; do
  jid="${p%%:*}"; f="${p#*:}"
  OUT["$jid"]="$f"
  IDS="${IDS:+$IDS,}$jid"
done
declare -A DONE
for c in $(seq 1 60); do
  live=$(squeue -h -j "$IDS" -o '%i:%T' 2>/dev/null | tr '\n' ' ')
  # Report any job that was live before and is now absent.
  for jid in "${!OUT[@]}"; do
    if [ -z "${DONE[$jid]:-}" ] && ! echo " $live " | grep -q " $jid:"; then
      DONE["$jid"]=1
      last=$(grep -E 'iter [0-9]+: ' "${OUT[$jid]}" 2>/dev/null | tail -1)
      state=$(sacct -j "$jid" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
      echo "DONE $jid state=$state | $last"
    fi
  done
  remaining=0
  for jid in "${!OUT[@]}"; do [ -z "${DONE[$jid]:-}" ] && remaining=$((remaining+1)); done
  echo "TICK $c live=[$live] remaining=$remaining"
  [ "$remaining" -eq 0 ] && { echo "ALL-DONE"; break; }
  sleep 30
done
