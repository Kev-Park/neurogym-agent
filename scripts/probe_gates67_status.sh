#!/usr/bin/env bash
# One-line-per-job status of the gates 6-7 jobs listed in
# /scratch/kp0374/native_spike/gates67_jobs.txt, plus terminal markers from
# their outputs. Always prints a sentinel first so an empty result means the
# connection failed, never "nothing new".
OUT=/scratch/kp0374/native_spike
echo "GATES67 $(date +%H:%M)"
count() { grep -c "$1" "$2" 2>/dev/null || true; }
while read -r tag jid; do
  [ -z "$jid" ] && continue
  q=$(squeue -j "$jid" -h -o '%T' 2>/dev/null | sort | uniq -c | tr -s ' \n' ' ')
  if [ -z "$q" ]; then
    q="gone:$(sacct -j "$jid" -n -X -o State | sort | uniq -c | tr -s ' \n' ' ')"
  fi
  m="no-out"
  case "$tag" in
    g6-*)
      f=$OUT/slurm-gate6-$jid.out
      if [ -f "$f" ]; then
        m="iters=$(count '^iter ' "$f") $(grep -o 'GATE6-DONE.*' "$f" | tail -1) tb=$(count Traceback "$f")"
      fi ;;
    g7-sim-*)
      f=$OUT/slurm-nativeeval-$jid.out
      if [ -f "$f" ]; then
        m="pairs=$(count '^\[eval\] pair' "$f") tb=$(count Traceback "$f")"
      fi ;;
    g7-chrome-*)
      done_n=$(ls -1 $OUT/slurm-evalbshard-${jid}_*.out 2>/dev/null | xargs -r grep -l 'SHARD-COMPLETE' 2>/dev/null | wc -l)
      tb=$(cat $OUT/slurm-evalbshard-${jid}_*.out 2>/dev/null | grep -c Traceback || true)
      m="shards-done=$done_n tb=$tb" ;;
  esac
  echo "$tag $jid [$q] $m"
done < "$OUT/gates67_jobs.txt"
