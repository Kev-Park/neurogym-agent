#!/usr/bin/env bash
# One-line-per-job status of the gates 6-7 jobs listed in
# /scratch/kp0374/native_spike/gates67_jobs.txt, plus terminal markers from
# their outputs. Always prints a sentinel first so an empty result means the
# connection failed, never "nothing new".
OUT=/scratch/kp0374/native_spike
echo "GATES67 $(date +%H:%M)"
while read -r tag jid; do
  [ -z "$jid" ] && continue
  q=$(squeue -j "$jid" -h -o '%T' 2>/dev/null | sort | uniq -c | tr -s ' \n' ' ')
  if [ -z "$q" ]; then
    q="gone:$(sacct -j "$jid" -n -X -o State | sort | uniq -c | tr -s ' \n' ' ')"
  fi
  case "$tag" in
    g6-*)   f=$(ls -1 $OUT/slurm-gate6-$jid.out 2>/dev/null)
            [ -n "$f" ] && m="iters=$(grep -c '^iter ' "$f") $(grep -o 'GATE6-DONE.*' "$f" | tail -1) tb=$(grep -c Traceback "$f")" || m="no-out" ;;
    g7-sim-*) f=$(ls -1 $OUT/slurm-nativeeval-$jid.out 2>/dev/null)
            [ -n "$f" ] && m="pairs=$(grep -c '^\[eval\] pair' "$f") tb=$(grep -c Traceback "$f")" || m="no-out" ;;
    g7-chrome-*) m="shards-done=$(ls -1 $OUT/slurm-evalbshard-${jid}_*.out 2>/dev/null | xargs -r grep -l 'sup\] done\|grep -l 'SHARD-COMPLETE' 2>/dev/null | wc -l) tb=$(cat $OUT/slurm-evalbshard-${jid}_*.out 2>/dev/null | grep -c Traceback)" ;;
  esac
  echo "$tag $jid [$q] $m"
done < "$OUT/gates67_jobs.txt"
