#!/usr/bin/env bash
# Gates 6-7 wrap-up after launch_gates67.sh: merge the Chrome shards, compare
# pre vs post per backend (eval_compare), and compare training throughput
# (gate6_compare). Run on the login node from the refactor checkout:
#
#   bash native/finish_gates67.sh
set -u
cd "$(dirname "$0")/.."
O=/scratch/kp0374/native_spike
J=$O/gates67_jobs.txt
jid() { awk -v t="$1" '$1==t{print $2}' "$J" | tail -1; }

for arm in pre post; do
  echo "== merge Chrome shards, $arm"
  uv run --no-sync python scripts/merge_eval_shards.py --prefix "$O/gate7_chrome_$arm" \
    --expect 200 --repeats 5 --output "$O/gate7_chrome_${arm}_merged.json" 2>&1 \
    | grep -i 'overall\|missing\|n_pairs\|wrote' | head -5
done

echo; echo "== GATE 7, simulator: pre vs post (k=5)"
uv run --no-sync python scripts/eval_compare.py "$O/gate7_sim_pre.json" "$O/gate7_sim_post.json" --labels pre post 2>&1 | tail -22
echo; echo "== GATE 7, Chrome: pre vs post (k=5)"
uv run --no-sync python scripts/eval_compare.py "$O/gate7_chrome_pre_merged.json" "$O/gate7_chrome_post_merged.json" --labels pre post 2>&1 | tail -22

echo; echo "== GATE 6, simulator sps (same 22-iter window) + pace740 production log"
uv run --no-sync python scripts/gate6_compare.py --arm "pre=$O/slurm-gate6-$(jid g6-sim-pre).out" \
  --arm "post=$O/slurm-gate6-$(jid g6-sim-post).out" \
  --arm "pace740=$O/slurm-nativetrain-873607.out" --warmup 5 --first 22
echo "-- post, all iterations, vs pace740 full"
uv run --no-sync python scripts/gate6_compare.py --arm "pace740=$O/slurm-nativetrain-873607.out" \
  --arm "post=$O/slurm-gate6-$(jid g6-sim-post).out" --warmup 5

echo; echo "== GATE 6, Chrome sps"
PRE=$(jid g6-chrome-pre); POST=$(jid g6-chrome-post2)
uv run --no-sync python scripts/gate6_compare.py --arm "pre=$O/slurm-gate6-$PRE.out" \
  --arm "post=$O/slurm-gate6-$POST.out" --warmup 3 --first 16
uv run --no-sync python scripts/gate6_compare.py --arm "post=$O/slurm-gate6-$POST.out" --warmup 3
for j in $PRE $POST; do
  echo "-- $j on $(sacct -j "$j" -n -X -o NodeList | tr -d ' ') glitches:"
  grep -o 'resilient: env glitch [A-Za-z]*\|wedged past 300s\|not be recreated\|watchdog killed [a-zA-Z ]*\|degraded_exit' \
    "$O/slurm-gate6-$j.out" | sort | uniq -c
done
