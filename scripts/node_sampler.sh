#!/bin/bash
# Per-node 1 Hz resource sampler for storm-precursor analysis.
# Writes: epoch_ts,gpu_mem_used_mib,gpu_util_pct,mem_util_pct,load1
# Usage: node_sampler.sh <output_csv>
OUT="$1"
echo "ts,gpu_mem_mib,gpu_util,mem_util,load1,host" > "$OUT"
HOST=$(hostname)
while true; do
  ts=$(date +%s.%N)
  g=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  l=$(cut -d' ' -f1 /proc/loadavg)
  echo "${ts},${g},${l},${HOST}" >> "$OUT"
  sleep 1
done
