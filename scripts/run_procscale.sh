#!/usr/bin/env bash
# Grab a GPU allocation on a vulkan-good node and run the per-node
# process/GPU-scaling sweep (probe_procscale.py) inside it, then release.
#
#   bash scripts/run_procscale.sh [node] [ngpu]
#
# Kept as a script (not an inline ssh heredoc) so the salloc/srun variables
# don't have to survive WSL->ssh quote-escaping.
set -u
cd /scratch/kp0374/neurogym-agent

NODE="${1:-sarekl15-4}"
NGPU="${2:-3}"

OUT=$(salloc --no-shell -p preempt -A pni --gres=gpu:3090:"${NGPU}" \
      -c 24 --mem=150G -t 0:45:00 --nodelist="${NODE}" -J procscale 2>&1)
echo "$OUT"
JID=$(echo "$OUT" | grep -oP 'Granted job allocation \K[0-9]+')
echo "JID=${JID}"
if [ -z "${JID}" ]; then echo "ALLOC-FAILED"; exit 1; fi

echo "=== running procscale sweep on ${NODE} (${NGPU} GPU) ==="
srun --jobid="${JID}" --overlap -N1 bash -lc \
  'cd /scratch/kp0374/neurogym-agent && UV_CACHE_DIR=/tmp/uvcache_kp0374 TMPDIR=/tmp uv run --no-sync python scripts/probe_procscale.py'
rc=$?

echo "=== SWEEP-DONE (rc=${rc}), releasing alloc ${JID} ==="
scancel "${JID}"
exit "${rc}"
