#!/bin/bash
# One-shot status sentinel for a coordinator run (+ optional eval job log).
# Exists so Win10-side monitors can `ssh pni 'bash .../probe_run_status.sh
# <run> [eval-out]'` with ONE quoting level — triple-nested quoting through
# the Windows->WSL->ssh bridge proved fragile (escapes expanded in the wrong
# shell after a harness restart, 2026-08-27). Always prints SENTINEL_OK on
# success so empty output means connection failure, never "no news".
#
#   bash scripts/probe_run_status.sh coord-v9 [/path/to/eval.out]
RUN=${1:?run name}
EVAL_OUT=${2:-}
ST=$(squeue -h -u kp0374 -n ngllib-agent-coord -o %T | head -1)
CP=$(pgrep -fc 'distributed[.]coordinator')
IT=$(python3 -c "import json;print(json.load(open('/scratch/kp0374/checkpoints/${RUN}/meta.json')).get('iteration',-1))" 2>/dev/null || echo -1)
# Newest learner log — respawns/re-sallocs rotate to learner-00N.log.
LOGF=$(ls -t "/scratch/kp0374/coord-state/logs-${RUN}"/learner-*.log 2>/dev/null | head -1)
L=$(grep -hE '^iter [0-9]+:' "$LOGF" 2>/dev/null | tail -1 | cut -c1-90)
BL=""
if [ -n "$EVAL_OUT" ]; then
    BL=$(grep -hE 'pair [0-9]+/200|success rate|@[0-9]+ |budget' $EVAL_OUT 2>/dev/null | tail -1 | cut -c1-80)
fi
echo "ST=${ST:-GONE} CP=$CP IT=$IT <$L> BL<${BL:-nolog}>"
echo SENTINEL_OK
