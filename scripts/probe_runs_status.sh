#!/bin/bash
# Status sentinel for one or more training runs (coordinator or single-node).
# Same rationale as probe_run_status.sh: Win10-side monitors call it with ONE
# quoting level through the Windows->WSL->ssh bridge. Unlike that script this
# one is run-agnostic (no coordinator assumption, no learner-001.log pin) and
# reads the authoritative meta.json that train.py publishes each iteration.
# Always prints SENTINEL_OK last, so EMPTY output means connection failure,
# never "no news".
#
#   bash scripts/probe_runs_status.sh native-v9-noleft native-svc-noleft
for RUN in "$@"; do
    python3 - "$RUN" <<'PY'
import json, sys
run = sys.argv[1]
try:
    m = json.load(open("/scratch/kp0374/checkpoints/%s/meta.json" % run))
    print("RUN=%s IT=%s STEPS=%s" % (run, m.get("iteration", -1),
                                     m.get("total_steps", -1)))
except Exception:
    print("RUN=%s IT=- STEPS=-" % run)
PY
done
echo "Q=$(squeue -h -u kp0374 -o '%i:%j:%T' | tr '\n' ' ')"
echo SENTINEL_OK
