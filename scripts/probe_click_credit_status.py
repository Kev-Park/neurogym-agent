"""Bounded, sentinel-bearing SLURM status probe for click-credit experiments."""
import argparse
import json
import subprocess
from pathlib import Path


ap = argparse.ArgumentParser()
ap.add_argument('--jobs', nargs='+', required=True)
args = ap.parse_args()
output = subprocess.check_output(['sacct', '-j', ','.join(args.jobs), '-n', '-P',
                                  '-o', 'JobIDRaw,State,ExitCode'], text=True, timeout=20)
states = {}
for line in output.splitlines():
    parts = line.split('|')
    if len(parts) >= 3 and parts[0] in args.jobs:
        states[parts[0]] = dict(state=parts[1], exit_code=parts[2])
live = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'}
terminal = len(states) == len(args.jobs) and all(s['state'].split()[0] not in live for s in states.values())
failed = [j for j, s in states.items() if s['state'].split()[0] not in live and s['state'] != 'COMPLETED']
meta = Path('/scratch/kp0374/checkpoints/zmaxleft-click-credit/meta.json')
result = dict(jobs=states, terminal=terminal, failed=failed,
              iteration=json.loads(meta.read_text()).get('iteration') if meta.exists() else None,
              gradient_report=Path('click_credit_report.json').exists(),
              cf_completed={p.name: len(json.loads(p.read_text()).get('completed_indices', []))
                            for p in Path('.').glob('cf_credit_part*.json')})
print('CREDIT_STATUS_JSON ' + json.dumps(result))
