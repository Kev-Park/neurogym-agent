"""Select balanced positive/negative-advantage misses from the first PPO batch."""
import argparse
import json
from pathlib import Path

import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument('directory')
ap.add_argument('--output', required=True)
args = ap.parse_args()
directory = Path(args.directory)
labels = {}
for path in sorted(directory.glob('learner-*.jsonl')):
    for line in path.open():
        packet = json.loads(line)
        # 16001 collected actions at minibatch 512: the first 32 calls cover
        # the first epoch, sampled by the unchanged source checkpoint 1001.
        if packet['call'] > 32:
            break
        for values in packet['rows']:
            row = dict(zip(packet['columns'], values))
            labels.setdefault((int(row['episode']), int(row['step'])), row)
candidates = []
for path in sorted((directory / 'snapshots').glob('*.json')):
    case = json.loads(path.read_text())
    label = labels.get((case['episode'], case['step']))
    if label is None:
        continue
    assert label['verb'] == 3 and label['pane3d'] == 1 and not label['selection_changed']
    case['standardized_advantage'] = label['standardized_advantage']
    candidates.append(case)
rng = np.random.default_rng(7)
selected = []
for positive in [True, False]:
    pool = [c for c in candidates if (c['standardized_advantage'] > 0) == positive]
    assert len(pool) >= 4, f'Only {len(pool)} captured cases for positive={positive}'
    selected.extend(pool[int(i)] for i in rng.choice(len(pool), 4, replace=False))
Path(args.output).write_text(json.dumps(selected, indent=2))
print(json.dumps({'eligible_cases': len(candidates), 'selected': [
    {k: c[k] for k in ['episode', 'step', 'standardized_advantage']} for c in selected]}))
