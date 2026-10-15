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
source_files = sorted(directory.glob('learner-*.jsonl'), key=lambda p: p.stat().st_mtime)
assert source_files, 'No learner records'
# A resumed allocation has a new learner PID and a later source policy.
# Counterfactuals use checkpoint 1001, so select only from the original file.
for path in source_files[:1]:
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
represented_ids = {}
for path in sorted((directory / 'snapshots').glob('*.json')):
    case = json.loads(path.read_text())
    # RLlib converts the float64 info vector to float32. Preserve the raw
    # snapshot ID, but join through its actual learner representation and
    # reject any ambiguity from that conversion.
    represented_id = int(np.float32(case['episode']))
    assert represented_id not in represented_ids or represented_ids[represented_id] == case['episode'], 'Ambiguous episode ID after float32 conversion'
    represented_ids[represented_id] = case['episode']
    label = labels.get((represented_id, case['step']))
    if label is None:
        continue
    assert label['verb'] == 3 and label['pane3d'] == 1 and not label['selection_changed']
    case['standardized_advantage'] = label['standardized_advantage']
    case['learner_episode_id'] = represented_id
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
