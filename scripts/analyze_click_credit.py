"""Summarize real PPO advantages and separate direct logit-loss contributions."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(rows):
    episodes = sorted({r['episode'] for r in rows})
    result = {'n': len(rows), 'episodes': len(episodes)}
    if not rows:
        return result
    result['positive_advantage_fraction'] = float(np.mean([r['standardized_advantage'] > 0 for r in rows]))
    episode_index = {e: i for i, e in enumerate(episodes)}
    labels = np.array([episode_index[r['episode']] for r in rows])
    counts = np.bincount(labels, minlength=len(episodes))
    keys = ['standardized_advantage', 'reward', 'ratio', 'p_double', 'p_3d_given_double']
    keys += [term + suffix for term in ['policy', 'entropy', 'kl']
             for suffix in ['_chosen_cell_direction', '_toward_3d_direction', '_double_verb_direction']]
    keys += ['combined_double_verb_direction']
    for key in keys:
        values = np.array([r[key] for r in rows])
        totals = np.column_stack([np.bincount(labels, weights=values, minlength=len(episodes)), counts])
        rng = np.random.default_rng(7)
        resampled = totals[rng.integers(len(episodes), size=(4000, len(episodes)))].sum(1)
        result[key] = {'mean': float(values.mean()),
                       'episode_bootstrap_95ci': np.quantile(resampled[:, 0] / resampled[:, 1], [.025, .975]).tolist()}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    rows, calls = [], 0
    for path in sorted(Path(args.directory).glob('learner-*.jsonl')):
        for line in path.open():
            payload = json.loads(line)
            calls += 1
            for values in payload['rows']:
                row = dict(zip(payload['columns'], values))
                row['combined_double_verb_direction'] = sum(row[term + '_double_verb_direction']
                    for term in ['policy', 'entropy', 'kl'])
                rows.append(row)
    unique = {}
    for row in rows:
        key = (row['episode'], row['step'])
        unique.setdefault(key, row)
    assert rows, 'No learner diagnostic rows'
    for row in rows:
        assert row['verb'] == row['verb_label'], 'Action/outcome labels disagree'
        if row['verb'] == 3:
            assert row['policy_chosen_cell_direction'] * row['standardized_advantage'] >= -1e-7
    groups = {
        'all_actions': lambda r: True,
        '3d_miss': lambda r: r['verb'] == 3 and r['pane3d'] == 1 and not r['selection_changed'],
        '3d_hit': lambda r: r['verb'] == 3 and r['pane3d'] == 1 and r['selection_changed'],
        '2d_miss': lambda r: r['verb'] == 3 and r['pane3d'] == 0 and not r['selection_changed'],
        '2d_hit': lambda r: r['verb'] == 3 and r['pane3d'] == 0 and r['selection_changed'],
    }
    output = {'minibatch_calls': calls, 'row_presentations': len(rows), 'unique_actions': len(unique),
              'limitations': [
                  'Advantages are the actual batch-standardized GAE values, not causal action values.',
                  'First presentation counts each sampled action once; all presentations include PPO reuse.',
                  'Directions are direct output-logit derivatives before the common minibatch mean.',
                  'Shared network, value-loss, Adam and gradient-clipping effects are not captured by these directions.',
                  'Episode bootstrap does not represent training-seed uncertainty.'],
              'first_presentation': {k: summarize([r for r in unique.values() if fn(r)]) for k, fn in groups.items()},
              'all_presentations': {k: summarize([r for r in rows if fn(r)]) for k, fn in groups.items()}}
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
