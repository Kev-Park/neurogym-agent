"""Paired continuations for training misses with linked PPO advantages."""
import argparse
import json
from pathlib import Path

import numpy as np


def paired(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'states': 0}
    rng = np.random.default_rng(7)
    means = a[rng.integers(len(a), size=(4000, len(a)))].mean(1)
    return {'states': len(a), 'mean_difference': float(a.mean()),
            'state_bootstrap_95ci': np.quantile(means, [.025, .975]).tolist()}


ap = argparse.ArgumentParser()
ap.add_argument('--output', default='click_credit_counterfactual_report.json')
args = ap.parse_args()
cases = json.load(open('click_credit_cases.json'))
rows, completed = [], []
for i in range(4):
    payload = json.load(open(f'cf_credit_part{i}.json'))
    rows.extend(payload['records'])
    completed.extend(payload['completed_indices'])
assert len(completed) == len(set(completed)) == len(cases) == 8
assert set(completed) == {c['episode'] for c in cases}
assert all(r.get('steps') == 101 for r in rows), 'Incomplete or skipped continuation'
lookup = {(r['idx'], r['repeat'], r['branch']): r for r in rows}
assert len(lookup) == len(rows)
report = {'states': len(cases), 'records': len(rows), 'cases': [], 'paired_vs_neutral_zoom': {},
          'limitations': [
              'Four positive and four non-positive advantage cases deliberately balance signs; they do not represent overall prevalence.',
              'Fresh rendering preserves physical state and reward history, not asynchronous queues or the original observation.',
              'Settling can turn the original miss into a hit; those cases are separately identified.',
              'Highest-probability valid click alternatives use privileged target knowledge.',
              'Only two continuation seeds and 100 continuation actions per branch; intervals resample eight states, not training seeds.']}
for case in cases:
    idx = case['episode']
    sampled = lookup[idx, 0, 'sampled']
    report['cases'].append({'episode': idx, 'source_step': case['step'],
        'standardized_advantage': case['standardized_advantage'],
        'settled_sampled_is_miss': sampled['settled_sampled_is_miss'],
        'branches': {name: {metric: float(np.mean([r[metric] for r in rows if r['idx'] == idx and r['branch'] == name]))
                           for metric in ['first_reward', 'discounted_return', 'new_peak_dz']}
                     for name in sorted({r['branch'] for r in rows if r['idx'] == idx})}})
for group, keep in {
    'all': lambda c: True,
    'positive_advantage': lambda c: c['standardized_advantage'] > 0,
    'nonpositive_advantage': lambda c: c['standardized_advantage'] <= 0,
    'still_misses_after_settling': lambda c: c['settled_sampled_is_miss'],
}.items():
    report['paired_vs_neutral_zoom'][group] = {}
    for branch in ['sampled', 'noop', 'small_rotation', 'valid_2d', 'valid_3d']:
        selected = [c for c in report['cases'] if keep(c) and branch in c['branches']]
        report['paired_vs_neutral_zoom'][group][branch] = {
            metric: paired([c['branches'][branch][metric] - c['branches']['neutral_zoom'][metric] for c in selected])
            for metric in ['discounted_return', 'new_peak_dz']}
Path(args.output).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
