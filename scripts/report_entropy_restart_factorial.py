"""Matched-start contrasts for verb entropy and the global resume-schedule bug."""
import json
from pathlib import Path

import numpy as np


def bootstrap(values):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(7)
    means = a[rng.integers(len(a), size=(4000, len(a)))].mean(1)
    return {'episodes': len(a), 'mean_difference': float(a.mean()),
            'episode_bootstrap_95ci': np.quantile(means, [.025, .975]).tolist()}


paths = [Path('verb_entropy_report.json'), Path('verb_entropy_fixed_report.json')]
if not all(p.exists() for p in paths):
    print('FACTORIAL_WAITING_FOR_OTHER_PAIR')
else:
    legacy, fixed = [json.loads(p.read_text()) for p in paths]
    cells = {'spike_on': legacy['arms']['on'], 'spike_off': legacy['arms']['off'],
             'fixed_on': fixed['arms']['on'], 'fixed_off': fixed['arms']['off']}
    episodes = {key: cell['per_episode'] for key, cell in cells.items()}
    indices = set(episodes['spike_on'])
    assert len(indices) == 40 and all(set(e) == indices for e in episodes.values())
    report = {'cells': {k: {name: cell[name] for name in ['3d_miss_action_fraction', 'double_clicks',
        'misses', 'misses_3d', 'hits_2d', 'peak_dz']} for k, cell in cells.items()}, 'contrasts': {},
        'limitations': [
            'One training continuation per cell; evaluation starts and seeds are matched, but training-seed uncertainty is not measured.',
            'Ten updates per cell; inherited policy history includes earlier resume spikes.',
            'Legacy arms intentionally reproduce .2 global entropy on the first update of each allocation, then .01.',
            'The number of legacy spikes depends on actual allocation restarts; raw learner traces retain each coefficient.',
            'Double-grid entropy is zero in all cells; right-click, rotation and zoom entropy remain active.']}
    contrasts = {
        'restart_spike_effect_verb_on': {'spike_on': 1, 'fixed_on': -1},
        'restart_spike_effect_verb_off': {'spike_off': 1, 'fixed_off': -1},
        'verb_entropy_effect_fixed_schedule': {'fixed_on': 1, 'fixed_off': -1},
        'verb_entropy_effect_legacy_schedule': {'spike_on': 1, 'spike_off': -1},
        'interaction': {'spike_on': 1, 'spike_off': -1, 'fixed_on': -1, 'fixed_off': 1},
    }
    for name, weights in contrasts.items():
        report['contrasts'][name] = {metric: bootstrap([
            sum(w * episodes[cell][idx][metric] for cell, w in weights.items()) for idx in sorted(indices)])
            for metric in ['misses_3d', 'double_clicks', 'hits_2d', 'peak_dz']}
    Path('entropy_restart_factorial_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
