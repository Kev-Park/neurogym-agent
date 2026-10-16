"""Episode-clustered credit sensitivity to rollout cuts and return estimators."""
import argparse
import json
from pathlib import Path

import numpy as np

from ngllib_agent.boundary_credit import chunk_advantages


def interval(values, episodes):
    if not len(values):
        return None
    _, labels = np.unique(episodes, return_inverse=True)
    sums = np.bincount(labels, weights=values)
    counts = np.bincount(labels)
    rng = np.random.default_rng(7)
    picks = rng.integers(len(sums), size=(4000, len(sums)))
    means = sums[picks].sum(1) / counts[picks].sum(1)
    return {'mean': float(np.mean(values)), 'episode_bootstrap_95ci': np.quantile(means, [.025, .975]).tolist()}


def arm_report(root):
    packets = []
    max_error = 0.
    for path in sorted(Path(root).glob('boundary-*.jsonl')):
        for line in path.open():
            p = json.loads(line)
            d = np.asarray(p['rows'], dtype=float)
            names = {n: i for i, n in enumerate(p['columns'])}
            get = lambda name: d[:, names[name]]
            valid = get('valid').astype(bool)
            raw = get('value_target') - get('value')
            # Centering on valid transitions isolates estimator changes from
            # the separate effect of including masked bootstrap slots.
            center = lambda a: (a - a[valid].mean()) / max(1e-4, a[valid].std())
            raw_valid_norm = center(raw)
            lambda1, finite = np.zeros(len(d)), np.zeros(len(d))
            start = 0
            for end in np.flatnonzero(~valid):
                r, v, terminal = get('reward')[start:end], get('value')[start:end + 1], get('terminated')[start:end]
                independent = chunk_advantages(r, v, terminal, p['gamma'], p['lambda'])
                max_error = max(max_error, float(np.max(np.abs(independent - raw[start:end]))))
                lambda1[start:end] = chunk_advantages(r, v, terminal, p['gamma'], 1.)
                zero_tail = v.copy(); zero_tail[-1] = 0.
                finite[start:end] = chunk_advantages(r, zero_tail, terminal, p['gamma'], 1.)
                start = end + 1
            assert start == len(d)
            # Validate the actual stored normalization, including its masked
            # slots, against the exact raw advantages from the installed GAE.
            installed_norm = (raw - raw.mean()) / max(1e-4, raw.std())
            assert np.max(np.abs(installed_norm[valid] - get('standardized_advantage')[valid])) < 2e-3
            ids = get('episode_hi') * 65536 + get('episode_lo')
            matrix = np.column_stack([ids, get('verb_label'), get('pane3d'), get('selection_changed'),
                get('full_episode'), get('env_end'), get('steps_to_chunk_end'),
                get('standardized_advantage'), raw_valid_norm, center(lambda1), center(finite)])
            packets.append(matrix[valid])
    assert packets, 'No completed boundary batches'
    a = np.concatenate(packets)
    miss = (a[:, 1] == 3) & (a[:, 2] == 1) & (a[:, 3] == 0)
    groups = {'all': np.ones(len(a), bool), 'complete_episode': a[:, 4] == 1,
              'ends_at_rollout_cut': a[:, 5] == 0,
              'tail_of_cut_episode': (a[:, 5] == 1) & (a[:, 4] == 0),
              'within_20_steps_of_rollout_cut': (a[:, 5] == 0) & (a[:, 6] < 20),
              'earlier_in_cut_fragment': (a[:, 5] == 0) & (a[:, 6] >= 20)}
    out = {'batches': len(packets), 'actions': len(a), 'episodes': len(np.unique(a[:, 0])),
           'max_independent_gae_error': max_error, 'groups': {}}
    for name, mask in groups.items():
        m = mask & miss
        episodes = a[m, 0]
        entry = {'actions': int(mask.sum()), 'action_fraction': float(mask.mean()),
                 '3d_misses': int(m.sum()), '3d_miss_rate': float(m.sum() / mask.sum()) if mask.any() else None}
        for label, col in [('actual_gae_positive', 7), ('valid_only_normalization_positive', 8),
                           ('lambda1_bootstrapped_positive', 9)]:
            entry[label] = interval((a[m, col] > 0).astype(float), episodes)
        entry['positive_gae_turns_nonpositive_with_lambda1'] = interval(
            ((a[m, 7] > 0) & (a[m, 9] <= 0)).astype(float), episodes)
        if name == 'complete_episode':
            entry['finite_episode_return_positive'] = interval((a[m, 10] > 0).astype(float), episodes)
        out['groups'][name] = entry
    return out


ap = argparse.ArgumentParser()
ap.add_argument('--output', default='boundary_credit_report.json')
ap.add_argument('--arms', nargs='+', default=['default', 'complete', 'short'])
args = ap.parse_args()
result = {'arms': {arm: arm_report('boundary_' + arm + '/batches') for arm in args.arms},
          'limitations': [
              'Positive advantages are not proof an action causally helped.',
              'Lambda=1 uses the actual critic bootstrap at fragment/time-limit ends; it is a full observed-episode return only for complete episodes.',
              'Finite-episode returns use zero tail value and change the interpretation of TimeLimit versus continuing-task bootstrapping.',
              'Whole-batch centering is held to valid transitions for estimator comparisons; actual GAE also includes masked bootstrap slots in normalization.',
              'Three updates per arm and one training seed are a diagnostic, not an asymptotic convergence test.']}
Path(args.output).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
