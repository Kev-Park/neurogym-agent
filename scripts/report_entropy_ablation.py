"""Summarize the matched entropy arms and paired click interventions."""
import json
from pathlib import Path

import numpy as np


def paired(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'n': 0}
    rng = np.random.default_rng(7)
    means = a[rng.integers(len(a), size=(4000, len(a)))].mean(axis=1)
    return {'n': len(a), 'mean_difference': float(a.mean()),
            'median_difference': float(np.median(a)),
            'bootstrap_mean_95ci': np.quantile(means, [0.025, 0.975]).tolist()}


def main():
    output = {'limitations': [
        'One training continuation per arm; bootstrap intervals do not represent training-seed uncertainty.',
        'Counterfactuals use settled fresh rendering, not exact cloning of asynchronous renderer queues.',
        'Valid-target branches use privileged hit testing for diagnosis, not a deployable policy.',
        '100-step counterfactual horizon can miss longer-term selection benefits.']}
    arms = {}
    evals = {}
    for arm in ['control', 'verb', 'grid', 'both']:
        run = f'zmaxleft-v7i-{arm}'
        evaluation = json.load(open(f'eval_{run}_full.json'))
        assert evaluation['summary']['n'] == 200 and not evaluation['summary']['partial']
        stats = json.load(open(f'dbl_{run}_stats.json'))
        assert stats['episodes'] == 40
        videos = json.load(open(f'eval_videos_{run}/manifest.json'))
        assert len(videos) == 6
        rows = evaluation['per_state']
        evals[arm] = {r['idx']: r for r in rows}
        attempts, hits = stats['overall']['attempts'], stats['overall']['hits']
        arms[arm] = {'eval': evaluation['summary'], 'double_clicks_per_audit_episode': attempts / 40,
                     'misses_per_audit_episode': (attempts - hits) / 40,
                     'selection_changes_per_audit_episode': hits / 40,
                     'double_click_accuracy': stats['overall']['accuracy'],
                     'pane_accuracy': stats['panes'], 'grid': stats['grid'],
                     'double_clicks_per_full_eval_episode': np.mean([r['diag']['verbs'][3] for r in rows])}
    for arm in ['verb', 'grid', 'both']:
        keys = sorted(set(evals['control']) & set(evals[arm]))
        arms[arm]['paired_vs_control_dz500'] = paired(
            [evals[arm][k]['dz@500'] - evals['control'][k]['dz@500'] for k in keys])
    output['arms'] = arms
    records = [r for i in range(4) for r in json.load(open(f'cf_click_part{i}.json'))['records']
               if 'branch' in r]
    by_pair = {}
    for r in records:
        by_pair.setdefault((r['idx'], r['repeat']), {})[r['branch']] = r
    cf = {}
    for branch in ['sampled', 'valid_2d', 'valid_3d']:
        by_state = {}
        for (idx, _), choices in by_pair.items():
            if branch in choices and 'noop' in choices:
                b, n = choices[branch], choices['noop']
                by_state.setdefault(idx, []).append([b['discounted_return'] - n['discounted_return'],
                                                     b['new_peak_dz'] - n['new_peak_dz']])
        values = np.asarray([np.mean(v, axis=0) for v in by_state.values()])
        cf[branch] = {'vs_noop_discounted_return': paired(values[:, 0] if len(values) else []),
                      'vs_noop_climb': paired(values[:, 1] if len(values) else [])}
    output['counterfactuals'] = cf
    Path('entropy_v7i_report.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
