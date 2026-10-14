"""Summarize the matched entropy arms and paired click interventions."""
import json
import argparse
import shutil
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--compact', action='store_true')
    args = ap.parse_args()
    prefix = 'zmaxleft-v7j-compact' if args.compact else 'zmaxleft-v7j'
    stem = 'entropy_v7j_compact' if args.compact else 'entropy_v7j'
    output = {'limitations': [
        'One training continuation per arm; bootstrap intervals do not represent training-seed uncertainty.',
        'The higher no-op cost also affects unsuccessful 3D right-clicks; it is not double-click-specific.',
        'Removing selection costs changes both fixed load cost and unexplored-headroom cost.',
        'A 100-update continuation can reveal sensitivity but cannot prove asymptotic convergence.',
        'Click-audit probability summaries condition on sampled double-click decisions.']}
    arms = {}
    evals = {}
    audit_episodes = {}
    for arm in ['control', 'zero', 'free', 'both', 'noop']:
        run = f'{prefix}-{arm}'
        evaluation = json.load(open(f'eval_{run}_full.json'))
        assert evaluation['summary']['n'] == 200 and not evaluation['summary']['partial']
        stats = json.load(open(f'dbl_{run}_stats.json'))
        assert stats['episodes'] == 40
        videos = json.load(open(f'eval_videos_{run}/manifest.json'))
        assert len(videos) == 6
        rows = evaluation['per_state']
        evals[arm] = {r['idx']: r for r in rows}
        attempts, hits = stats['overall']['attempts'], stats['overall']['hits']
        payloads = [json.load(open(f'dbl_{run}_{i}.json')) for i in [0, 10, 20, 30]]
        covered = [i for p in payloads for i in p['completed_indices']]
        assert len(covered) == len(set(covered)) == 40
        records = [r for p in payloads for r in p['records']]
        audit_episodes[arm] = {}
        for idx in covered:
            episode = [r for r in records if r['idx'] == idx]
            audit_episodes[arm][idx] = {
                '3d_attempts': sum(r['pane'] == '3d' for r in episode),
                '2d_attempts': sum(r['pane'] == '2d' for r in episode),
                '3d_misses': sum(r['pane'] == '3d' and not r['hit'] for r in episode),
                'misses': sum(not r['hit'] for r in episode),
                'selection_changes': sum(r['hit'] for r in episode),
                'additions': sum(r['added'] for r in episode),
                'removals': sum(r['removed'] for r in episode)}
        arms[arm] = {'eval': evaluation['summary'], 'double_clicks_per_audit_episode': attempts / 40,
                     'misses_per_audit_episode': (attempts - hits) / 40,
                     'selection_changes_per_audit_episode': hits / 40,
                     'double_click_accuracy': stats['overall']['accuracy'],
                     'pane_accuracy': stats['panes'], 'grid': stats['grid'],
                     'double_clicks_per_full_eval_episode': np.mean([r['diag']['verbs'][3] for r in rows])}
        if records and all('historical_headroom' in r for r in records):
            arms[arm]['mechanism_diagnostics'] = {
                'sampled_3d_fraction': float(np.mean([r['pane'] == '3d' for r in records])),
                'fraction_entropy_gradient_pushes_3d': float(np.mean([
                    r['entropy_gradient_toward_2d'] < 0 for r in records])),
                'mean_raw_entropy_gradient_toward_2d': float(np.mean([
                    r['entropy_gradient_toward_2d'] for r in records])),
                'fraction_total_entropy_pushes_double': float(np.mean([
                    r['total_entropy_gradient_toward_double'] > 0 for r in records])),
                'mean_total_entropy_gradient_toward_double': float(np.mean([
                    r['total_entropy_gradient_toward_double'] for r in records])),
                'fraction_with_deselected_history_surcharge': float(np.mean([
                    r['historical_headroom'] > r['visible_headroom'] + 1e-6 for r in records])),
                'mean_hypothetical_baseline_selection_cost': float(np.mean([
                    .0015 + .00005 * r['historical_headroom'] for r in records])),
                'mean_hypothetical_baseline_extra_cost_from_deselected_history': float(np.mean([
                    .00005 * (r['historical_headroom'] - r['visible_headroom']) for r in records])),
                'note': 'Cost diagnostics use the baseline formula even in free-selection arms; they are not observed rewards.'}
    for arm in ['zero', 'free', 'both', 'noop']:
        keys = sorted(set(evals['control']) & set(evals[arm]))
        arms[arm]['paired_vs_control_dz500'] = paired(
            [evals[arm][k]['dz@500'] - evals['control'][k]['dz@500'] for k in keys])
    output['arms'] = arms
    output['paired_audit_vs_control'] = {}
    for arm in ['zero', 'free', 'both', 'noop']:
        assert set(audit_episodes[arm]) == set(audit_episodes['control'])
        output['paired_audit_vs_control'][arm] = {
            metric: paired([audit_episodes[arm][idx][metric] - audit_episodes['control'][idx][metric]
                            for idx in sorted(audit_episodes[arm])])
            for metric in next(iter(audit_episodes[arm].values()))}
    # The 2x2 core distinguishes entropy removal from selection-cost removal.
    # Outcomes remain in common task units, rather than comparing changed rewards.
    output['factorial_dz500'] = {}
    assert all(set(evals[a]) == set(evals['control']) for a in evals)
    for name, weights in {
        'entropy_removal_main_effect': {'zero': .5, 'control': -.5, 'both': .5, 'free': -.5},
        'selection_cost_removal_main_effect': {'free': .5, 'control': -.5, 'both': .5, 'zero': -.5},
        'interaction': {'both': 1, 'zero': -1, 'free': -1, 'control': 1},
    }.items():
        output['factorial_dz500'][name] = paired([
            sum(w * evals[a][idx]['dz@500'] for a, w in weights.items())
            for idx in sorted(evals['control'])])
    output['design'] = 'Shared v7i-grid/900 source; 100 updates; control, zero grid entropy, free new selections, both, 10x no-op cost. Verb entropy unchanged.'
    output['runner_layout'] = '2 runners x 2 envs; dedicated learner GPU and one runner GPU' if args.compact else '18 runners x 2 envs; four GPUs'
    Path(f'{stem}_report.json').write_text(json.dumps(output, indent=2))
    destination = Path(f'{stem}_results')
    destination.mkdir(exist_ok=True)
    for name in [f'{stem}_report.json', f'{stem}_pipeline.json']:
        shutil.copy2(name, destination / name)
    for arm in arms:
        run = f'{prefix}-{arm}'
        for name in [f'eval_{run}_full.json', f'dbl_{run}_stats.json']:
            shutil.copy2(name, destination / name)
        for i in [0, 10, 20, 30]:
            shutil.copy2(f'dbl_{run}_{i}.json', destination / f'dbl_{run}_{i}.json')
        shutil.copytree(f'eval_videos_{run}', destination / f'eval_videos_{run}', dirs_exist_ok=True)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
