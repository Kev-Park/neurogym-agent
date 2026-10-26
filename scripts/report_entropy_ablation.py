"""Summarize the matched entropy arms and paired click interventions."""
import json
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
    output = {'limitations': [
        'One training continuation per arm; bootstrap intervals do not represent training-seed uncertainty.',
        'Counterfactuals use settled fresh rendering, not exact cloning of asynchronous renderer queues.',
        'Valid-target branches use privileged hit testing for diagnosis, not a deployable policy.',
        'Valid branches target never-selected segments; reselect branches target previously seen, currently deselected segments. Selected targets are excluded because they toggle visibility.',
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
    candidate_coverage = {}
    grid_coverage = {}
    for (idx, _), choices in by_pair.items():
        record = choices.get('sampled')
        if record is None or 'grid_hit_distribution' not in record:
            continue
        grid_coverage.setdefault(idx, record['grid_hit_distribution'])
    output['grid_hit_distribution_by_state'] = grid_coverage
    output['grid_hit_distribution_summary'] = {}
    if grid_coverage:
        for pane in ['2d', '3d']:
            output['grid_hit_distribution_summary'][pane] = {}
            for category in ['empty', 'selected_visible', 'selected_hidden', 'reselect', 'new']:
                rows = [state[pane][category] for state in grid_coverage.values()]
                output['grid_hit_distribution_summary'][pane][category] = {
                    'mean_cells': float(np.mean([r['cells'] for r in rows])),
                    'mean_policy_probability_mass': float(np.mean([r['probability_mass'] for r in rows])),
                    'mean_distinct_segments': float(np.mean([r['distinct_segments'] for r in rows])),
                    'states_with_any_cells': sum(r['cells'] > 0 for r in rows),
                }
    for branch in ['sampled', 'hit_3d', 'valid_2d', 'valid_3d', 'reselect_2d', 'reselect_3d']:
        by_state = {}
        target_statuses = {}
        diagnostics = {'first_reward': {}, 'new_selection': {}, 'made_visible': {}}
        for (idx, _), choices in by_pair.items():
            if branch in choices and 'noop' in choices:
                b, n = choices[branch], choices['noop']
                by_state.setdefault(idx, []).append([b['discounted_return'] - n['discounted_return'],
                                                     b['new_peak_dz'] - n['new_peak_dz']])
                diagnostics['first_reward'].setdefault(idx, []).append(b['first_reward'])
                diagnostics['new_selection'].setdefault(idx, []).append(
                    bool((b.get('selection_delta') or {}).get('added')))
                diagnostics['made_visible'].setdefault(idx, []).append(
                    bool((b.get('visibility_delta') or {}).get('shown')))
                target = b.get('candidate_target') or {}
                status = target.get('target_status', 'unlabeled')
                target_statuses.setdefault(status, set()).add(idx)
        values = np.asarray([np.mean(v, axis=0) for v in by_state.values()])
        cf[branch] = {'vs_noop_discounted_return': paired(values[:, 0] if len(values) else []),
                      'vs_noop_climb': paired(values[:, 1] if len(values) else []),
                      'first_reward': paired([np.mean(v) for v in diagnostics['first_reward'].values()]),
                      'new_selection_rate': paired([np.mean(v) for v in diagnostics['new_selection'].values()]),
                      'made_visible_rate': paired([np.mean(v) for v in diagnostics['made_visible'].values()])}
        candidate_coverage[branch] = {
            'states_with_candidate': len(by_state),
            'states_by_target_status': {status: len(indices)
                                        for status, indices in target_statuses.items()}}
    output['counterfactuals'] = cf
    output['counterfactual_candidate_coverage'] = candidate_coverage
    Path('entropy_v7i_report.json').write_text(json.dumps(output, indent=2))
    destination = Path('entropy_v7i_results')
    destination.mkdir(exist_ok=True)
    for name in ['entropy_v7i_report.json', 'entropy_v7i_pipeline.json']:
        shutil.copy2(name, destination / name)
    for arm in arms:
        run = f'zmaxleft-v7i-{arm}'
        for name in [f'eval_{run}_full.json', f'dbl_{run}_stats.json']:
            shutil.copy2(name, destination / name)
        shutil.copytree(f'eval_videos_{run}', destination / f'eval_videos_{run}', dirs_exist_ok=True)
    for i in range(4):
        shutil.copy2(f'cf_click_part{i}.json', destination / f'cf_click_part{i}.json')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
