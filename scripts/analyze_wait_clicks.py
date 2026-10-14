"""Episode-clustered loading associations in click audits and all-action traces."""
import argparse
import json
from pathlib import Path

import numpy as np


def association(records, condition, outcome):
    """Rate difference (condition true minus false), resampling whole episodes."""
    episodes = sorted({r['idx'] for r in records})
    counts = np.zeros((len(episodes), 4))
    for i, idx in enumerate(episodes):
        for r in records:
            if r['idx'] != idx:
                continue
            k = 0 if condition(r) else 2
            counts[i, k] += 1
            counts[i, k + 1] += float(outcome(r))
    totals = counts.sum(0)
    rates = [totals[k + 1] / totals[k] if totals[k] else None for k in (0, 2)]
    rng = np.random.default_rng(7)
    samples = counts[rng.integers(len(episodes), size=(4000, len(episodes)))].sum(1)
    valid = (samples[:, 0] > 0) & (samples[:, 2] > 0)
    diffs = samples[valid, 1] / samples[valid, 0] - samples[valid, 3] / samples[valid, 2]
    both = (counts[:, 0] > 0) & (counts[:, 2] > 0)
    within = counts[both, 1] / counts[both, 0] - counts[both, 3] / counts[both, 2]
    within_ci = None
    if len(within):
        means = within[rng.integers(len(within), size=(4000, len(within)))].mean(1)
        within_ci = np.quantile(means, [.025, .975]).tolist()
    return dict(episodes=len(episodes), condition_n=int(totals[0]), other_n=int(totals[2]),
                condition_rate=rates[0], other_rate=rates[1],
                difference=None if None in rates else rates[0] - rates[1],
                within_episode_n=len(within),
                within_episode_equal_weight_difference=float(within.mean()) if len(within) else None,
                within_episode_bootstrap_95ci=within_ci,
                episode_bootstrap_95ci=np.quantile(diffs, [.025, .975]).tolist() if len(diffs) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--trace', nargs='*', default=[])
    args = ap.parse_args()
    result = {'limitations': [
        'Observational associations do not establish that waiting caused a miss.',
        'Historical audits condition on sampled double clicks; they lack exposure at other verbs.',
        'Historical mesh_loading includes refinement and completed-but-not-adopted futures.',
        'Stale tiles need not imply an unusable observation or an unfinished fetch.',
        'Intervals resample episodes and do not represent training-seed uncertainty.']}
    result['limitations'].append('Multiple exploratory associations are reported without multiplicity correction.')
    result['historical'] = {}
    for arm in ['control', 'zero', 'free', 'both', 'noop']:
        rr = [r for i in (0, 10, 20, 30) for r in json.load(open(
            f'dbl_zmaxleft-v7j-compact-{arm}_{i}.json'))['records']]
        r3 = [r for r in rr if r['pane'] == '3d']
        misses = [r for r in r3 if not r['hit']]
        tests = {
            '3d_choice_by_mesh_loading': association(rr, lambda r: r['mesh_loading'], lambda r: r['pane'] == '3d'),
            '3d_choice_by_stale_tiles': association(rr, lambda r: r['stale_tiles'], lambda r: r['pane'] == '3d'),
            '3d_miss_by_mesh_loading': association(r3, lambda r: r['mesh_loading'], lambda r: not r['hit']),
            '3d_miss_by_stale_tiles': association(r3, lambda r: r['stale_tiles'], lambda r: not r['hit']),
        }
        tests['3d_misses'] = len(misses)
        tests['3d_misses_mesh_loading'] = sum(r['mesh_loading'] for r in misses)
        tests['3d_misses_visible_mesh_missing'] = sum(r['loaded_meshes'] < r['n_visible'] for r in misses)
        tests['3d_misses_stale_tiles'] = sum(r['stale_tiles'] for r in misses)
        tests['3d_misses_no_loading_flags'] = sum(not r['stale_tiles'] and not r['mesh_loading'] for r in misses)
        result['historical'][arm] = tests
    if args.trace:
        rr = [r for p in args.trace for r in json.load(open(p))['records']]
        assert len({(r['idx'], r['step']) for r in rr}) == len(rr)
        for idx in sorted({r['idx'] for r in rr}):
            last_move = last_select = -1000
            for r in sorted((s for s in rr if s['idx'] == idx), key=lambda s: s['step']):
                r['recent_move'] = r['step'] - last_move <= 5
                r['recent_selection'] = r['step'] - last_select <= 5
                if r['position_changed']:
                    last_move = r['step']
                if r['added']:
                    last_select = r['step']
        conditions = {
            'stale_tiles': lambda r: r['before']['stale_tiles'],
            'tile_fetch_pending': lambda r: r['before']['tile_fetch_pending'],
            'mesh_fetch_pending': lambda r: r['before']['mesh_fetch_pending'],
            'visible_mesh_missing': lambda r: r['before']['visible_mesh_missing'],
            'any_pending_fetch': lambda r: r['before']['tile_fetch_pending'] or r['before']['mesh_fetch_pending'],
            'recent_move_5_steps': lambda r: r['recent_move'],
            'recent_selection_5_steps': lambda r: r['recent_selection'],
        }
        outcomes = {
            'double_miss': lambda r: r['double_miss'],
            '3d_double_miss': lambda r: r['double_miss'] and r['double_pane'] == '3d',
            'double_click': lambda r: r['action'][0] == 3,
            'state_unchanged': lambda r: r['state_unchanged'],
            'policy_probability_double': lambda r: r['p_double'],
            'policy_probability_3d_double': lambda r: r['p_double'] * (1 - r['grid_2d_mass']),
        }
        result['all_steps'] = {name: {out: association(rr, cond, fn) for out, fn in outcomes.items()}
                               for name, cond in conditions.items()}
        result['episodes'] = len({r['idx'] for r in rr})
        result['steps'] = len(rr)
        result['neutral_nonclicks'] = sum(r['state_unchanged'] and r['action'][0] in (1, 2, 4) for r in rr)
        result['3d_misses_with_pending_fetch'] = sum(r['double_miss'] and r['double_pane'] == '3d'
            and conditions['any_pending_fetch'](r) for r in rr)
        result['3d_misses_total'] = sum(r['double_miss'] and r['double_pane'] == '3d' for r in rr)
        result['per_episode'] = [{
            'idx': idx, 'steps': sum(r['idx'] == idx for r in rr),
            '3d_misses': sum(r['idx'] == idx and r['double_miss'] and r['double_pane'] == '3d' for r in rr),
            'pending_fetch_steps': sum(r['idx'] == idx and conditions['any_pending_fetch'](r) for r in rr),
        } for idx in sorted({r['idx'] for r in rr})]
        result['examples'] = {}
        for idx in sorted({r['idx'] for r in rr})[:5]:
            episode = [r for r in rr if r['idx'] == idx]
            examples = [r for r in episode if r['double_miss'] and r['double_pane'] == '3d'][:3]
            result['examples'][idx] = [{
                'miss_step': r['step'],
                'window': [s for s in episode if r['step'] - 2 <= s['step'] <= r['step'] + 3]
            } for r in examples]
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'examples'}, indent=2))


if __name__ == '__main__':
    main()
