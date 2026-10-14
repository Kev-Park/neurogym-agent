"""Descriptive miss associations with episode-cluster bootstrap intervals."""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    payloads = [json.load(open(path)) for path in args.inputs]
    records = [r for payload in payloads for r in payload['records']]
    covered = {idx for payload in payloads for idx in payload.get(
        'completed_indices', sorted({r['idx'] for r in payload['records']}))}
    rng = np.random.default_rng(42)

    def summary(rows):
        return {'attempts': len(rows), 'hits': sum(r['hit'] for r in rows),
                'accuracy': float(np.mean([r['hit'] for r in rows])) if rows else None}

    def association(rows, factor):
        episodes = sorted(covered)
        counts = []
        for idx in episodes:
            episode = [r for r in rows if r['idx'] == idx]
            yes = [r for r in episode if factor(r)]
            no = [r for r in episode if not factor(r)]
            counts.append([sum(not r['hit'] for r in yes), len(yes),
                           sum(not r['hit'] for r in no), len(no)])
        counts = np.array(counts)
        total = counts.sum(axis=0)
        if not total[1] or not total[3]:
            return {'factor_true_n': int(total[1]), 'factor_false_n': int(total[3]),
                    'estimable': False}
        bootstrap = counts[rng.integers(0, len(episodes), size=(2000, len(episodes)))].sum(axis=1)
        bootstrap = bootstrap[(bootstrap[:, 1] > 0) & (bootstrap[:, 3] > 0)]
        differences = bootstrap[:, 0] / bootstrap[:, 1] - bootstrap[:, 2] / bootstrap[:, 3]
        return {'factor_true_n': int(total[1]), 'factor_false_n': int(total[3]),
                'miss_rate_true': float(total[0] / total[1]),
                'miss_rate_false': float(total[2] / total[3]),
                'miss_rate_difference': float(total[0] / total[1] - total[2] / total[3]),
                'episode_bootstrap_95ci': np.quantile(differences, [0.025, 0.975]).tolist()}

    output = {'episodes': len(covered), 'episodes_with_double_clicks': len({r['idx'] for r in records}),
              'overall': summary(records),
              'panes': {p: summary([r for r in records if r['pane'] == p]) for p in ['2d', '3d']}}
    misses = [r for r in records if not r['hit']]
    output['miss_breakdown'] = {
        'no_target_anywhere_in_pane': sum(not r['pane_has_target'] for r in misses),
        'target_elsewhere_but_selected_cell_empty': sum(r['pane_has_target'] and not r['cell_has_target'] for r in misses),
        'cell_target_but_no_selection_change': sum(r['cell_has_target'] for r in misses)}
    output['outcomes'] = {'additions': sum(r['added'] for r in records),
                          'removals': sum(r['removed'] for r in records)}
    output['associations'] = {}
    for pane in ['all', '2d', '3d']:
        rows = records if pane == 'all' else [r for r in records if r['pane'] == pane]
        if not rows:
            continue
        entropy_median = float(np.median([r['grid_entropy'] for r in rows]))
        zoom_median = float(np.median([r['projection_scale'] for r in rows]))
        factors = {'3d_pane': lambda r: r['pane'] == '3d',
                   'mesh_loading': lambda r: r['mesh_loading'],
                   'stale_tiles': lambda r: r['stale_tiles'],
                   'missing_tiles': lambda r: r.get('missing_tiles', False),
                   'repeat_previous_miss': lambda r: r['repeat_previous_miss'],
                   'no_visible_segments': lambda r: r['n_visible'] == 0,
                   'edge_row': lambda r: r['row'] in [0, 15],
                   'higher_grid_entropy': lambda r: r['grid_entropy'] > entropy_median,
                   'higher_projection_scale': lambda r: r['projection_scale'] > zoom_median}
        output['associations'][pane] = {name: association(rows, factor) for name, factor in factors.items()}
    output['grid'] = {'mean_entropy_nats': float(np.mean([r['grid_entropy'] for r in records])),
                      'mean_effective_cells': float(np.mean([np.exp(r['grid_entropy']) for r in records])),
                      'mean_2d_probability_mass': float(np.mean([r['grid_2d_mass'] for r in records]))}
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
