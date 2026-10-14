"""Measure local entropy pressure toward each pane, clustered by episode."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='dbl_pane_entropy_v7i_grid.json')
    ap.add_argument('--output', default='pane_entropy_v7i_grid_summary.json')
    args = ap.parse_args()
    data = json.loads(Path(args.input).read_text())
    records = data['records']
    episodes = sorted(data['completed_indices'])
    assert len(episodes) == len(set(episodes)) and records
    # Per-state grid bonus at the source checkpoint: entropy_coeff=.01,
    # double-click grid scale=.01, normalized by ln(512), verb weight detached.
    beta = .01 * .01 / math.log(512)
    metrics = {
        'grid_2d_mass': lambda r: r['grid_2d_mass'],
        '2d_conditional_entropy': lambda r: r['2d_conditional_entropy'],
        '3d_conditional_entropy': lambda r: r['3d_conditional_entropy'],
        'fraction_entropy_pushes_3d': lambda r: float(r['entropy_gradient_toward_2d'] < 0),
        'raw_entropy_gradient_toward_2d': lambda r: r['entropy_gradient_toward_2d'],
        'weighted_bonus_gradient_toward_2d': lambda r: beta * r['p_double'] * r['entropy_gradient_toward_2d'],
    }
    if all('total_entropy_gradient_toward_double' in r for r in records):
        metrics.update({
            'verb_only_entropy_gradient_toward_double': lambda r: r['verb_only_entropy_gradient_toward_double'],
            'branch_weight_entropy_gradient_toward_double': lambda r: r['branch_weight_entropy_gradient_toward_double'],
            'total_entropy_gradient_toward_double': lambda r: r['total_entropy_gradient_toward_double'],
            'fraction_total_entropy_pushes_double': lambda r: float(r['total_entropy_gradient_toward_double'] > 0),
            'weighted_bonus_gradient_toward_double': lambda r: .01 * r['total_entropy_gradient_toward_double'],
        })
    rng = np.random.default_rng(42)
    indices = rng.integers(len(episodes), size=(4000, len(episodes)))
    output = {'episodes': len(episodes), 'sampled_double_click_states': len(records),
              'conditional_entropy_coefficient': beta,
              'limitations': ['Conditions on sampled double-click states.',
                              'Measures direct logit-space entropy gradient, not the full shared-network update.',
                              'A negative gradient establishes local pressure, not its dominance over PPO advantage gradients.'],
              'metrics': {}}
    for name, fn in metrics.items():
        groups = [[fn(r) for r in records if r['idx'] == idx] for idx in episodes]
        counts = np.asarray([len(v) for v in groups])
        sums = np.asarray([sum(v) for v in groups])
        denominator = counts[indices].sum(axis=1)
        boot = sums[indices].sum(axis=1)[denominator > 0] / denominator[denominator > 0]
        output['metrics'][name] = {'mean': float(sums.sum() / counts.sum()),
                                  'episode_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist()}
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
