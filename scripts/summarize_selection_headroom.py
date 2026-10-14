"""Summarize costs attributable to segments that are no longer selected."""
import json
from pathlib import Path

import numpy as np


def main():
    data = json.loads(Path('selection_headroom_v7i_grid.json').read_text())
    assert len(data['completed_indices']) == len(set(data['completed_indices'])) == 24
    records = data['records']
    assert records
    extras = np.array([r['extra_cost_from_deselected_history'] for r in records])
    costs = np.array([r['historical_selection_cost'] for r in records])
    rng = np.random.default_rng(42)
    boot = extras[rng.integers(len(extras), size=(4000, len(extras)))].mean(axis=1)
    output = {'checkpoint': 'zmaxleft-v7i-grid/900', 'completed_episodes': 24,
              'late_double_click_decisions': len(records),
              'decisions_with_deselected_history_surcharge': int((extras > 1e-9).sum()),
              'fraction_with_deselected_history_surcharge': float((extras > 1e-9).mean()),
              'mean_extra_cost': float(extras.mean()),
              'mean_extra_cost_episode_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist(),
              'max_extra_cost': float(extras.max()),
              'median_historical_selection_cost': float(np.median(costs)),
              'median_historical_selection_cost_in_miss_units': float(np.median(costs) / .0005),
              'limitations': ['One late sampled double-click decision per episode.',
                              'Fresh rendering; descriptive attribution, not a reward intervention.',
                              'This measures the price of a hypothetical unseen selection, not the sampled action reward.']}
    Path('selection_headroom_v7i_grid_summary.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
