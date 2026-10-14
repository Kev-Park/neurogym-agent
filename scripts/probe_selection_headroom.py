"""Compare historical versus currently visible headroom at click decisions."""
import argparse
import copy
import json
import os
import sys

import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(__file__))
from eval_zmaxleft import state_from_row
from probe_double_clicks import AuditPolicy
from ngllib_agent.env_build import build_env, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='selection_headroom_v7i_grid.json')
    ap.add_argument('--limit', type=int, default=24)
    args = ap.parse_args()
    cfg = load_config('configs/native_zmaxleft_v7h_eval.yaml')
    cfg['env']['pane_mode'] = 'fresh'
    env = build_env(cfg)
    raw = {}
    original = env.unwrapped._make_obs
    def tap(state, image):
        obs = original(state, image)
        raw.update(copy.deepcopy(obs))
        return obs
    env.unwrapped._make_obs = tap
    policy = AuditPolicy('/scratch/kp0374/checkpoints/zmaxleft-v7i-grid/ckpt_000900.pkl',
                         env, cfg['model'], stochastic=True)
    rows = pq.read_table('/scratch/kp0374/neurogym-agent/eval_zmaxleft_v1.parquet').to_pylist()[::8][:args.limit]
    records, completed = [], []
    try:
        for row in rows:
            idx = int(row['idx'])
            torch.manual_seed(20000 + idx)
            state, info = state_from_row(row)
            obs, _ = env.reset(options={'state': state, 'task_info': info})
            seen, heights = set(), {}
            best = float(raw['position'][2])
            for step in range(400):
                visible = {str(s) for s in raw['segments']}
                seen.update(visible)
                best = max(best, float(raw['position'][2]))
                for sid, height in zip(raw['segments'], raw.get('segment_zmax', ())):
                    if float(height) > 0:
                        heights[str(sid)] = float(height)
                a = policy.act(obs)
                if step >= 50 and int(a[0]) == 3:
                    historical = max([0.] + [heights[s] - best for s in seen if s in heights])
                    current = max([0.] + [heights[s] - best for s in visible if s in heights])
                    records.append({'idx': idx, 'step': step, 'historical_headroom': historical,
                                    'visible_headroom': current,
                                    'extra_cost_from_deselected_history': .00005 * (historical - current),
                                    'historical_selection_cost': .0015 + .00005 * historical,
                                    'visible_selection_cost': .0015 + .00005 * current,
                                    'sampled_pane': '2d' if int(a[2]) % 32 < 16 else '3d',
                                    **policy.audit})
                    break
                obs, _, terminated, truncated, _ = env.step(a)
                if terminated or truncated:
                    break
            completed.append(idx)
            with open(args.output, 'w') as f:
                json.dump({'completed_indices': completed, 'records': records}, f)
            print(json.dumps({'idx': idx, 'decisions': len(records)}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
