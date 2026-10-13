"""Record actual double-click outcomes and pre-click conditions on holdout starts."""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(__file__))
from eval_zmaxleft import StatePklPolicy, state_from_row
from ngllib import state as S
from ngllib_agent.env_build import build_env, load_config
from ray.rllib.core.columns import Columns


class AuditPolicy(StatePklPolicy):
    def act(self, obs):
        batch = {Columns.OBS: {k: torch.from_numpy(np.asarray(v, np.float32)).unsqueeze(0)
                               for k, v in obs.items()}}
        with torch.no_grad():
            out = self.module.forward_inference(batch)
            dist = self.dist_cls.from_logits(out[Columns.ACTION_DIST_INPUTS])
            a = dist.sample().squeeze(0).cpu().numpy()
            grid = dist._cats[2]
            probabilities = torch.softmax(grid.logits, dim=-1)
            self.grid_probabilities = probabilities[0].cpu().numpy()
            self.audit = {"grid_entropy": float(grid.entropy().item()),
                          "p_double": float(dist._type_probs()[0, 3]),
                          "grid_2d_mass": float(probabilities[0].reshape(16, 32)[:, :16].sum()),
                          "chosen_probability": float(probabilities[0, int(a[2])])}
        return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default='configs/native_zmaxleft_v7h_eval.yaml')
    ap.add_argument("--state-pkl", default='/scratch/kp0374/checkpoints/zmaxleft-v7h/ckpt_000800.pkl')
    args = ap.parse_args()
    cfg = load_config(args.config)
    env = build_env(cfg)
    renderer = env.unwrapped._renderer
    policy = AuditPolicy(args.state_pkl,
                         env, cfg['model'], stochastic=True)
    sample_orig, click_orig = renderer._cell_sample, renderer.click
    current = {}

    def sample(valid, x, y, *, pane_x, toolbar):
        result = sample_orig(valid, x, y, pane_x=pane_x, toolbar=toolbar)
        if current:
            current['pane_has_target'] |= bool(valid.any())
            current['cell_has_target'] |= result is not None
        return result

    def click(kind, x, y, modifiers):
        if kind != 'double_click':
            return click_orig(kind, x, y, modifiers)
        st = renderer._state
        current.update(pane='3d' if x >= 900 else '2d', n_visible=len(S.visible_segments(st['segments'])),
                       loaded_meshes=sum(renderer._renderer.has_mesh(rid) for rid in S.visible_segments(st['segments'])),
                       missing_tiles=renderer._tiles.get('ids') is None or renderer._tiles.get('em') is None,
                       mesh_loading=bool(renderer._mesh_futs),
                       stale_tiles=renderer._tile_key != renderer._tile_key_for(st['position'], st['crossSectionScale']),
                       projection_scale=float(st['projectionScale']), xs_scale=float(st['crossSectionScale']),
                       pane_has_target=False, cell_has_target=False)
        before = set(S.visible_segments(st['segments']))
        click_orig(kind, x, y, modifiers)
        after = set(S.visible_segments(renderer._state['segments']))
        current.update(hit=before != after, added=len(after - before), removed=len(before - after))

    renderer._cell_sample, renderer.click = sample, click
    rows = pq.read_table('/scratch/kp0374/neurogym-agent/eval_zmaxleft_v1.parquet').to_pylist()
    # Evenly cover all 200 starts, with shard offsets selecting disjoint episodes.
    picks = rows[::5][args.offset:args.offset + args.limit]
    records = []
    try:
        for row in picks:
            idx = int(row['idx'])
            torch.manual_seed(10000 + idx)
            state, info = state_from_row(row)
            obs, _ = env.reset(options={'state': state, 'task_info': info})
            previous_cell = None
            previous_miss = False
            for step in range(500):
                a = policy.act(obs)
                current.clear()
                obs, reward, terminated, truncated, _ = env.step(a)
                if int(a[0]) == 3:
                    cell = int(a[2])
                    current.update(idx=idx, step=step, cell=cell, row=cell // 32, col=cell % 32,
                                   repeat_previous_double=cell == previous_cell,
                                   repeat_previous_miss=cell == previous_cell and previous_miss,
                                   **policy.audit)
                    records.append(dict(current))
                    previous_cell, previous_miss = cell, not current['hit']
                if terminated or truncated:
                    break
            with open(args.output, 'w') as f:
                json.dump({'episodes': args.offset + len([r for r in picks if int(r['idx']) <= idx]),
                           'records': records}, f)
            print(json.dumps({'idx': idx, 'double_clicks': len(records)}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
