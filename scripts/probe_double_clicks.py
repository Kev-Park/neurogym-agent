"""Record actual double-click outcomes and pre-click conditions on holdout starts."""
import argparse
import json
import math
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


def selectable_cells(valid, *, pane_x, toolbar, scale, grid, bounds):
    """Cells containing valid displayed pixels, using picker pixel centres."""
    rows, cols = grid
    x0, y0, x1, y1 = bounds
    yy, xx = np.nonzero(valid)
    xs, ys = (xx + pane_x + .5) / scale, (yy + toolbar + .5) / scale
    inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    cc = ((xs[inside] - x0) * cols / (x1 - x0)).astype(int)
    rr = ((ys[inside] - y0) * rows / (y1 - y0)).astype(int)
    result = np.zeros(rows * cols, dtype=bool)
    result[rr * cols + cc] = True
    return result


def double_verb_entropy_pressure(dist):
    """Exact dH/d(double-verb-logit) for the detached split-grid policy."""
    assert dist._split and dist._cell_entropy_stopgrad
    p = dist._type_probs()[0]
    pd = p[3]
    nverb, _, _, nr, _, _, nz = dist._input_lens
    ln = math.log if dist._normalize_entropy else lambda _: 1.
    verb = dist._verb_entropy_scale * pd * ((p * p.clamp_min(1e-30).log()).sum()
                                          - pd.clamp_min(1e-30).log()) / ln(nverb)
    rotation = sum(dist._cats[i].entropy()[0] for i in [3, 4, 5])
    rotation /= 3 * ln(nr) if dist._normalize_entropy else 1.
    zoom = dist._cats[6].entropy()[0] / ln(nz)
    others = -pd * (p[1] * rotation + (p[2] + p[4]) * zoom)
    return float(verb), float(others), float(verb + others)


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
            pane_probs = probabilities[0].reshape(16, 32)
            left, right = pane_probs[:, :16].flatten(), pane_probs[:, 16:].flatten()
            p2, p3 = left.sum(), right.sum()
            def conditional_entropy(values, mass):
                q = values / mass.clamp_min(1e-30)
                return -(q * q.clamp_min(1e-30).log()).sum()
            h2, h3 = conditional_entropy(left, p2), conditional_entropy(right, p3)
            # Exact dH/d(delta): add delta to every 2D logit, holding the
            # conditional distributions within each pane fixed. Negative
            # means the entropy bonus locally favors shifting mass into 3D.
            pane_gradient = p2 * p3 * (p3.clamp_min(1e-30).log()
                                      - p2.clamp_min(1e-30).log() + h2 - h3)
            verb_gradient, branch_gradient, total_verb_gradient = double_verb_entropy_pressure(dist)
            self.audit = {"grid_entropy": float(grid.entropy().item()),
                          "verb_only_entropy_gradient_toward_double": verb_gradient,
                          "branch_weight_entropy_gradient_toward_double": branch_gradient,
                          "total_entropy_gradient_toward_double": total_verb_gradient,
                          "2d_conditional_entropy": float(h2),
                          "3d_conditional_entropy": float(h3),
                          "entropy_gradient_toward_2d": float(pane_gradient),
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
    raw = {}
    original_obs = env.unwrapped._make_obs
    def tap_obs(state, image):
        result = original_obs(state, image)
        raw.update(result)
        return result
    env.unwrapped._make_obs = tap_obs
    renderer = env.unwrapped._renderer
    policy = AuditPolicy(args.state_pkl,
                         env, cfg['model'], stochastic=True)
    sample_orig, click_orig = renderer._cell_sample, renderer.click
    current = {}
    target_cells = np.zeros(512, dtype=bool)

    def sample(valid, x, y, *, pane_x, toolbar):
        result = sample_orig(valid, x, y, pane_x=pane_x, toolbar=toolbar)
        if current:
            target_cells[:] |= selectable_cells(valid, pane_x=pane_x, toolbar=toolbar,
                                                scale=renderer.layout.capture_scale,
                                                grid=renderer.click_grid, bounds=renderer.click_bounds)
            current['pane_has_target'] |= bool(valid.any())
            current['cell_has_target'] |= result is not None
        return result

    def click(kind, x, y, modifiers):
        if kind != 'double_click':
            return click_orig(kind, x, y, modifiers)
        st = renderer._state
        target_cells.fill(False)
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
        pane = np.arange(512) % 32 < 16
        if current['pane'] == '3d':
            pane = ~pane
        p = policy.grid_probabilities
        current.update(selectable_cell_count=int(target_cells[pane].sum()),
                       uniform_target_hit_chance=float(target_cells[pane].mean()),
                       policy_target_hit_chance=float(p[target_cells & pane].sum() / p[pane].sum()))

    renderer._cell_sample, renderer.click = sample, click
    rows = pq.read_table('/scratch/kp0374/neurogym-agent/eval_zmaxleft_v1.parquet').to_pylist()
    # Evenly cover all 200 starts, with shard offsets selecting disjoint episodes.
    picks = rows[::5][args.offset:args.offset + args.limit]
    records = []
    completed = []
    try:
        for row in picks:
            idx = int(row['idx'])
            torch.manual_seed(10000 + idx)
            state, info = state_from_row(row)
            obs, _ = env.reset(options={'state': state, 'task_info': info})
            seen, heights = set(), {}
            best_z = float(raw['position'][2])
            previous_cell = None
            previous_miss = False
            for step in range(500):
                visible = {str(s) for s in raw['segments']}
                seen.update(visible)
                best_z = max(best_z, float(raw['position'][2]))
                for sid, height in zip(raw['segments'], raw.get('segment_zmax', ())):
                    if float(height) > 0:
                        heights[str(sid)] = float(height)
                historical_headroom = max([0.] + [heights[s] - best_z for s in seen if s in heights])
                visible_headroom = max([0.] + [heights[s] - best_z for s in visible if s in heights])
                a = policy.act(obs)
                current.clear()
                obs, reward, terminated, truncated, _ = env.step(a)
                if int(a[0]) == 3:
                    cell = int(a[2])
                    current.update(idx=idx, step=step, cell=cell, row=cell // 32, col=cell % 32,
                                   historical_headroom=historical_headroom,
                                   visible_headroom=visible_headroom,
                                   baseline_extra_cost_from_deselected_history=.00005 * (historical_headroom - visible_headroom),
                                   repeat_previous_double=cell == previous_cell,
                                   repeat_previous_miss=cell == previous_cell and previous_miss,
                                   **policy.audit)
                    records.append(dict(current))
                    previous_cell, previous_miss = cell, not current['hit']
                if terminated or truncated:
                    break
            completed.append(idx)
            with open(args.output, 'w') as f:
                json.dump({'episodes': len(completed), 'completed_indices': completed,
                           'records': records}, f)
            print(json.dumps({'idx': idx, 'double_clicks': len(records)}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
