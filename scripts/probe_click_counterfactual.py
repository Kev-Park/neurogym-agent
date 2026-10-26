"""Paired fresh-renderer continuations at sampled double-click decisions.

Branches share physical state and continuation RNG seeds, not asynchronous
renderer queues. Fresh mode settles each branch. Preserve source reward history
when reconstructing the mature (novelty-free) training reward externally.
"""
import argparse
import copy
import json
import os
import sys
from dataclasses import fields

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(__file__))
from eval_zmaxleft import state_from_row
from probe_double_clicks import AuditPolicy
from ngllib import state as S
from ngllib_agent.env_build import action_spec_from_config, build_env, load_config
from ngllib_agent.wrappers.action import cell_to_pixel, decode
from ngllib_agent.rewards.z_free import ZFreeRewardConfig, make_zfree_reward_factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--limit', type=int, default=4)
    ap.add_argument('--horizon', type=int, default=100)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--output', required=True)
    ap.add_argument('--state-pkl', default='/scratch/kp0374/checkpoints/zmaxleft-v7h/ckpt_000800.pkl')
    ap.add_argument('--train-config', default='configs/native_zmaxleft_v7h.yaml')
    ap.add_argument('--only-3d-misses', action='store_true')
    ap.add_argument('--cases-file', help='Captured training states with linked learner advantages')
    args = ap.parse_args()
    cfg = load_config('configs/native_zmaxleft_v7h_eval.yaml')
    cfg['env']['pane_mode'] = 'fresh'
    source, branch = build_env(cfg), build_env(cfg)
    spec = action_spec_from_config(cfg['action'])
    policy = AuditPolicy(args.state_pkl,
                         source, cfg['model'], stochastic=True)
    train_cfg = load_config(args.train_config)
    keys = {f.name for f in fields(ZFreeRewardConfig)}
    rc = ZFreeRewardConfig(**{k: v for k, v in train_cfg['reward'].items() if k in keys})
    # v7h/800 is far beyond the novelty scaffold's end.
    from dataclasses import replace
    rc = replace(rc, select_novelty_bonus=0.0)
    raw = {}
    for name, env in [('source', source), ('branch', branch)]:
        base = env.unwrapped
        original = base._make_obs
        def tap(st, image, original=original, name=name):
            obs = original(st, image)
            raw[name] = copy.deepcopy(obs)
            return obs
        base._make_obs = tap
    # -1 is a diagnostic-only explicit left-click no-op, outside policy space.
    wrapper = branch
    while not hasattr(wrapper, '_action_spec'):
        wrapper = wrapper.env
    translate = wrapper.action
    def action(a):
        if int(a[0]) != -1:
            return translate(a)
        result = decode(np.array([1, 0, 0, 4, 4, 4, 4]), spec)
        result['action_type'] = 0
        return result
    wrapper.action = action
    rows = pq.read_table('/scratch/kp0374/neurogym-agent/eval_zmaxleft_v1.parquet').to_pylist()[::8][:24]
    if args.cases_file:
        rows = json.load(open(args.cases_file))
    records = []
    completed = []
    try:
        for row in rows[args.offset:args.offset + args.limit]:
            idx = int(row['episode'] if args.cases_file else row['idx'])
            torch.manual_seed(20000 + idx)
            state, info = (copy.deepcopy(row['state']), {}) if args.cases_file else state_from_row(row)
            obs, _ = source.reset(options={'state': state, 'task_info': info})
            seen, zmax = set(raw['source']['segments']), {}
            best_z = float(raw['source']['position'][2])
            sampled = np.asarray(row['action']) if args.cases_file else None
            step = int(row['step']) if args.cases_file else 0
            if args.cases_file:
                seen, zmax, best_z = set(row['seen']), dict(row['zmax']), float(row['best_z'])
                policy.act(obs)  # diagnostic candidate probabilities on the settled snapshot
            for step in range(0 if args.cases_file else 400):
                pre = raw['source']
                best_z = max(best_z, float(pre['position'][2]))
                seen.update(pre['segments'])
                for sid, zm in zip(pre['segments'], pre.get('segment_zmax', ())):
                    if float(zm) > 0:
                        zmax[str(sid)] = float(zm)
                a = policy.act(obs)
                if step >= 50 and int(a[0]) == 3:
                    eligible = True
                    if args.only_3d_misses:
                        x, y = cell_to_pixel(int(a[2]), spec)
                        eligible = int(a[2]) % 32 >= 16 and source.unwrapped._renderer._segment_under_3d(x, y) is None
                    if eligible:
                        sampled = a.copy()
                        break
                obs, _, term, trunc, _ = source.step(a)
                if term or trunc:
                    break
            if sampled is None:
                records.append({'idx': idx, 'skipped': 'no late double-click decision'})
                completed.append(idx)
                with open(args.output, 'w') as f:
                    json.dump({'fresh_mode': True, 'completed_indices': completed, 'records': records}, f)
                continue
            snapshot = copy.deepcopy(source.unwrapped._renderer._state)
            probabilities = policy.grid_probabilities.copy()
            candidates = {'sampled': sampled, 'noop': np.array([-1, 0, 0, 4, 4, 4, 4])}
            candidate_targets = {'noop': None}
            if args.only_3d_misses:
                candidates['neutral_zoom'] = np.array([2, 0, 0, 4, 4, 4, 4])
                candidates['small_rotation'] = np.array([1, 0, 0, 5, 4, 4, 4])
            r = source.unwrapped._renderer
            sx, sy = cell_to_pixel(int(sampled[2]), spec)
            sampled_pane = '2d' if sampled[2] % 32 < 16 else '3d'
            sampled_rid = (r._segment_under_2d(sx, sy) if sampled_pane == '2d'
                           else r._segment_under_3d(sx, sy))
            settled_sampled_is_miss = sampled_rid is None if sampled_pane == '3d' else None
            selected_ids = {str(s).removeprefix('!') for s in snapshot['segments']}
            seen_ids = {str(s).removeprefix('!') for s in seen}
            visible_ids = set(S.visible_segments(snapshot['segments']))
            sampled_id = None if sampled_rid is None else str(sampled_rid)
            candidate_targets['sampled'] = {
                'pane': sampled_pane, 'cell': int(sampled[2]),
                'segment_id': sampled_id,
                'target_status': ('empty' if sampled_id is None else
                                  'selected' if sampled_id in selected_ids else
                                  'reselect' if sampled_id in seen_ids else 'new')}
            # Candidate selection uses privileged hit-testing for diagnosis.
            # "valid" means a new, never-selected target. Previously seen but
            # currently deselected targets are reported separately as reselections.
            grid_hit_distribution = {}
            category_ids = {}
            for pane in ['2d', '3d']:
                category_ids[pane] = {name: set() for name in
                                      ['empty', 'selected_visible', 'selected_hidden', 'reselect', 'new']}
                counts = {name: {'cells': 0, 'probability_mass': 0.0} for name in category_ids[pane]}
                cells = [int(c) for c in np.argsort(probabilities)[::-1]
                         if (int(c) % 32 < 16) == (pane == '2d')]
                for cell in cells:
                    x, y = cell_to_pixel(cell, spec)
                    rid = r._segment_under_2d(x, y) if pane == '2d' else r._segment_under_3d(x, y)
                    rid = None if rid is None else str(rid)
                    if rid is None:
                        status = 'empty'
                    elif rid in visible_ids:
                        status = 'selected_visible'
                    elif rid in selected_ids:
                        status = 'selected_hidden'
                    elif rid in seen_ids:
                        status = 'reselect'
                    else:
                        status = 'new'
                    counts[status]['cells'] += 1
                    counts[status]['probability_mass'] += float(probabilities[cell])
                    if rid is not None:
                        category_ids[pane][status].add(rid)
                    if pane == '3d' and status == 'selected_visible' and 'hit_3d' not in candidates:
                        candidate = sampled.copy()
                        candidate[2] = cell
                        candidates['hit_3d'] = candidate
                        candidate_targets['hit_3d'] = {
                            'pane': pane, 'cell': cell, 'segment_id': rid,
                            'target_status': status}
                    if status in ['new', 'reselect']:
                        label = f'valid_{pane}' if status == 'new' else f'reselect_{pane}'
                        if label not in candidates:
                            candidate = sampled.copy()
                            candidate[2] = cell
                            candidates[label] = candidate
                            candidate_targets[label] = {
                                'pane': pane, 'cell': cell, 'segment_id': rid,
                                'target_status': status}
                grid_hit_distribution[pane] = {
                    name: {**counts[name], 'distinct_segments': len(category_ids[pane][name])}
                    for name in counts}
            for repeat in range(args.repeats):
                seed = 30000 + idx * 10 + repeat
                for label, first in candidates.items():
                    torch.manual_seed(seed)
                    ob, _ = branch.reset(options={'state': copy.deepcopy(snapshot), 'task_info': info})
                    before = copy.deepcopy(raw['branch'])
                    reward_fn = make_zfree_reward_factory(rc)({})
                    history = {**before, 'position': [*before['position'][:2], best_z],
                               'segments': tuple(seen),
                               'segment_zmax': tuple(zmax.get(s, 0.0) for s in seen)}
                    reward_fn(history, None, history, False)
                    reward_fn(before, None, before, False)
                    peak, discounted, total = best_z, 0.0, 0.0
                    first_reward, changed = None, None
                    selection_delta, visibility_delta = None, None
                    for t in range(args.horizon + 1):
                        av = first if t == 0 else policy.act(ob)
                        ra = action(av)
                        prev = copy.deepcopy(raw['branch'])
                        ob, _, term, trunc, _ = branch.step(av)
                        after = raw['branch']
                        reward = reward_fn(after, ra, prev, False)
                        if t == 0:
                            first_reward = reward
                            changed = set(after['segments']) != set(prev['segments'])
                            before_selected = {str(s).removeprefix('!') for s in prev['segments']}
                            after_selected = {str(s).removeprefix('!') for s in after['segments']}
                            before_visible = set(S.visible_segments(prev['segments']))
                            after_visible = set(S.visible_segments(after['segments']))
                            selection_delta = {
                                'added': sorted(after_selected - before_selected),
                                'removed': sorted(before_selected - after_selected)}
                            visibility_delta = {
                                'shown': sorted(after_visible - before_visible),
                                'hidden': sorted(before_visible - after_visible)}
                        total += reward
                        discounted += (0.99 ** t) * reward
                        peak = max(peak, float(after['position'][2]))
                        if term or trunc:
                            break
                    records.append({'idx': idx, 'source_step': step, 'repeat': repeat,
                                    'branch': label, 'sampled_cell': int(sampled[2]),
                                    'sampled_pane': '2d' if sampled[2] % 32 < 16 else '3d',
                                    'candidate_target': candidate_targets.get(label),
                                    'grid_hit_distribution': grid_hit_distribution,
                                    'settled_sampled_is_miss': settled_sampled_is_miss,
                                    'source_standardized_advantage': row.get('standardized_advantage'),
                                    'first_reward': first_reward, 'selection_changed': changed,
                                    'selection_delta': selection_delta,
                                    'visibility_delta': visibility_delta,
                                    'discounted_return': discounted, 'return': total,
                                    'new_peak_dz': peak - best_z, 'steps': t + 1})
                    with open(args.output, 'w') as f:
                        json.dump({'fresh_mode': True, 'records': records}, f)
                    print(json.dumps(records[-1]), flush=True)
            completed.append(idx)
            with open(args.output, 'w') as f:
                json.dump({'fresh_mode': True, 'completed_indices': completed, 'records': records}, f)
    finally:
        source.close()
        branch.close()


if __name__ == '__main__':
    main()
