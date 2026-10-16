"""Capture GAE inputs and episode-cut metadata before SGD minibatching."""
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing


class BoundaryInfoWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        # This wrapper sits outside TimeLimit, stagger and renderer recovery.
        info['boundary_env_done'] = bool(terminated or truncated)
        return obs, reward, terminated, truncated, info


def chunk_advantages(rewards, values, terminated, gamma, lambda_):
    """Independent GAE recursion over real transitions plus one bootstrap value."""
    rewards, values = np.asarray(rewards), np.asarray(values)
    assert len(values) == len(rewards) + 1
    result = np.zeros(len(rewards), dtype=np.float64)
    running = 0.
    for t in reversed(range(len(rewards))):
        nonterminal = 1. - float(terminated[t])
        delta = rewards[t] + gamma * nonterminal * values[t + 1] - values[t]
        running = delta + gamma * lambda_ * nonterminal * running
        result[t] = running
    return result


class CaptureBoundaryGAE(ConnectorV2):
    def __init__(self, directory, gamma, lambda_):
        super().__init__()
        self.directory, self.gamma, self.lambda_ = Path(directory), gamma, lambda_
        self.calls = 0

    def __call__(self, *, rl_module, batch, **kwargs):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.calls += 1
        for mid, mb in batch.items():
            module = rl_module[mid]
            assert not module.is_stateful(), 'Boundary diagnostic expects feed-forward policy'
            with torch.no_grad():
                values = module.compute_values(mb).detach().cpu().numpy()
            def array(key):
                return mb[key].detach().cpu().numpy()
            labels, meta = array('click_credit'), array('boundary_meta')
            rewards, mask = array(Columns.REWARDS), array(Columns.LOSS_MASK).astype(bool)
            terminated, truncated = array(Columns.TERMINATEDS), array(Columns.TRUNCATEDS)
            targets = array(Postprocessing.VALUE_TARGETS)
            raw = targets - values
            start = 0
            errors = []
            for end in np.flatnonzero(~mask):
                n = end - start
                assert np.all(mask[start:end]) and int(meta[start, 2]) == n
                independent = chunk_advantages(rewards[start:end], values[start:end + 1],
                                               terminated[start:end], self.gamma, self.lambda_)
                errors.append(float(np.max(np.abs(independent - raw[start:end]))) if n else 0.)
                start = end + 1
            assert start == len(mask)
            assert max(errors, default=0.) < 2e-3, 'Installed GAE disagrees with independent chunk recursion'
            columns = ['verb_label', 'pane3d', 'selection_changed', 'added', 'removed',
                       'tiles_pending', 'meshes_pending', 'legacy_episode', 'step',
                       'episode_hi', 'episode_lo', 'chunk_length', 'chunk_start_step',
                       'env_end', 'full_episode', 'steps_to_chunk_end',
                       'reward', 'value', 'value_target', 'standardized_advantage',
                       'terminated', 'truncated', 'valid', 'behavior_p_double', 'behavior_p_3d_given_double']
            behavior = module.get_exploration_action_dist_cls().from_logits(mb[Columns.ACTION_DIST_INPUTS])
            pd = torch.softmax(behavior._cats[0].logits, -1)[:, 3].detach().cpu().numpy()
            grid = torch.softmax(behavior._cats[2].logits, -1).detach().cpu().numpy()
            p3 = grid[:, np.arange(grid.shape[1]) % 32 >= 16].sum(-1)
            matrix = np.column_stack([labels, meta, rewards, values, targets,
                                      array(Postprocessing.ADVANTAGES), terminated, truncated, mask, pd, p3])
            with (self.directory / f'boundary-{os.getpid()}.jsonl').open('a') as stream:
                stream.write(json.dumps({'batch': self.calls, 'module': mid, 'gamma': self.gamma,
                    'lambda': self.lambda_, 'max_gae_error': max(errors, default=0.),
                    'columns': columns, 'rows': matrix.tolist()}) + '\n')
        return batch
