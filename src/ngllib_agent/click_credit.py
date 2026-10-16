"""Opt-in action outcomes and direct PPO logit-gradient attribution."""
import json
import os
import uuid
import copy
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing


class ClickOutcomeWrapper(gym.Wrapper):
    def __init__(self, env, directory):
        super().__init__(env)
        self._directory = Path(directory) / 'snapshots'
        self._directory.mkdir(parents=True, exist_ok=True)

    def _note_history(self, obs):
        self._seen.update(str(s) for s in obs['segments'])
        self._best_z = max(self._best_z, float(obs['position'][2]))
        for sid, zm in zip(obs['segments'], obs.get('segment_zmax', ())):
            if float(zm) > 0:
                self._zmax[str(sid)] = float(zm)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._previous = obs
        self._episode = uuid.uuid4().int % (2 ** 31)
        self._step = 0
        self._seen, self._zmax = set(), {}
        self._best_z = float(obs['position'][2])
        self._snapshot_taken = False
        self._note_history(obs)
        return obs, info

    def step(self, action):
        r = self.unwrapped._renderer
        pending = r._pending
        tiles = pending is not None and not all(f.done() for f in pending[1].values())
        meshes = any(not f.done() for _, f in r._mesh_futs.values())
        verb = int(action[0])
        pane = int(action[2]) % 32 >= 16 if verb == 3 else -1
        snapshot = None
        if verb == 3 and pane and self._step >= 50 and not self._snapshot_taken:
            snapshot = dict(episode=self._episode, step=self._step, state=copy.deepcopy(r._state),
                            action=[int(v) for v in action], seen=sorted(self._seen),
                            zmax=dict(self._zmax), best_z=self._best_z)
        obs, reward, term, trunc, info = self.env.step(action)
        before, after = set(self._previous['segments']), set(obs['segments'])
        if snapshot is not None and before == after:
            (self._directory / f'{self._episode}-{self._step}.json').write_text(json.dumps(snapshot))
            self._snapshot_taken = True
        info = dict(info)
        info['click_credit'] = np.asarray([
            verb, pane, before != after, len(after - before), len(before - after),
            tiles, meshes, self._episode, self._step,
        ], dtype=np.float64)
        self._step += 1
        self._previous = obs
        self._note_history(obs)
        return obs, reward, term, trunc, info


class AddClickOutcomes(ConnectorV2):
    def __init__(self, boundary=False):
        super().__init__()
        self.boundary = boundary

    def __call__(self, *, batch, episodes, **kwargs):
        for episode in self.single_agent_episode_iterator(episodes, agents_that_stepped_only=False):
            # Infos are observation-indexed: info[t+1] describes action[t]. PPO
            # has already added its masked artificial bootstrap timestep.
            infos = episode.get_infos(slice(1, len(episode) + 1))
            values = [info.get('click_credit', np.full(9, -1.)) for info in infos]
            self.add_n_batch_items(batch, 'click_credit', items_to_add=values,
                                   num_items=len(episode), single_agent_episode=episode)
            if self.boundary:
                n = len(episode) - 1
                assert n > 0
                first = infos[0]['click_credit']
                eid, initial_step = int(first[7]), int(first[8])
                ended = bool(infos[n - 1].get('boundary_env_done', False))
                metadata = [np.asarray([eid // 65536, eid % 65536, n, initial_step,
                    ended, ended and initial_step == 0, n - 1 - t], dtype=np.float32)
                    for t in range(n + 1)]
                self.add_n_batch_items(batch, 'boundary_meta', items_to_add=metadata,
                                       num_items=len(episode), single_agent_episode=episode)
        return batch


class ClickCreditLearner(PPOTorchLearner):
    def build(self):
        super().build()
        directory = self.module['default_policy'].unwrapped().model_config.get('boundary_directory')
        if directory:
            from .boundary_credit import CaptureBoundaryGAE
            self._learner_connector.append(CaptureBoundaryGAE(directory, self.config.gamma, self.config.lambda_))

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        loss = super().compute_loss_for_module(module_id=module_id, config=config,
                                               batch=batch, fwd_out=fwd_out)
        logits = fwd_out[Columns.ACTION_DIST_INPUTS]
        module = self.module[module_id].unwrapped()
        dist = module.get_train_action_dist_cls().from_logits(logits)
        previous = module.get_exploration_action_dist_cls().from_logits(batch[Columns.ACTION_DIST_INPUTS])
        actions = batch[Columns.ACTIONS]
        valid = batch.get(Columns.LOSS_MASK, torch.ones(len(actions), dtype=torch.bool, device=logits.device)).bool()
        labels = batch['click_credit']
        assert torch.all(labels[valid, 0].long() == actions[valid, 0].long()), 'Outcome/action alignment failed'
        advantages = batch[Postprocessing.ADVANTAGES]
        ratio = (dist.logp(actions) - batch[Columns.ACTION_LOGP]).exp()
        surrogate = torch.minimum(advantages * ratio,
                                  advantages * ratio.clamp(1 - config.clip_param, 1 + config.clip_param))
        entropy_coeff = self.entropy_coeff_schedulers_per_module[module_id].get_current_value()
        terms = {'policy': -surrogate, 'entropy': -entropy_coeff * dist.entropy()}
        if config.use_kl_loss:
            terms['kl'] = self.curr_kl_coeffs_per_module[module_id] * previous.kl(dist)
        else:
            terms['kl'] = logits.sum(-1) * 0
        nverb, nright, ndouble = dist._input_lens[:3]
        offset = nverb + nright
        chosen = offset + actions[:, 2].long()
        row = torch.arange(len(actions), device=logits.device)
        right_pane = torch.arange(ndouble, device=logits.device) % 32 >= 16
        values = [labels, actions[:, :1].float(), advantages[:, None], batch[Columns.REWARDS][:, None],
                  ratio[:, None], torch.softmax(logits[:, :nverb], -1)[:, 3:4],
                  torch.softmax(logits[:, offset:offset + ndouble], -1)[:, right_pane].sum(-1, keepdim=True),
                  torch.softmax(logits[:, offset:offset + ndouble], -1)[row, actions[:, 2].long(), None]]
        names = ['verb_label', 'pane3d', 'selection_changed', 'added', 'removed', 'tiles_pending',
                 'meshes_pending', 'episode', 'step', 'verb', 'standardized_advantage', 'reward',
                 'ratio', 'p_double', 'p_3d_given_double', 'p_chosen_double_cell']
        for name, term in terms.items():
            # Negative loss derivative is the direct descent direction. These
            # are per-row derivatives before the common valid-batch mean.
            grad, = torch.autograd.grad(term[valid].sum(), logits, retain_graph=True)
            values.extend([-grad[row, chosen, None],
                           -grad[:, offset:offset + ndouble][:, right_pane].sum(-1, keepdim=True),
                           -grad[:, 3:4]])
            names.extend([name + '_chosen_cell_direction', name + '_toward_3d_direction',
                          name + '_double_verb_direction'])
        data = torch.cat(values, -1)[valid].detach().cpu().double().numpy()
        directory = Path(module.model_config['click_credit_directory'])
        directory.mkdir(parents=True, exist_ok=True)
        self._click_credit_calls = getattr(self, '_click_credit_calls', 0) + 1
        with (directory / f'learner-{os.getpid()}.jsonl').open('a') as stream:
            stream.write(json.dumps({'call': self._click_credit_calls, 'columns': names,
                                     'entropy_coeff': float(entropy_coeff), 'rows': data.tolist()}) + '\n')
        return loss
