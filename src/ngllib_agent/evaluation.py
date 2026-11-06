"""Shared policy loading and reproducible episode evaluation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np


def checkpoint_config(checkpoint, explicit_config=None):
    """Use paired settings; legacy weights require an explicit config."""
    sidecar = Path(checkpoint).with_suffix(".json")
    paired = json.loads(sidecar.read_text()).get("config") if sidecar.exists() else None
    if explicit_config is None:
        if paired is None:
            raise ValueError("Unpaired checkpoint requires --config")
        return copy.deepcopy(paired)
    if paired is not None:
        for section in ("obs", "action", "model"):
            if explicit_config.get(section) != paired.get(section):
                raise ValueError(f"Evaluation {section} differs from the checkpoint architecture")
    return copy.deepcopy(explicit_config)


def rollout(env, policy, start, max_steps, seed):
    """One frozen start; retain task-independent per-step outcomes."""
    obs, info = env.reset(seed=seed, options={"state": start["state"],
                                            "task_info": start["task_info"]})
    initial = info["json_state"]["position"]
    trace = []
    for _ in range(max_steps):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        trace.append(dict(action=np.asarray(action).tolist(), reward=float(reward),
                          position=list(info["json_state"]["position"]),
                          terminated=bool(terminated), truncated=bool(truncated)))
        if terminated or truncated:
            break
    return dict(id=str(start["id"]), initial_position=list(initial), trace=trace,
                episode_return=sum(step["reward"] for step in trace),
                steps=len(trace),
                z_gain_voxels=max([initial[2]] + [s["position"][2] for s in trace]) - initial[2])


def compare_results(left, right, allow_backend_difference=False):
    """Paired differences require the same starts and rollout protocol."""
    if not left.get("complete") or not right.get("complete"):
        raise ValueError("Cannot compare incomplete evaluations")
    protocols = [copy.deepcopy(result["protocol"]) for result in (left, right)]
    if allow_backend_difference:
        for protocol in protocols:
            protocol["env"].pop("backend", None)
    if protocols[0] != protocols[1]:
        raise ValueError("Evaluation protocols differ")
    a = {row["id"]: row for row in left["episodes"]}
    b = {row["id"]: row for row in right["episodes"]}
    if len(a) != len(left["episodes"]) or len(b) != len(right["episodes"]) or a.keys() != b.keys():
        raise ValueError("Evaluations must have unique, identical frozen-start IDs")
    return {key: {metric: b[key][metric] - a[key][metric]
                  for metric in ("episode_return", "z_gain_voxels")} for key in a}


class CheckpointPolicy:
    """RLlib checkpoint-restored policy. Deterministic (explore=False)."""
    def __init__(self, checkpoint_path: str):
        # Lazy import so --random-policy mode doesn't need Ray installed.
        from ray.rllib.algorithms.algorithm import Algorithm
        self.algo = Algorithm.from_checkpoint(checkpoint_path)

    def act(self, obs):
        return self.algo.compute_single_action(obs, explore=False)


class StatePklPolicy:
    """Policy from a `ngllib_agent.train` state pickle (`ckpt_*.pkl`).

    Rebuilds HierarchicalPPOModule with the env's spaces, loads the module
    slice of the pickled Algorithm state, and acts via forward_inference —
    per-head argmax by default, or sampled (training-matched) with
    stochastic=True. Argmax on the 1024-way click head can be degenerate
    for a policy that only ever acted by sampling; eval both when in doubt.
    """

    def __init__(self, pkl_path: str, env, model_config: dict,
                 stochastic: bool = False):
        import torch

        from ngllib_agent.distributed.checkpoint import load_checkpoint
        from ngllib_agent.policies import HierarchicalPPOModule

        self._torch = torch
        self._stochastic = stochastic
        # inference_only=False so the state (which includes the vf head) maps 1:1.
        self.module = HierarchicalPPOModule(
            observation_space=env.observation_space,
            action_space=env.action_space,
            model_config=model_config,
        )
        state = load_checkpoint(pkl_path)
        module_state = state["learner_group"]["learner"]["rl_module"]["default_policy"]
        self.module.set_state(module_state)
        self.module.eval()
        self.dist_cls = self.module.get_inference_action_dist_cls()

    def act(self, obs):
        torch = self._torch
        from ray.rllib.core.columns import Columns

        batch = {
            Columns.OBS: {
                k: torch.from_numpy(np.asarray(v, np.float32)).unsqueeze(0)
                for k, v in obs.items()
            }
        }
        with torch.no_grad():
            out = self.module.forward_inference(batch)
        dist = self.dist_cls.from_logits(out[Columns.ACTION_DIST_INPUTS])
        if not self._stochastic:
            dist = dist.to_deterministic()
        return dist.sample().squeeze(0).cpu().numpy()

