"""Widen a shared-click-head checkpoint into a split-click-head one.

The split-head policy (action.split_click_heads) differs from v7 only in the
per-patch click scorer `_cell_score`: 1 output channel -> 2 (right-click,
double-click). This copies v7's channel into BOTH new channels -- the warm
start begins with identical right/double-click distributions, exactly v7's
behaviour -- for the module weights wherever they appear in the algorithm state
(learner + env runners) and for the Adam moments of those two tensors, which
are located by parameter NAME (the optimizer keys state by parameter position,
and the split adds no parameters, so every other position is unchanged).

    uv run --no-sync python native/convert_split_click_ckpt.py \
        --config configs/native_zmaxleft_v7f.yaml \
        --src /scratch/kp0374/checkpoints/zmaxleft-v7/ckpt_000450.pkl \
        --dst /scratch/kp0374/checkpoints/zmaxleft-v7f/ckpt_000450.pkl
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import numpy as np
from gymnasium import spaces

from ngllib_agent.distributed.checkpoint import atomic_pickle, load_checkpoint
from ngllib_agent.env_build import action_spec_from_config, load_config
from ngllib_agent.policies import HierarchicalPPOModule

KEYS = ("_cell_score.weight", "_cell_score.bias")


def widen(arr, n: int):
    a = np.asarray(arr)
    if a.shape[0] == n:
        return arr
    if a.shape[0] != 1:
        raise ValueError(f"expected a 1-channel tensor to widen, got {a.shape}")
    out = np.repeat(a, n, axis=0)
    return out.astype(a.dtype)


def widen_module_states(node, n: int, path: str = "") -> int:
    """Widen every `_cell_score.*` array found anywhere in the nested state."""
    hits = 0
    if isinstance(node, dict):
        for k in list(node.keys()):
            v = node[k]
            if isinstance(k, str) and k.endswith(KEYS) and hasattr(v, "shape"):
                node[k] = widen(v, n)
                print(f"[convert] {path}/{k}: {tuple(np.shape(v))} -> {tuple(np.shape(node[k]))}")
                hits += 1
            else:
                hits += widen_module_states(v, n, f"{path}/{k}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the SPLIT-head training config")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec = action_spec_from_config(cfg["action"])
    if not spec.split_click_heads:
        raise SystemExit("--config must set action.split_click_heads: true")
    panes = 2 if cfg["obs"].get("use_left_pane", True) else 1
    obs_space = spaces.Dict({
        "pos_state": spaces.Box(-np.inf, np.inf, (8,), np.float32),
        "patch_features": spaces.Box(-np.inf, np.inf, (panes, 16, 16, 384), np.float16),
    })
    module = HierarchicalPPOModule(observation_space=obs_space,
                                   action_space=spaces.MultiDiscrete(spec.nvec()),
                                   model_config=cfg["model"])
    names = [n for n, _ in module.named_parameters()]
    n_heads = module._n_click_heads
    print(f"[convert] split module: {len(names)} parameter tensors, {n_heads} click heads")

    state = load_checkpoint(args.src)
    hits = widen_module_states(state, n_heads)
    if hits == 0:
        raise SystemExit("no _cell_score tensors found in the source checkpoint")

    opt = state["learner_group"]["learner"]["optimizer"]
    for oname, ost in opt.items():
        per_param = ost["state"]["state"]
        for key in KEYS:
            idx = names.index(key)
            entry = per_param.get(idx)
            if entry is None:
                continue
            for m in ("exp_avg", "exp_avg_sq"):
                before = tuple(np.shape(entry[m]))
                entry[m] = widen(entry[m], n_heads)
                print(f"[convert] optimizer {oname} param {idx} ({key}) {m}: "
                      f"{before} -> {tuple(np.shape(entry[m]))}")

    # Sanity: the widened module state must load into the split module.
    import torch

    mod_state = state["learner_group"]["learner"]["rl_module"]["default_policy"]
    module.set_state({k: v for k, v in mod_state.items()})
    w = module._cell_score.weight.detach()
    assert torch.equal(w[0], w[1]), "both click heads must start identical"

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    atomic_pickle(state, args.dst)
    print(f"[convert] wrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
