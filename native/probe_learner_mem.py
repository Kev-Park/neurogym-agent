"""Peak learner GPU memory for the spatial (token) policy at a real batch size.

Builds the v7 module, puts a fake train batch of the real obs shape on the GPU
(fp16 tokens, as RLlib would), runs compute_values over the whole batch (what
the GAE connector does) and one minibatch forward+backward, and prints
torch.cuda.max_memory_allocated after each phase. Needs one GPU.

    uv run --no-sync python native/probe_learner_mem.py configs/native_zmaxleft_v7.yaml 16000 512
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import numpy as np
import torch
import yaml
from gymnasium import spaces
from ray.rllib.core.columns import Columns

from ngllib_agent.policies import HierarchicalPPOModule
from ngllib_agent.wrappers.action import ActionSpec


def main() -> int:
    cfg_path, batch, mb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    cfg = yaml.safe_load(open(cfg_path))
    ac = cfg["action"]
    panes = 2 if cfg["obs"].get("use_left_pane", True) else 1
    spec = ActionSpec(verbs=ac["verbs"], grid_rows=ac["grid_rows"], grid_cols=ac["grid_cols"],
                      rotation_bins_per_axis=ac["rotation_bins_per_axis"],
                      zoom_bins=ac["zoom_bins"])
    obs_space = spaces.Dict({
        "pos_state": spaces.Box(-np.inf, np.inf, (8,), np.float32),
        "patch_features": spaces.Box(-np.inf, np.inf, (panes, 16, 16, 384), np.float16),
    })
    m = HierarchicalPPOModule(observation_space=obs_space,
                              action_space=spaces.MultiDiscrete(spec.nvec()),
                              model_config=cfg["model"]).cuda()
    gb = lambda: torch.cuda.max_memory_allocated() / 2**30
    full = {Columns.OBS: {
        "pos_state": torch.randn(batch, 8, device="cuda"),
        "patch_features": torch.randn(batch, panes, 16, 16, 384, device="cuda").half(),
    }}
    print(f"batch on GPU: peak {gb():.2f} GB", flush=True)
    with torch.no_grad():
        v = m.compute_values(full)
    print(f"compute_values over {batch}: peak {gb():.2f} GB  (values {tuple(v.shape)})", flush=True)
    sub = {Columns.OBS: {k: t[:mb] for k, t in full[Columns.OBS].items()}}
    out = m.forward_train(sub)
    loss = out[Columns.ACTION_DIST_INPUTS].float().mean() + m.compute_values(
        sub, embeddings=out[Columns.EMBEDDINGS]).mean()
    loss.backward()
    print(f"minibatch {mb} fwd+bwd: peak {gb():.2f} GB", flush=True)
    print("PROBE-OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
