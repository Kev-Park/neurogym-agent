"""Per-head policy entropy of several checkpoints on the SAME states.

The logged training entropy is the scaled, summed bonus (verb + s*cell + rot +
zoom), so it cannot say how sharp the click distribution actually is. This
collects real states by rolling the FIRST checkpoint on holdout starts, then
evaluates every checkpoint's heads on those identical observations and prints,
per checkpoint:
  verb   entropy / ln(n_verb) and the mean verb probabilities
  cell   entropy / ln(n_cell), effective cells e^H, and the probability mass
         on the top-1 / top-10 cells (all unscaled -- the real distribution)
  rot, zoom  entropy / ln(bins)

    uv run --no-sync python scripts/probe_head_entropy.py \
        --config configs/native_zmaxleft_v7_eval.yaml \
        --holdout <parquet> --label v7=<pkl> --label v7d=<pkl> ...
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from eval_zmaxleft import StatePklPolicy, state_from_row  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--label", action="append", required=True,
                    help="name=path/to/ckpt.pkl; the first one generates the states")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from ray.rllib.core.columns import Columns

    from ngllib_agent.env_build import build_env, load_config

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = "dino"
    cfg.setdefault("env", {})["max_episode_steps"] = args.max_steps
    env = build_env(cfg)
    labels = [tuple(s.split("=", 1)) for s in args.label]
    policies = {name: StatePklPolicy(path, env, cfg.get("model", {}), stochastic=True)
                for name, path in labels}

    rows = pq.read_table(args.holdout).to_pylist()
    rng = np.random.default_rng(args.seed)
    picks = [rows[i] for i in rng.choice(len(rows), size=min(args.n, len(rows)), replace=False)]
    gen = policies[labels[0][0]]
    torch.manual_seed(args.seed)
    states = []
    for row in picks:
        state, info = state_from_row(row)
        obs, _ = env.reset(options={"state": state, "task_info": info})
        for t in range(args.max_steps):
            if t % args.stride == 0:
                states.append({k: np.asarray(v).copy() for k, v in obs.items()})
            obs, _, term, trunc, _ = env.step(gen.act(obs))
            if term or trunc:
                break
    print(f"[heads] {len(states)} states from {len(picks)} episodes of {labels[0][0]}", flush=True)

    for name, pol in policies.items():
        lens = list(pol.dist_cls._input_lens)
        n_verb, n_cell, r, _, _, n_zoom = lens
        acc = {"verb": [], "cell": [], "rot": [], "zoom": [], "top1": [], "top10": [], "p": []}
        for i in range(0, len(states), 64):
            chunk = states[i:i + 64]
            batch = {Columns.OBS: {k: torch.from_numpy(np.stack([s[k] for s in chunk]))
                                   for k in chunk[0]}}
            with torch.no_grad():
                logits = pol.module.forward_inference(batch)[Columns.ACTION_DIST_INPUTS].float()
            parts = torch.split(logits, lens, dim=-1)
            probs = [torch.softmax(p, -1) for p in parts]
            ent = [-(p * torch.log(p.clamp_min(1e-12))).sum(-1) for p in probs]
            acc["verb"].append((ent[0] / math.log(n_verb)).numpy())
            acc["cell"].append((ent[1] / math.log(n_cell)).numpy())
            acc["rot"].append(((ent[2] + ent[3] + ent[4]) / (3 * math.log(r))).numpy())
            acc["zoom"].append((ent[5] / math.log(n_zoom)).numpy())
            srt = torch.sort(probs[1], -1, descending=True).values
            acc["top1"].append(srt[:, 0].numpy())
            acc["top10"].append(srt[:, :10].sum(-1).numpy())
            acc["p"].append(probs[0].numpy())
        m = {k: np.concatenate(v).mean(0) for k, v in acc.items()}
        cell_h = m["cell"] * math.log(n_cell)
        print(f"[heads] {name:<6} verb H={m['verb']:.3f}  cell H={m['cell']:.3f} "
              f"(e^H={math.exp(cell_h):.0f} of {n_cell} cells, top1 {m['top1']:.3f}, top10 {m['top10']:.3f})  "
              f"rot H={m['rot']:.3f}  zoom H={m['zoom']:.3f}  "
              f"verb p=[{', '.join(f'{x:.3f}' for x in m['p'])}]", flush=True)
    print("HEADS-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
