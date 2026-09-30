"""Summarize a run's learner curves from wandb: does the critic settle?

    .venv/bin/python scripts/wandb_critic_summary.py zmaxleft-v4 zmaxleft-v5b

Prints, per tenth of the run's iterations, the median of the learner scalars
train.py logs (vf_loss, vf_explained_var, mean_kl_loss, entropy, policy_loss)
plus the episode return, so a glance shows whether the value loss / explained
variance plateau or keep drifting. Login-node safe (network only).
"""

from __future__ import annotations

import argparse
import statistics

KEYS = [
    "learner/vf_loss",
    "learner/vf_explained_var",
    "learner/mean_kl_loss",
    "learner/entropy",
    "learner/policy_loss",
    "env_runners/episode_return_mean",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="wandb run display names")
    ap.add_argument("--project", default="neurogym-agent")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--list", action="store_true",
                    help="treat RUNS as name substrings; list matching runs "
                         "(name, id, state, last step) instead of summarizing")
    args = ap.parse_args()

    import wandb

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    if args.list:
        for run in api.runs(path):
            if any(s in run.name for s in args.runs):
                print(f"{run.name:<28} {run.id:<10} {run.state:<9} "
                      f"step={run.summary.get('_step')} "
                      f"iters={run.summary.get('training_iteration')}")
        return 0
    for name in args.runs:
        runs = list(api.runs(path, filters={"display_name": name}))
        if not runs:
            print(f"== {name}: NOT FOUND in {path}")
            continue
        # crashed relaunches share the display name; take the longest run
        run = max(runs, key=lambda r: r.summary.get("_step") or 0)
        rows = [r for r in run.scan_history(keys=["_step", *KEYS]) if r.get("_step") is not None]
        rows.sort(key=lambda r: r["_step"])
        if not rows:
            print(f"== {name}: no history")
            continue
        last = rows[-1]["_step"]
        print(f"== {name} ({run.id}): {len(rows)} iters, last _step={last}")
        short = [k.split("/")[-1] for k in KEYS]
        print(f"{'iters':>12} " + " ".join(f"{s:>16}" for s in short))
        n = len(rows)
        for b in range(args.bins):
            lo, hi = b * n // args.bins, (b + 1) * n // args.bins
            chunk = rows[lo:hi]
            if not chunk:
                continue
            cells = []
            for k in KEYS:
                vals = [float(r[k]) for r in chunk if r.get(k) is not None]
                cells.append(f"{statistics.median(vals):>16.4g}" if vals else f"{'-':>16}")
            print(f"{chunk[0]['_step']:>5}-{chunk[-1]['_step']:<6} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
