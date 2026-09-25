"""Evaluate a policy on the frozen zmax-left holdout (paired delta-z protocol).

zmax-left has no per-start success predicate (the reachable max would need a
contact-graph oracle), so evaluation is DISTRIBUTIONAL and PAIRED: every
checkpoint runs the SAME 200 frozen start states (eval_zmaxleft_v1.parquet,
complete states: position, quaternion, projectionScale, crossSectionScale) and
the per-state best-so-far z gain dz@budget is the outcome. Comparisons between
checkpoints difference dz per state (scripts/compare_zmaxleft.py) — the unknown
per-start achievable max cancels in the pairing.

Per-state metrics (budgets post-hoc from the z series, default 250,500):
  dz@b            max z over the first b steps, minus z0
  beat_ceiling@b  1 if max z over first b steps > the START segment's own
                  skeleton z-max — geometric proof a HOP raised the climb
  diag            the env's [zmaxleft-ep] counters (new_segs, hop_climb,
                  showall_steps, verb usage), captured from the follow-up
                  reset's `zmaxleft_prev` info.

Usage (GPU node):
  uv run --no-sync python scripts/eval_zmaxleft.py \
      --holdout /scratch/kp0374/neurogym-agent/eval_zmaxleft_v1.parquet \
      --state-pkl /scratch/kp0374/checkpoints/<run>/ckpt_<it>.pkl \
      --output eval_zmaxleft_<run>_<it>.json
  (--random-policy for the null baseline; --repeats K for Tier-2 noise floors.)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def state_from_row(row: dict) -> tuple[dict, dict]:
    """Holdout row -> (NglState, task_info). The state is COMPLETE in the file;
    nothing is sampled here, so every eval sees byte-identical starts."""
    state = {
        "position": [float(row["x"]), float(row["y"]), float(row["z"])],
        "projectionOrientation": [float(row["qx"]), float(row["qy"]),
                                  float(row["qz"]), float(row["qw"])],
        "projectionScale": float(row["projection_scale"]),
        "crossSectionScale": float(row["cross_section_scale"]),
        "segments": [str(row["root_id"])],
    }
    task_info = {
        "segment_id": str(row["root_id"]),
        "z_max": float(row["seg_z_max"]),   # START segment's own extent —
        "z_min": float(row["seg_z_min"]),   # reporting only; z_free ignores it
    }
    return state, task_info


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()


class StatePklPolicy:
    """Policy from a train.py state pickle — mirrors eval_d0.StatePklPolicy."""

    def __init__(self, pkl_path: str, env, model_config: dict,
                 stochastic: bool = False):
        import torch

        from ngllib_agent.distributed.checkpoint import load_checkpoint
        from ngllib_agent.policies import HierarchicalPPOModule

        self._torch = torch
        self._stochastic = stochastic
        self.module = HierarchicalPPOModule(
            observation_space=env.observation_space,
            action_space=env.action_space,
            model_config=model_config,
        )
        state = load_checkpoint(pkl_path)
        module_state = state["learner_group"]["learner"]["rl_module"]["default_policy"]
        self.module.set_state(module_state)
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


class CheckpointPolicy:
    def __init__(self, checkpoint_path: str):
        from ray.rllib.algorithms.algorithm import Algorithm
        self.algo = Algorithm.from_checkpoint(checkpoint_path)

    def act(self, obs):
        return self.algo.compute_single_action(obs, explore=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/native_zmaxleft.yaml")
    ap.add_argument("--holdout", required=True,
                    help="eval_zmaxleft parquet (complete frozen start states).")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--report-budgets", default="250,500")
    ap.add_argument("--output", default="eval_zmaxleft_results.json")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--checkpoint", help="RLlib checkpoint dir.")
    grp.add_argument("--state-pkl", help="train.py state pickle (ckpt_*.pkl).")
    grp.add_argument("--random-policy", action="store_true",
                     help="Null baseline: sample env.action_space each step.")
    ap.add_argument("--stochastic", action="store_true",
                    help="Sample policy actions (training-matched); --state-pkl only.")
    ap.add_argument("--torch-seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1,
                    help="Rollouts per state (identical start; Tier-2 pairing).")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    budgets = sorted(int(b) for b in args.report_budgets.split(","))
    assert budgets[-1] <= args.max_steps

    from ngllib_agent.env_build import build_env, load_config
    cfg = load_config(args.config)
    cfg.setdefault("env", {})["max_episode_steps"] = args.max_steps
    env = build_env(cfg)

    class EpisodeTimeout(Exception):
        pass

    def _on_alarm(signum, frame):  # noqa: ARG001
        raise EpisodeTimeout()

    signal.signal(signal.SIGALRM, _on_alarm)

    rows = pq.read_table(args.holdout).to_pylist()
    if args.offset > 0:
        rows = rows[args.offset:]
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"[eval-zl] {len(rows)} states x {args.repeats} repeat(s), "
          f"budgets={budgets}, config={args.config}", flush=True)

    if args.random_policy:
        policy = RandomPolicy(env.action_space)
        print("[eval-zl] policy: RandomPolicy (null baseline)", flush=True)
    elif args.state_pkl:
        import torch
        torch.manual_seed(args.torch_seed)
        policy = StatePklPolicy(args.state_pkl, env, cfg.get("model", {}),
                                stochastic=args.stochastic)
        print(f"[eval-zl] policy: state pickle {args.state_pkl} "
              f"(stochastic={args.stochastic})", flush=True)
    else:
        policy = CheckpointPolicy(args.checkpoint)
        print(f"[eval-zl] policy: checkpoint {args.checkpoint}", flush=True)

    zscale = float((cfg.get("obs", {}).get("pos_state_scale") or [1e5] * 8)[2])

    def _z_of(obs) -> float:
        if isinstance(obs, dict):
            if "pos_state" in obs:
                return float(obs["pos_state"][2]) * zscale
            return float(np.asarray(obs["position"])[2])
        return float(np.asarray(obs)[2]) * zscale

    results: list[dict[str, Any]] = []
    t_start = time.monotonic()
    plan = [(row, r) for row in rows for r in range(args.repeats)]
    pending_diag_for: int | None = None  # index into results awaiting diag

    def _reset(state, task_info):
        nonlocal pending_diag_for
        obs, info = env.reset(options={"state": state, "task_info": task_info})
        # the diag wrapper reports the PREVIOUS episode on this reset
        if pending_diag_for is not None and "zmaxleft_prev" in info:
            results[pending_diag_for]["diag"] = info["zmaxleft_prev"]
        pending_diag_for = None
        return obs

    for i, (row, rep) in enumerate(plan):
        idx = int(row["idx"])
        state, task_info = state_from_row(row)
        if args.repeats > 1 and args.state_pkl:
            import torch
            torch.manual_seed(args.torch_seed + 7919 * rep + 31 * idx)

        z_series: list[float] = []
        wedged = False
        signal.alarm(900)
        try:
            obs = _reset(state, task_info)
            z_series.append(_z_of(obs))
            for _step in range(args.max_steps):
                action = policy.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                z_series.append(_z_of(obs))
                if terminated or truncated:
                    break
        except EpisodeTimeout:
            wedged = True
            print(f"[eval-zl] state {i+1}: WEDGED past 900s — recording, "
                  "rebuilding env", flush=True)
            signal.alarm(30)
            try:
                env.close()
            except Exception:
                pass
            finally:
                signal.alarm(0)
            env = build_env(cfg)
            pending_diag_for = None
        finally:
            signal.alarm(0)
        if not z_series:
            z_series = [float(state["position"][2])]

        zs = np.asarray(z_series)
        z0 = float(zs[0])
        rec: dict[str, Any] = {
            "idx": idx, "rep": rep, "root_id": str(row["root_id"]),
            "z0": round(z0, 1),
            "seg_z_max": round(float(row["seg_z_max"]), 1),
            "steps": len(zs) - 1, "wedged": wedged,
            "diag": None,  # filled by the NEXT reset's zmaxleft_prev
        }
        for b in budgets:
            peak = float(zs[: b + 1].max())
            rec[f"dz@{b}"] = round(peak - z0, 1)
            rec[f"beat_ceiling@{b}"] = int(peak > float(row["seg_z_max"]))
        results.append(rec)
        pending_diag_for = len(results) - 1

        if len(rows) <= 50 or (i + 1) % 10 == 0:
            with open(args.output, "w") as f:
                json.dump({"summary": {"partial": True, "n_done": len(results)},
                           "per_state": results}, f)
        b0 = budgets[-1]
        med = float(np.median([r[f"dz@{b0}"] for r in results]))
        print(f"[eval-zl] {i+1}/{len(plan)} idx={idx} rep={rep} "
              f"dz@{b0}={rec[f'dz@{b0}']} beat_ceiling={rec[f'beat_ceiling@{b0}']} "
              f"median_dz={med:.1f} elapsed={time.monotonic()-t_start:.0f}s",
              flush=True)

    # one throwaway reset so the LAST episode's diag is captured
    try:
        signal.alarm(300)
        _reset(*state_from_row(rows[0]))
    except Exception:
        pass
    finally:
        signal.alarm(0)

    summary: dict[str, Any] = {"n": len(results), "repeats": args.repeats,
                               "budgets": budgets, "partial": False}
    for b in budgets:
        dzs = np.asarray([r[f"dz@{b}"] for r in results], dtype=float)
        summary[f"dz@{b}"] = {
            "median": round(float(np.median(dzs)), 1),
            "q25": round(float(np.percentile(dzs, 25)), 1),
            "q75": round(float(np.percentile(dzs, 75)), 1),
            "mean": round(float(dzs.mean()), 1),
            "frac_positive": round(float((dzs > 0).mean()), 3),
            "frac_beat_ceiling": round(
                float(np.mean([r[f"beat_ceiling@{b}"] for r in results])), 3),
        }
    diags = [r["diag"] for r in results if r.get("diag")]
    if diags:
        summary["diag"] = {
            "mean_new_segs": round(float(np.mean([d["new_segs"] for d in diags])), 2),
            "frac_hop_climb": round(float(np.mean([d["hop_climb"] for d in diags])), 3),
            "mean_showall_steps": round(
                float(np.mean([d["showall_steps"] for d in diags])), 1),
            "n_diag": len(diags),
        }
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_state": results}, f)
    print(f"[eval-zl] SUMMARY {json.dumps(summary)}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
