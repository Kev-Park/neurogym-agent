"""Evaluate a policy on the frozen eval d0.

For each (root_id, node_index) pair: reset the env to the specified state,
run the policy until termination (success), truncation (TimeLimit or resilient),
or max_steps. Aggregate:
  - Overall success rate (fraction of pairs where termination fired = z within
    tolerance of z_max).
  - Per-quartile success rate, quartile boundaries computed over the d0 file's
    `length_nm` column (matches the distribution the pairs were sampled from).

Policy sources:
  --checkpoint <path>: load an RLlib checkpoint and use compute_single_action.
  --random-policy    : sample env.action_space (for pipeline sanity checks).

Deterministic-per-pair orientation: seeded with `--orientation-seed-base + pair_idx`,
so re-running with the same seed base gives identical initial states.

Run on a vulkan-capable GPU node via SLURM (needs a real browser).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

import duckdb
import numpy as np
import pyarrow.parquet as pq

# Ray >=2.43 auto-ships the CWD as a runtime_env; disable for the same reason
# ppo_smoke does (see scripts/ppo_smoke.py).
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


# ============================================================================
# State assembly for a specific (root_id, node_index) pair
# ============================================================================


def _random_quaternion(rng: np.random.Generator) -> list[float]:
    """Uniform random unit quaternion (Shoemake)."""
    u1, u2, u3 = (float(x) for x in rng.random(3))
    return [
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    ]


class StateBuilder:
    """Assemble (NglState, task_info) for eval pairs from skeleton data."""

    def __init__(self, skeleton_parquet: str,
                 projection_scale: float = 14000.0,
                 cross_section_scale: float = 2.0):
        self.projection_scale = projection_scale
        self.cross_section_scale = cross_section_scale
        self._con = duckdb.connect()
        esc = skeleton_parquet.replace("'", "''")
        self._con.execute(
            f"CREATE VIEW skel AS SELECT * FROM read_parquet('{esc}')"
        )

    def build(self, root_id: str, node_index: int, orientation_seed: int
              ) -> tuple[dict, dict]:
        res = self._con.execute(
            "SELECT x, y, z FROM skel WHERE CAST(root_id AS VARCHAR) = ?",
            [root_id],
        ).fetchnumpy()
        n = len(res["x"])
        if n == 0:
            raise ValueError(f"root_id {root_id!r} not found in skeleton")
        if node_index >= n:
            raise ValueError(f"node_index {node_index} >= n_nodes {n} for {root_id}")
        x, y, z = float(res["x"][node_index]), float(res["y"][node_index]), float(res["z"][node_index])
        z_max = float(res["z"].max())

        rng = np.random.default_rng(orientation_seed)
        state = {
            "position": [x, y, z],
            "projectionOrientation": _random_quaternion(rng),
            "projectionScale": self.projection_scale,
            "crossSectionScale": self.cross_section_scale,
            "segments": [root_id],
        }
        task_info = {"segment_id": root_id, "z_max": z_max}
        return state, task_info


# ============================================================================
# Policy adapters
# ============================================================================


class RandomPolicy:
    """Uniform-random policy over env.action_space. For pipeline sanity."""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()


class CheckpointPolicy:
    """RLlib checkpoint-restored policy. Deterministic (explore=False)."""
    def __init__(self, checkpoint_path: str):
        # Lazy import so --random-policy mode doesn't need Ray installed.
        from ray.rllib.algorithms.algorithm import Algorithm
        self.algo = Algorithm.from_checkpoint(checkpoint_path)

    def act(self, obs):
        return self.algo.compute_single_action(obs, explore=False)


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo_zmax_navigate.yaml")
    ap.add_argument("--eval-d0", required=True, help="Parquet with pair_idx, root_id, node_index, length_nm.")
    ap.add_argument("--skeleton", required=True, help="Skeleton parquet for state assembly.")
    ap.add_argument("--max-steps", type=int, default=300,
                    help="Per-episode step cap. TimeLimit still applies via config.")
    ap.add_argument("--orientation-seed-base", type=int, default=1000,
                    help="Seed = base + pair_idx for the initial quaternion.")
    ap.add_argument("--output", default="eval_results.json",
                    help="Where to write the JSON results.")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--checkpoint", help="RLlib checkpoint dir.")
    grp.add_argument("--random-policy", action="store_true",
                     help="Sample env.action_space each step (for infra tests).")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only run this many pairs (for smoke tests).")
    args = ap.parse_args()

    # Build env same shape as training.
    from ngllib_agent.env_build import build_env, load_config
    cfg = load_config(args.config)
    env = build_env(cfg)

    # Load eval pairs.
    d0_tbl = pq.read_table(args.eval_d0)
    pairs = d0_tbl.to_pylist()
    if args.limit > 0:
        pairs = pairs[:args.limit]
    print(f"[eval] {len(pairs)} pairs, config={args.config}", flush=True)

    # Policy.
    if args.random_policy:
        policy = RandomPolicy(env.action_space)
        print("[eval] policy: RandomPolicy (uniform action_space sample)", flush=True)
    else:
        print(f"[eval] policy: loading checkpoint {args.checkpoint}", flush=True)
        policy = CheckpointPolicy(args.checkpoint)
        print("[eval] policy: checkpoint loaded", flush=True)

    builder = StateBuilder(args.skeleton)

    results: list[dict[str, Any]] = []
    t_start = time.monotonic()
    for i, pair in enumerate(pairs):
        pair_idx = int(pair["pair_idx"])
        root_id = str(pair["root_id"])
        node_index = int(pair["node_index"])
        length_nm = int(pair["length_nm"])

        seed = args.orientation_seed_base + pair_idx
        state, task_info = builder.build(root_id, node_index, seed)

        obs, info = env.reset(options={"state": state, "task_info": task_info})
        terminated = False
        truncated = False
        steps_taken = 0
        for step in range(args.max_steps):
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            steps_taken = step + 1
            if terminated or truncated:
                break

        results.append({
            "pair_idx": pair_idx,
            "root_id": root_id,
            "node_index": node_index,
            "length_nm": length_nm,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "steps": steps_taken,
        })
        rate_so_far = sum(1 for r in results if r["terminated"]) / len(results)
        elapsed = time.monotonic() - t_start
        print(
            f"[eval] pair {i+1}/{len(pairs)} "
            f"root_id={root_id} steps={steps_taken} "
            f"term={terminated} trunc={truncated} "
            f"avg_success={rate_so_far:.1%} elapsed={elapsed:.0f}s",
            flush=True,
        )

    env.close()

    # Aggregate.
    n = len(results)
    n_success = sum(1 for r in results if r["terminated"])
    overall = n_success / n if n else 0.0

    # Per-quartile — quartiles computed from THIS d0 file's length distribution,
    # so the buckets are stable across runs even if predicate/n_pairs change.
    lengths = np.asarray([r["length_nm"] for r in results])
    q1, q2, q3 = (float(x) for x in np.quantile(lengths, [0.25, 0.5, 0.75]))
    buckets = [
        ("q1  (min - q25)", -np.inf, q1),
        ("q2  (q25 - q50)", q1, q2),
        ("q3  (q50 - q75)", q2, q3),
        ("q4  (q75 - max)", q3, np.inf),
    ]
    per_quartile = []
    for label, lo, hi in buckets:
        bucket = [r for r in results if lo <= r["length_nm"] < hi]
        n_b = len(bucket)
        s_b = sum(1 for r in bucket if r["terminated"])
        rate = s_b / n_b if n_b else 0.0
        per_quartile.append({
            "label": label,
            "lo": None if math.isinf(lo) else lo,
            "hi": None if math.isinf(hi) else hi,
            "n": n_b,
            "n_success": s_b,
            "success_rate": rate,
        })

    summary = {
        "n_pairs": n,
        "overall_success_rate": overall,
        "quartiles": per_quartile,
        "policy": "random" if args.random_policy else f"checkpoint:{args.checkpoint}",
        "config": args.config,
        "eval_d0": args.eval_d0,
        "elapsed_s": round(time.monotonic() - t_start, 1),
    }

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_pair": results}, f, indent=2)

    print("\n============ eval summary ============", flush=True)
    print(f"n_pairs             : {n}", flush=True)
    print(f"overall success rate: {overall:.2%}", flush=True)
    for q in per_quartile:
        print(
            f"  {q['label']:20s} "
            f"n={q['n']:3d} success={q['n_success']:3d} rate={q['success_rate']:.2%}",
            flush=True,
        )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
