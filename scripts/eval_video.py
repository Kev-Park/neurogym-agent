"""Record rollout videos of a checkpoint policy on frozen-d0 pairs, by quartile.

For each length quartile of the d0 pool: roll episodes (stochastic sampling,
the decided eval protocol — REFINEMENT R11) in pool order until one SUCCESS
and one FAILURE episode are captured (or --attempts-per-quartile exhausted),
recording the raw two-pane frame each step. Encode each kept episode to MP4.

Frame tap: the raw `obs["image"]` exists below DinoObservationWrapper (the
policy consumes features, not pixels), so we wrap that instance's
`observation` method. Instance-local patch on purpose — threading a
video-only hook through build_env would add API surface for one viz script.

Reproducibility: orientation seed = --orientation-seed-base + pair_idx (same
scheme as eval_d0.py); torch is re-seeded per pair with the same value, so a
pair's rollout is repeatable in isolation.

Run on a vulkan GPU node via SLURM (r_eval_video.slurm).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_d0 import StateBuilder, StatePklPolicy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo_zmax_navigate.yaml")
    ap.add_argument("--eval-d0", required=True)
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--state-pkl", required=True)
    ap.add_argument("--out-dir", default="eval_videos")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--orientation-seed-base", type=int, default=1000)
    ap.add_argument("--attempts-per-quartile", type=int, default=6)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    import imageio.v2 as imageio
    import torch

    from ngllib_agent.env_build import build_env, load_config
    from ngllib_agent.wrappers import DinoObservationWrapper

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = "dino"
    env = build_env(cfg)

    # Locate the Dino wrapper and tap raw frames as observations flow through.
    dino_w = env
    while not isinstance(dino_w, DinoObservationWrapper):
        dino_w = dino_w.env
    frames: list[np.ndarray] = []
    orig_observation = dino_w.observation

    def tapped(obs):
        frames.append(np.asarray(obs["image"], dtype=np.uint8).copy())
        return orig_observation(obs)

    dino_w.observation = tapped

    policy = StatePklPolicy(args.state_pkl, env, cfg.get("model", {}),
                            stochastic=True)
    builder = StateBuilder(args.skeleton)

    pairs = pq.read_table(args.eval_d0).to_pylist()
    lengths = np.asarray([p["length_nm"] for p in pairs])
    q1, q2, q3 = np.quantile(lengths, [0.25, 0.5, 0.75])
    buckets = [("q1", -np.inf, q1), ("q2", q1, q2),
               ("q3", q2, q3), ("q4", q3, np.inf)]

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    for label, lo, hi in buckets:
        bucket_pairs = [p for p in pairs if lo <= p["length_nm"] < hi]
        got = {"success": False, "failure": False}
        for attempt, pair in enumerate(bucket_pairs[: args.attempts_per_quartile]):
            if all(got.values()):
                break
            pair_idx = int(pair["pair_idx"])
            seed = args.orientation_seed_base + pair_idx
            torch.manual_seed(seed)
            state, task_info = builder.build(
                str(pair["root_id"]), int(pair["node_index"]), seed)

            frames.clear()
            obs, _ = env.reset(options={"state": state, "task_info": task_info})
            terminated = truncated = False
            ep_return, steps = 0.0, 0
            for _ in range(args.max_steps):
                obs, reward, terminated, truncated, _ = env.step(policy.act(obs))
                ep_return += float(reward)
                steps += 1
                if terminated or truncated:
                    break

            outcome = "success" if terminated else "failure"
            print(f"[video] {label} pair {pair_idx} ({pair['length_nm']} nm): "
                  f"{outcome} steps={steps} return={ep_return:.3f} "
                  f"frames={len(frames)}", flush=True)
            if got[outcome]:
                continue  # already have a video for this outcome
            got[outcome] = True
            path = os.path.join(
                args.out_dir, f"{label}_{outcome}_pair{pair_idx}_{steps}steps.mp4")
            imageio.mimwrite(path, frames, fps=args.fps,
                             codec="libx264", quality=8)
            manifest.append({
                "quartile": label, "outcome": outcome, "pair_idx": pair_idx,
                "root_id": str(pair["root_id"]),
                "length_nm": int(pair["length_nm"]),
                "steps": steps, "episode_return": round(ep_return, 4),
                "file": os.path.basename(path),
            })
        missing = [k for k, v in got.items() if not v]
        if missing:
            print(f"[video] {label}: no {'/'.join(missing)} episode within "
                  f"{args.attempts_per_quartile} attempts", flush=True)

    env.close()
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[video] wrote {len(manifest)} videos -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
