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


def action_label(a, spec) -> str:
    """Compact human label for a MultiDiscrete action sample."""
    a_type, cell, dx, dy, dz, dzoom = (int(v) for v in a)
    if a_type == 0:
        return f"click r{cell // spec.grid_cols},c{cell % spec.grid_cols}"
    if a_type == 1:
        c = spec.rotation_bins_per_axis // 2
        return f"rotate x{dx - c:+d} y{dy - c:+d} z{dz - c:+d}"
    return f"zoom {(dzoom - spec.zoom_bins // 2) * spec.zoom_step:+.0f}"


def annotate_frames(frames, zs, z_max, z_tol, terminated, labels):
    """HUD per frame: step counter, z readout vs the target band, and the
    action that PRODUCED this frame (frame 0 is the reset — no action).
    Final frame gets a SUCCESS/TIMEOUT stamp and is held ~1.5s.
    """
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=16)
        font_big = ImageFont.load_default(size=28)
    except TypeError:  # Pillow < 10.1
        font = font_big = ImageFont.load_default()

    out = []
    n = len(frames)
    for i, (frame, z) in enumerate(zip(frames, zs)):
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, img.width, 44], fill=(0, 0, 0))
        d.text((8, 2),
               f"step {i}/{n - 1}   z={z:.1f}   target z_max={z_max:.1f} "
               f"+/-{z_tol:g}   dz={z - z_max:+.1f}",
               fill=(255, 255, 255), font=font)
        act = labels[i - 1] if i > 0 and i - 1 < len(labels) else "-- (reset)"
        d.text((8, 23), f"action: {act}", fill=(180, 220, 255), font=font)
        if i == n - 1:
            label = "SUCCESS" if terminated else "TIMEOUT (no success)"
            color = (0, 220, 0) if terminated else (255, 60, 60)
            d.text((8, 52), label, fill=color, font=font_big)
        out.append(np.asarray(img))
    out.extend([out[-1]] * 15)  # hold the outcome frame ~1.5s at 10 fps
    return out


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

    import gymnasium as gym
    import imageio.v2 as imageio
    import torch

    from ngllib_agent.env_build import action_spec_from_config, build_env, load_config

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = "dino"
    env = build_env(cfg)

    # Locate the DINO obs wrapper and tap raw frames as observations flow
    # through. DinoObservationWrapper is a factory returning a nested class
    # (lazy gym import), so match by shape: the ObservationWrapper whose INNER
    # env still exposes the raw "image" space.
    dino_w = env
    while True:
        inner = getattr(dino_w, "env", None)
        if inner is None:
            raise RuntimeError("no image->features wrapper found in env stack")
        inner_space = getattr(inner, "observation_space", None)
        if (hasattr(dino_w, "observation")
                and isinstance(inner_space, gym.spaces.Dict)
                and "image" in inner_space.spaces):
            break
        dino_w = inner
    frames: list[np.ndarray] = []
    zs: list[float] = []
    orig_observation = dino_w.observation

    def tapped(obs):
        frames.append(np.asarray(obs["image"], dtype=np.uint8).copy())
        zs.append(float(np.asarray(obs["position"])[2]))
        return orig_observation(obs)

    dino_w.observation = tapped

    policy = StatePklPolicy(args.state_pkl, env, cfg.get("model", {}),
                            stochastic=True)
    psr = cfg.get("env", {}).get("projection_scale_range")
    builder = StateBuilder(args.skeleton,
                           projection_scale_range=tuple(psr) if psr else None)

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
            zs.clear()
            obs, _ = env.reset(options={"state": state, "task_info": task_info})
            terminated = truncated = False
            ep_return, steps = 0.0, 0
            labels: list[str] = []
            spec = action_spec_from_config(cfg["action"])
            for _ in range(args.max_steps):
                action = policy.act(obs)
                labels.append(action_label(action, spec))
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)
                steps += 1
                if terminated or truncated:
                    break

            outcome = "success" if terminated else "failure"
            z_final, z_max = zs[-1], float(task_info["z_max"])
            from ngllib_agent.rewards import ZRewardConfig, effective_z_tolerance
            rc = cfg["reward"]
            z_tol = effective_z_tolerance(
                ZRewardConfig(z_tolerance=rc["z_tolerance"],
                              z_tolerance_frac=rc.get("z_tolerance_frac")),
                task_info)
            print(f"[video] {label} pair {pair_idx} ({pair['length_nm']} nm): "
                  f"{outcome} steps={steps} return={ep_return:.3f} "
                  f"dz_final={z_final - z_max:+.1f} frames={len(frames)}",
                  flush=True)
            if got[outcome]:
                continue  # already have a video for this outcome
            got[outcome] = True
            path = os.path.join(
                args.out_dir, f"{label}_{outcome}_pair{pair_idx}_{steps}steps.mp4")
            hud = annotate_frames(frames, zs, z_max, z_tol, terminated, labels)
            imageio.mimwrite(path, hud, fps=args.fps,
                             codec="libx264", quality=8)
            manifest.append({
                "quartile": label, "outcome": outcome, "pair_idx": pair_idx,
                "root_id": str(pair["root_id"]),
                "length_nm": int(pair["length_nm"]),
                "steps": steps, "episode_return": round(ep_return, 4),
                "z_start": round(zs[0], 1), "z_final": round(z_final, 1),
                "z_max": round(z_max, 1), "dz_final": round(z_final - z_max, 1),
                "z_min": round(float(task_info["z_min"]), 1),
                "z_tolerance": z_tol,
                "z_series": [round(z, 1) for z in zs],
                "actions": labels,
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
