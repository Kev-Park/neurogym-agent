"""Render zmax-left policy rollouts to MP4 for visual validation.

Rolls the policy (stochastic, eval protocol) on a handful of frozen holdout
states and records the raw two-pane frame each step, annotated with: step, z,
best-so-far gain, visible-segment count, and the action that produced the frame
— with double-clicks/right-clicks attributed to the 2D (left) or 3D (right)
pane, so left-pane hop behavior is visible at a glance. Final frame is stamped
with the episode's dz and whether the best climb involved a hop.

State selection (default): 2 lowest-headroom starts (hop REQUIRED to gain z),
2 median, 2 highest (long within-neuron climbs available) — or --indices.

Frame tap (same trick as eval_video.py): the raw obs["image"] exists below
DinoObservationWrapper; wrap that instance's observation method.

  uv run --no-sync python scripts/eval_zmaxleft_video.py \
      --holdout /scratch/kp0374/neurogym-agent/eval_zmaxleft_v1.parquet \
      --state-pkl /scratch/kp0374/checkpoints/zmaxleft-v2/ckpt_000200.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_d0 import StatePklPolicy  # noqa: E402
from eval_zmaxleft import state_from_row  # noqa: E402


class EpisodeTimeout(Exception):
    pass


VERBS = ["right_click", "rotate", "zoom", "dblclick", "xs_zoom"]


def action_label(a, spec) -> str:
    """Human label for a 5-verb MultiDiscrete sample, with pane attribution
    for clicks (grid_cols spans BOTH panes; col < grid_cols/2 = 2D pane)."""
    v = int(a[0])
    if v in (0, 3):
        cell = int(a[1])
        row, col = cell // spec.grid_cols, cell % spec.grid_cols
        pane = "2D" if col < spec.grid_cols // 2 else "3D"
        name = "right_click" if v == 0 else "DBLCLICK"
        return f"{name} {pane} r{row},c{col}"
    if v == 1:
        c = spec.rotation_bins_per_axis // 2
        return f"rotate x{int(a[2]) - c:+d} y{int(a[3]) - c:+d} z{int(a[4]) - c:+d}"
    if v == 2:
        return f"zoom3D {(int(a[5]) - spec.zoom_bins // 2) * spec.zoom_step:+.0f}"
    return f"xs_zoom bin {int(a[5]) - spec.zoom_bins // 2:+d}"


def annotate(frames, zs, nsegs, labels, z0, seg_z_max):
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=16)
        font_big = ImageFont.load_default(size=28)
    except TypeError:
        font = font_big = ImageFont.load_default()

    z_best_series = np.maximum.accumulate(np.asarray(zs))
    out = []
    n = len(frames)
    for i, (frame, z) in enumerate(zip(frames, zs)):
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, img.width, 44], fill=(0, 0, 0))
        d.text((8, 2),
               f"step {i}/{n - 1}  z={z:.0f}  dz_best={z_best_series[i] - z0:+.0f}  "
               f"segs={nsegs[i]}  own_ceiling={seg_z_max:.0f}",
               fill=(255, 255, 255), font=font)
        act = labels[i - 1] if i > 0 and i - 1 < len(labels) else "-- (reset)"
        color = (255, 200, 80) if "DBLCLICK" in act else (180, 220, 255)
        d.text((8, 23), f"action: {act}", fill=color, font=font)
        if i == n - 1:
            dz = z_best_series[-1] - z0
            beat = z_best_series[-1] > seg_z_max
            d.text((8, 52),
                   f"dz={dz:+.0f}  {'BEAT OWN CEILING (hop climb)' if beat else ''}",
                   fill=(0, 220, 0) if dz > 0 else (255, 60, 60), font=font_big)
        out.append(np.asarray(img))
    out.extend([out[-1]] * 15)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/native_zmaxleft.yaml")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--state-pkl", required=True)
    ap.add_argument("--out-dir", default="eval_videos_zmaxleft")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--indices", default="",
                    help="Comma-separated holdout idx values; default = 2 "
                         "lowest-/2 median-/2 highest-headroom starts.")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--torch-seed", type=int, default=0)
    args = ap.parse_args()

    import gymnasium as gym
    import imageio.v2 as imageio
    import torch

    from ngllib_agent.env_build import action_spec_from_config, build_env, load_config

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = "dino"
    cfg.setdefault("env", {})["max_episode_steps"] = args.max_steps

    frames: list[np.ndarray] = []
    zs: list[float] = []
    nsegs: list[int] = []

    def make_tapped_env():
        env = build_env(cfg)
        w = env
        while True:
            inner = getattr(w, "env", None)
            if inner is None:
                raise RuntimeError("no image->features wrapper found in env stack")
            sp = getattr(inner, "observation_space", None)
            if (hasattr(w, "observation") and isinstance(sp, gym.spaces.Dict)
                    and "image" in sp.spaces):
                break
            w = inner
        orig = w.observation

        def tapped(obs):
            frames.append(np.asarray(obs["image"], dtype=np.uint8).copy())
            zs.append(float(np.asarray(obs["position"])[2]))
            nsegs.append(len(obs.get("segments", ())))
            return orig(obs)

        w.observation = tapped
        return env

    env = make_tapped_env()
    torch.manual_seed(args.torch_seed)
    policy = StatePklPolicy(args.state_pkl, env, cfg.get("model", {}),
                            stochastic=True)
    spec = action_spec_from_config(cfg["action"])

    def _on_alarm(signum, frame):  # noqa: ARG001
        raise EpisodeTimeout()

    signal.signal(signal.SIGALRM, _on_alarm)

    rows = pq.read_table(args.holdout).to_pylist()
    if args.indices:
        want = {int(i) for i in args.indices.split(",")}
        picks = [r for r in rows if int(r["idx"]) in want]
    else:
        by_head = sorted(rows, key=lambda r: float(r["seg_z_max"]) - float(r["z"]))
        m = len(by_head) // 2
        picks = by_head[:2] + by_head[m - 1 : m + 1] + by_head[-2:]

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    for row in picks:
        idx = int(row["idx"])
        state, info = state_from_row(row)
        frames.clear(); zs.clear(); nsegs.clear()
        labels: list[str] = []
        signal.alarm(1200)
        try:
            obs, _ = env.reset(options={"state": state, "task_info": info})
            for _ in range(args.max_steps):
                a = policy.act(obs)
                labels.append(action_label(np.asarray(a).ravel(), spec))
                obs, r, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
        except EpisodeTimeout:
            print(f"[zl-video] idx={idx} WEDGED; rebuilding env", flush=True)
            try:
                env.close()
            except Exception:
                pass
            env = make_tapped_env()
            continue
        finally:
            signal.alarm(0)

        z0 = zs[0]
        dz = float(np.max(zs) - z0)
        headroom = float(row["seg_z_max"]) - float(row["z"])
        ann = annotate(frames, zs, nsegs, labels, z0, float(row["seg_z_max"]))
        name = f"zl_idx{idx:03d}_head{headroom:.0f}_dz{dz:+.0f}.mp4"
        imageio.mimwrite(os.path.join(args.out_dir, name), ann, fps=args.fps)
        manifest.append({"idx": idx, "file": name, "dz": round(dz, 1),
                         "headroom": round(headroom, 1),
                         "final_segs": nsegs[-1], "steps": len(zs) - 1})
        print(f"[zl-video] idx={idx} dz={dz:+.0f} headroom={headroom:.0f} "
              f"-> {name}", flush=True)

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[zl-video] DONE {len(manifest)} videos -> {args.out_dir}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
