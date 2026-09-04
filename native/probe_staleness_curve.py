"""Characterise 2D-pane SETTLING: Chrome vs the simulator.

Chrome streams EM chunks, so after a position change the 2D pane fills in
progressively over several frames. The simulator fetches the tile
asynchronously and swaps it in atomically when it lands. Both are "stale"
after a move, but with different SHAPES -- Chrome degrades gracefully
(partial detail), the simulator shows the PREVIOUS location entirely. The
binary fresh/stale counter in probe_obs_equivalence cannot see that
difference; this measures the actual curve so the simulator's staleness can
be tuned to Chrome's instead of guessed at.

Protocol, per trial, identically for both envs:
  1. reset to a provider-sampled state (both settle on reset)
  2. one click that moves position (the only thing that changes tile key)
  3. N no-op-ish steps, capturing the 2D pane each step
  4. settle: extra dwell, then capture the SETTLED pane as reference R
  5. report similarity(frame_i, R) per step -> the settling curve

Similarity is 1 - NRMSE over the pane (1.0 = identical to settled), plus
the fraction of pixels within 8/255, which is more legible for Chrome's
partial-chunk fill.

    uv run --no-sync python native/probe_staleness_curve.py [--steps 12]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
os.environ["RAY_ADDRESS"] = "local"

PANE = 450


def left_pane(obs) -> np.ndarray | None:
    img = obs.get("image") if isinstance(obs, dict) else None
    if img is None:
        return None
    return img[:, :PANE].astype(np.float32)


def sim(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(1 - NRMSE, fraction of pixels within 8/255) vs the settled frame."""
    d = a - b
    rmse = float(np.sqrt(np.mean(d * d)))
    close = float(np.mean(np.abs(d) <= 8.0))
    return max(0.0, 1.0 - rmse / 255.0), close


def run_arm(env, provider, rng, steps, dwell, label, trials):
    curves, closes = [], []
    for t in range(trials):
        state, ti = provider(rng, None)
        env.reset(options={"state": state, "task_info": ti})
        # One click: action_type 1 (right-click / move-to-mouse) at a random
        # 3D-pane pixel. This is what actually changes the tile key.
        click = np.array([0, int(rng.integers(1024)), 4, 4, 4, 4])
        obs, *_ = env.step(click)
        frames = []
        f = left_pane(obs)
        if f is None:
            print(f"[stale] {label}: no image in obs; check obs mode", flush=True)
            return None, None
        frames.append(f)
        # No-op-ish steps: rotate by the centre bin (delta 0) so position,
        # crossSectionScale and segment stay put and the tile key is stable.
        noop = np.array([1, 0, 4, 4, 4, 4])
        for _ in range(steps - 1):
            obs, *_ = env.step(noop)
            frames.append(left_pane(obs))
        time.sleep(dwell)
        obs, *_ = env.step(noop)
        settled = left_pane(obs)
        cur = [sim(fr, settled) for fr in frames]
        curves.append([c[0] for c in cur])
        closes.append([c[1] for c in cur])
        print(f"[stale] {label} trial {t}: "
              + " ".join(f"{c[0]:.3f}" for c in cur), flush=True)
    return np.array(curves), np.array(closes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--dwell", type=float, default=6.0,
                    help="seconds to let the pane settle before the reference")
    args = ap.parse_args()

    from ngllib_agent.env_build import build_env, load_config
    from ngllib_agent.providers import FlywireSkeletonProvider

    def raw(cfg):
        cfg.setdefault("obs", {})["mode"] = "raw"
        cfg["env"].update({"image_size": None, "left_pane": True,
                           "right_pane": True, "capture_scale": 0.5})
        return cfg

    nat_cfg = raw(load_config("configs/native.yaml"))
    br_cfg = raw(load_config("configs/ppo_zmax_navigate.yaml"))

    provider = FlywireSkeletonProvider(nat_cfg["env"]["parquet_path"])

    # Same seed per arm so both see identical states and clicks.
    nat_env = build_env(nat_cfg)
    nc, nk = run_arm(nat_env, provider, np.random.default_rng(4242),
                     args.steps, args.dwell, "SIM   ", args.trials)
    nat_env.close()

    br_env = build_env(br_cfg)
    bc, bk = run_arm(br_env, provider, np.random.default_rng(4242),
                     args.steps, args.dwell, "CHROME", args.trials)
    br_env.close()

    if nc is None or bc is None:
        return 1
    print("\n[stale] mean similarity-to-settled by step index "
          "(1.000 = already settled)", flush=True)
    print("step   " + " ".join(f"{i:5d}" for i in range(args.steps)), flush=True)
    print("SIM    " + " ".join(f"{v:5.3f}" for v in nc.mean(0)), flush=True)
    print("CHROME " + " ".join(f"{v:5.3f}" for v in bc.mean(0)), flush=True)
    print("\n[stale] mean fraction of pixels within 8/255 of settled")
    print("SIM    " + " ".join(f"{v:5.3f}" for v in nk.mean(0)), flush=True)
    print("CHROME " + " ".join(f"{v:5.3f}" for v in bk.mean(0)), flush=True)
    print("\n[stale] PROBE-OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
