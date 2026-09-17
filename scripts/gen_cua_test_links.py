"""Generate one-off Neuroglancer reset links + z targets for manual CUA checks.

Emits N public NG URLs (a random neuron loaded, viewer at a random node on it)
plus the z-navigate target (z_max) and tolerance bands, so a computer-use agent
(e.g. Claude in Chrome, running under a personal plan) can be pointed at each
link and scored by hand — before any real harness exists.

Pure logic: ngllib's URL builders + the FlyWire skeleton provider. No browser,
no GPU; login-node safe. The URL uses ngllib's packaged public start state
(neuroglancer-demo host + public S3 image + public GCS v783 segmentation) — no
auth. Coordinates are 4x4x40 nm voxels; the z NG shows == the z_max here.

    uv run --no-sync python scripts/gen_cua_test_links.py --n 5 --seed 20260917
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed (printed for reproducibility; random if unset).")
    ap.add_argument("--parquet",
                    default="/scratch/kp0374/neurogym-agent/segment_positions.parquet")
    ap.add_argument("--projection-scale", type=float, default=14000.0,
                    help="Fixed spawn zoom (viewer projectionScale).")
    args = ap.parse_args()

    from ngllib.dataset import (
        default_start_url,
        merge_state,
        split_state_url,
        state_to_url,
    )
    from ngllib_agent.providers import FlywireSkeletonProvider

    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % 2**31)
    rng = np.random.default_rng(seed)

    prefix, base = split_state_url(default_start_url())
    provider = FlywireSkeletonProvider(args.parquet, projection_scale=args.projection_scale)

    print(f"# {args.n} CUA test links  (seed={seed}; 4x4x40nm voxels; goal = "
          f"navigate viewer z to z_max)\n")
    for i in range(args.n):
        state, task_info = provider(rng, None)
        url = state_to_url(prefix, merge_state(base, state))
        seg = task_info["segment_id"]
        z0 = float(state["position"][2])
        zmax = float(task_info["z_max"])
        zmin = float(task_info["z_min"])
        extent = zmax - zmin
        climb = zmax - z0
        band5 = 0.05 * extent
        print(f"## Link {i + 1}  — neuron {seg}")
        print(f"   start z = {z0:.0f}   target z_max = {zmax:.0f}   "
              f"(neuron z-extent {zmin:.0f}..{zmax:.0f}, span {extent:.0f})")
        print(f"   climb from start = {climb:+.0f} vox   "
              f"success bands: strict +/-10 vox | lenient +/-5%%-extent "
              f"= +/-{band5:.0f} vox")
        print(f"   {url}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
