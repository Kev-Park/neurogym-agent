"""WHERE do the panes differ? A spatial map of the sim-vs-Chrome gap.

A policy using both panes can key on anything that differs SYSTEMATICALLY, and
that includes things which are not the data at all. Chrome draws UI inside the
capture -- layer tabs and a coordinate readout across the top strip, a scale
bar in the 2D pane's bottom-left, small buttons in both panes' top-right
corners, a "Sections" control bottom-right of the 3D pane -- and we render none
of it. Those are constant, structured, fixed-position cues: the easiest thing
in the frame for a network to latch onto, and they would flip the moment the
policy met a real browser.

So instead of one number per pane, this aggregates the per-block SSIM over
states and prints it as a map. Two very different outcomes:

  gap concentrated at edges/corners -> it is Chrome's UI, and the fix is to
      mask those regions in BOTH backends so neither carries the cue
  gap spread over the interior      -> it is the EM texture and the mesh, and
      masking buys nothing

    uv run --no-sync python native/probe_gap_map.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 12
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def block_map(a, b, block: int = 16):
    ga = np.asarray(a, np.float64)
    gb = np.asarray(b, np.float64)
    if ga.ndim == 3:
        ga = ga.mean(axis=2)
    if gb.ndim == 3:
        gb = gb.mean(axis=2)
    h = (ga.shape[0] // block) * block
    w = (ga.shape[1] // block) * block
    A = ga[:h, :w].reshape(h // block, block, w // block, block)
    B = gb[:h, :w].reshape(h // block, block, w // block, block)
    ma, mb = A.mean(axis=(1, 3)), B.mean(axis=(1, 3))
    va, vb = A.var(axis=(1, 3)), B.var(axis=(1, 3))
    cov = (A * B).mean(axis=(1, 3)) - ma * mb
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * ma * mb + c1) * (2 * cov + c2)) / (
        (ma ** 2 + mb ** 2 + c1) * (va + vb + c2))


def render(mapa, title):
    """Coarse ASCII picture of a per-block SSIM map; '#' is a big difference."""
    print(f"\n{title}  (rows x cols of 16px blocks; # worst, . best)")
    ramp = "#@%*+=-:. "
    for r in range(mapa.shape[0]):
        line = "".join(
            ramp[min(len(ramp) - 1, max(0, int(v * len(ramp))))]
            for v in mapa[r])
        print("   " + line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--config", default="configs/native_select.yaml")
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    from PIL import Image

    from ngllib.native import pane2d
    from ngllib.native.pane2d import mask_ui, mask_ui_enabled
    from ngllib_agent.env_build import build_env, load_config

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = "raw"
    cfg["env"].update({"backend": "native", "image_size": None,
                       "left_pane": True, "right_pane": True,
                       "capture_scale": 0.5})
    env = build_env(cfg)
    inner = env.unwrapped
    P = pane2d.PANE

    acc2, acc3 = [], []
    records = [json.loads(line) for line in
               open(os.path.join(args.pairs_dir, "states.jsonl"))][:args.limit]
    for rec in records:
        fa = os.path.join(args.pairs_dir, "frames", f"{rec['idx']:04d}_a.png")
        if not os.path.exists(fa):
            continue
        try:
            obs, _ = inner.reset(options={"state": rec["requested_state"]})
        except Exception as e:  # noqa: BLE001
            print(f"[{rec['idx']:04d}] reset failed: {e}", flush=True)
            continue
        ours = np.asarray(obs["image"], np.uint8)
        ref = np.asarray(Image.open(fa))[..., :3]
        if mask_ui_enabled():
            # The stored browser frames predate the mask. It is a pure function
            # of pixel position, so masking them here is exactly equivalent to
            # Chrome having applied it -- and without this the probe compares a
            # masked render against an unmasked reference and reports the mask
            # as a REGRESSION.
            ref = mask_ui(ref)
        acc2.append(block_map(ours[:, :P], ref[:, :P]))
        acc3.append(block_map(ours[:, P:2 * P], ref[:, P:2 * P]))
        print(f"[{rec['idx']:04d}] mapped", flush=True)
    env.close()

    if not acc2:
        print("no usable states")
        return 1
    m2 = np.median(np.stack(acc2), axis=0)
    m3 = np.median(np.stack(acc3), axis=0)
    render(m2, "2D pane (left): per-block SSIM vs Chrome")
    render(m3, "3D pane (right): per-block SSIM vs Chrome")

    tb = max(1, pane2d.TOOLBAR // 16 + 1)
    print(f"\n2D pane: toolbar rows (top {tb} blocks) median "
          f"{np.median(m2[:tb]):.3f} vs interior {np.median(m2[tb:]):.3f}")
    print(f"3D pane: toolbar rows (top {tb} blocks) median "
          f"{np.median(m3[:tb]):.3f} vs interior {np.median(m3[tb:]):.3f}")
    edge = np.ones_like(m2, dtype=bool)
    edge[2:-2, 2:-2] = False
    print(f"2D pane: outer border {np.median(m2[edge]):.3f} vs "
          f"core {np.median(m2[~edge]):.3f}")
    print(f"3D pane: outer border {np.median(m3[edge]):.3f} vs "
          f"core {np.median(m3[~edge]):.3f}")
    print("\nA policy using BOTH panes can key on whatever differs in a fixed "
          "place. If the worst blocks sit in the toolbar and the corners, they "
          "are Chrome's UI and can be masked out of BOTH backends -- the "
          "policy then never sees the cue, in training or in deployment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
