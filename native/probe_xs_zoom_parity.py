"""The 2D-pane zoom verb, both backends: same state, same pixels?

The arithmetic is shared (ngllib.state.next_cross_section_scale), so the state
half is close to a formality -- but the simulator has to react: crossSectionScale
sets its 2D fetch extent (pane2d.pane_extents_nm) and therefore which mip and
which tiles it pulls, while Chrome just re-renders. This checks both halves:

  - the state each backend reports after the same xs_zoom actions, including at
    the boundary where the rule keeps the previous value;
  - the 2D pane's pixels at each zoom level, Chrome vs simulator, as block-SSIM
    and colour-mask IoU (the same metrics the calibration gates use, in PNG --
    production JPEG depresses them).

Needs a GPU node.

    uv run --no-sync python native/probe_xs_zoom_parity.py --config configs/native_xszoom.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def block_ssim(a: np.ndarray, b: np.ndarray, block: int = 16) -> float:
    ga = a.mean(axis=2) if a.ndim == 3 else a.astype(float)
    gb = b.mean(axis=2) if b.ndim == 3 else b.astype(float)
    h, w = (ga.shape[0] // block) * block, (ga.shape[1] // block) * block
    ga, gb = ga[:h, :w], gb[:h, :w]
    ta = ga.reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3).reshape(-1, block * block)
    tb = gb.reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3).reshape(-1, block * block)
    mu_a, mu_b, va, vb = ta.mean(1), tb.mean(1), ta.var(1), tb.var(1)
    cov = ((ta - mu_a[:, None]) * (tb - mu_b[:, None])).mean(1)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float((((2 * mu_a * mu_b + c1) * (2 * cov + c2))
                  / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))).mean())


def coloured(x: np.ndarray, sat: int = 25) -> np.ndarray:
    return (x.max(axis=2).astype(np.int16) - x.min(axis=2).astype(np.int16)) > sat


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = float((a | b).sum())
    return float((a & b).sum()) / u if u else 1.0


def run_backend(cfg, backend, zoom_bins, settle_s, out_dir, seed_state=None):
    """One episode of xs_zoom steps. `seed_state` forces the SAME reset state
    as another backend: this config carries a step-anchored curriculum, so two
    independent resets land on different neurons and nothing is comparable
    (measured the hard way, job 972442: block_ssim 0.18 against 0.974 in the
    calibration gates)."""
    from PIL import Image

    from ngllib_agent.env_build import build_env

    c = json.loads(json.dumps(cfg))
    c["env"] = {**c.get("env", {}), "backend": backend}
    if backend == "chrome":
        c["env"]["screenshot_format"] = "png"
    env = build_env(c)
    frames, states = [], []
    try:
        if seed_state is None:
            obs, info = env.reset(seed=11)
        else:
            obs, info = env.reset(options={"state": seed_state[0],
                                           "task_info": seed_state[1]})
        states.append(dict(info["json_state"]))
        reset_seed = (dict(info["json_state"]), info.get("task_info"))
        mid = 4                                    # centre bin of 9 = no-op
        for i, zbin in enumerate(zoom_bins):
            # [verb=4 (xs_zoom), cell, rot x/y/z, zoom bin]
            obs, *_rest, info = env.step(np.array([4, 0, mid, mid, mid, zbin]))
            time.sleep(settle_s)
            obs, *_rest, info = env.step(np.array([1, 0, mid, mid, mid, mid]))  # no-op rotate: settle
            states.append(dict(info["json_state"]))
            img = obs["image"] if isinstance(obs, dict) and "image" in obs else obs
            img = np.asarray(img)
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] != img.shape[2]:
                img = np.transpose(img, (1, 2, 0))
            frames.append(img.astype(np.uint8))
            Image.fromarray(frames[-1]).save(f"{out_dir}/{backend}_{i}_bin{zbin}.png")
            print(f"  [{backend}] bin={zbin} xs={states[-1]['crossSectionScale']:.4f}", flush=True)
    finally:
        env.close()
    return frames, states, reset_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/native_xszoom.yaml")
    ap.add_argument("--out-dir", default="/scratch/kp0374/xs_zoom")
    ap.add_argument("--settle-s", type=float, default=3.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import yaml

    from ngllib.simulator.pane2d import mask_ui

    cfg = yaml.safe_load(open(args.config))
    cfg.setdefault("obs", {})["mode"] = "raw"
    cfg["env"].update({"left_pane": True, "right_pane": True, "image_size": None,
                       "capture_scale": 0.5})
    # Zoom in twice, out four times (crossing below the start), then a big
    # zoom-out to hit the keep-previous boundary from the other side.
    bins = [8, 8, 0, 0, 0, 0, 8]
    cf, cs, seed_state = run_backend(cfg, "chrome", bins, args.settle_s, args.out_dir)
    sf, ss, _ = run_backend(cfg, "simulator", bins, args.settle_s, args.out_dir,
                            seed_state=seed_state)

    print()
    print("=== crossSectionScale, step by step ===", flush=True)
    ok = True
    for i, (a, b) in enumerate(zip(cs, ss)):
        xa, xb = float(a["crossSectionScale"]), float(b["crossSectionScale"])
        same = abs(xa - xb) < 1e-6
        ok &= same
        print(f"step {i}: chrome {xa:10.4f}  simulator {xb:10.4f}  agree={same}", flush=True)

    print()
    print("=== 2D pane pixels at each zoom ===", flush=True)
    for i, (a, b) in enumerate(zip(cf, sf)):
        if a.shape != b.shape:
            print(f"frame {i}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            ok = False
            continue
        mid = a.shape[1] // 2
        ca, sa = mask_ui(a)[:, :mid], mask_ui(b)[:, :mid]
        print(f"frame {i} (bin {bins[i]}): block_ssim={block_ssim(ca, sa):.4f} "
              f"colour-IoU={iou(coloured(ca), coloured(sa)):.4f} "
              f"mean|diff|={float(np.abs(ca.astype(np.int16) - sa.astype(np.int16)).mean()):.2f}",
              flush=True)
    print(f"\nXSZOOM-PARITY {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
