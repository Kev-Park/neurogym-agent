"""How close can the simulator get to the Chrome fork, pixel for pixel?

Renders the SAME states through both backends via build_env (the production
path, masks and all), then decomposes the residual difference instead of
eyeballing it:

  - identity / mean abs diff / block-SSIM over the frame;
  - colour-mask IoU: which pixels each backend drew a segmentation colour on;
  - the (dy, dx) shift that maximises that IoU -- non-zero means a camera or
    pane-extent mismatch, zero with residual disagreement means labels/mip;
  - a per-tile table, so a localized disagreement (UI text, scale bar, section
    plane) is told apart from a global one.

Chrome runs with env.viewer_dist (the self-hosted fork), i.e. the build we
would deploy.

    uv run --no-sync python native/probe_fork_sim_match.py \
        --config configs/native_select.yaml --states 6
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def block_ssim(a: np.ndarray, b: np.ndarray, block: int = 16) -> float:
    """Mean SSIM over `block`-sized tiles of two frames."""
    ga = a.mean(axis=2) if a.ndim == 3 else a.astype(float)
    gb = b.mean(axis=2) if b.ndim == 3 else b.astype(float)
    h = (ga.shape[0] // block) * block
    w = (ga.shape[1] // block) * block
    ga, gb = ga[:h, :w], gb[:h, :w]
    ta = ga.reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3).reshape(-1, block * block)
    tb = gb.reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3).reshape(-1, block * block)
    mu_a, mu_b = ta.mean(1), tb.mean(1)
    va, vb = ta.var(1), tb.var(1)
    cov = ((ta - mu_a[:, None]) * (tb - mu_b[:, None])).mean(1)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def coloured(x: np.ndarray, sat: int = 25) -> np.ndarray:
    """Pixels a segmentation colour was drawn on (grey EM has ~no saturation)."""
    mx = x.max(axis=2).astype(np.int16)
    mn = x.min(axis=2).astype(np.int16)
    return (mx - mn) > sat


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = float((a | b).sum())
    return float((a & b).sum()) / u if u else 1.0


def best_offset(a: np.ndarray, b: np.ndarray, radius: int = 6):
    """(dy, dx, iou) maximising mask agreement -- a camera/extent probe."""
    best = (0, 0, iou(a, b))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sa = a[max(0, dy):a.shape[0] + min(0, dy), max(0, dx):a.shape[1] + min(0, dx)]
            sb = b[max(0, -dy):b.shape[0] + min(0, -dy), max(0, -dx):b.shape[1] + min(0, -dx)]
            v = iou(sa, sb)
            if v > best[2]:
                best = (dy, dx, v)
    return best


def region_table(a: np.ndarray, b: np.ndarray, rows: int = 6, cols: int = 6):
    """Per-tile identity and mean abs diff."""
    h, w = a.shape[:2]
    out = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols
            ta, tb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
            out.append((r, c, float((ta == tb).all(axis=2).mean()),
                        float(np.abs(ta.astype(np.int16) - tb.astype(np.int16)).mean())))
    return out


def as_hwc(img) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] != img.shape[2]:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img.astype(np.uint8)


def render(cfg, backend, extra, n_states, settle_s, out_dir):
    import yaml  # noqa: F401  (config already parsed; keep import local/lazy)
    from PIL import Image

    from ngllib_agent.env_build import build_env

    c = json.loads(json.dumps(cfg))
    c["env"] = {**c.get("env", {}), "backend": backend, **extra}
    env = build_env(c)
    frames, states = [], []
    try:
        for i in range(n_states):
            obs, info = env.reset(seed=1000 + i)
            time.sleep(settle_s)
            img = as_hwc(obs["image"] if isinstance(obs, dict) and "image" in obs else obs)
            frames.append(img)
            states.append(info.get("json_state"))
            Image.fromarray(img).save(f"{out_dir}/{backend}_{i}.png")
    finally:
        env.close()
    print(f"{backend}: {len(frames)} frames {frames[0].shape}", flush=True)
    return frames, states


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/native_select.yaml")
    ap.add_argument("--states", type=int, default=6)
    ap.add_argument("--out-dir", default="/scratch/kp0374/fork_match")
    ap.add_argument("--viewer-dist", default=os.environ.get("NGL_DIST", "/scratch/kp0374/ngl_fork_dist"))
    ap.add_argument("--settle-s", type=float, default=3.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import yaml

    cfg = yaml.safe_load(open(args.config))
    chrome, cstates = render(cfg, "chrome", {"viewer_dist": args.viewer_dist},
                             args.states, args.settle_s, args.out_dir)
    sim, sstates = render(cfg, "simulator", {}, args.states, args.settle_s, args.out_dir)

    print("\n=== fork-Chrome vs simulator, production path ===", flush=True)
    for i, (ca, sa) in enumerate(zip(chrome, sim)):
        if ca.shape != sa.shape:
            print(f"state {i}: SHAPE MISMATCH chrome={ca.shape} sim={sa.shape}")
            continue
        same_state = json.dumps(cstates[i], sort_keys=True) == json.dumps(sstates[i], sort_keys=True)
        mc, ms = coloured(ca), coloured(sa)
        dy, dx, best = best_offset(mc, ms)
        print(f"state {i}: same_json_state={same_state} "
              f"identical={float((ca == sa).all(axis=2).mean()):.4f} "
              f"mean|diff|={float(np.abs(ca.astype(np.int16) - sa.astype(np.int16)).mean()):5.2f} "
              f"block_ssim={block_ssim(ca, sa):.4f} colour-IoU={iou(mc, ms):.4f} "
              f"best-offset=({dy},{dx})->{best:.4f} cover chrome={mc.mean():.4f} sim={ms.mean():.4f}",
              flush=True)
        worst = sorted(region_table(ca, sa), key=lambda t: t[2])[:4]
        print("   worst tiles (row,col,identical,mean|diff|): "
              + ", ".join(f"({r},{c},{v:.3f},{m:.1f})" for r, c, v, m in worst), flush=True)
    print("FORKMATCH-DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
