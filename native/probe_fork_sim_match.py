"""How close can the simulator get to the Chrome fork, pixel for pixel?

Renders the SAME states through both renderers directly (the Renderer protocol,
production capture geometry, ngllib's UI mask applied to both), then decomposes
the residual difference instead of eyeballing it:

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


def states_for(base: dict) -> dict[str, dict]:
    """Deterministic states around the config default, covering the axes the
    two renderers could disagree on (zoom, orientation, position, selection)."""
    import copy

    def edit(**kw):
        st = copy.deepcopy(base)
        st.update(kw)
        return st

    ps = float(base["projectionScale"])
    return {
        "base": base,
        "zoom_in": edit(projectionScale=ps / 4),
        "zoom_out": edit(projectionScale=ps * 4),
        "rotated": edit(projectionOrientation=[0.0, 0.7071067811865476, 0.0, 0.7071067811865476]),
        "moved": edit(position=[float(base["position"][0]) + 256,
                                float(base["position"][1]) + 256,
                                float(base["position"][2])]),
    }


def render(make, tag, cases, settle_s, out_dir):
    """Renderer frames for every case, straight through the Renderer protocol
    (no wrappers: this needs pixels, not policy observations)."""
    from PIL import Image

    r = make()
    frames, states = {}, {}
    try:
        r.open()
        for name, st in cases.items():
            r.reset_to(st)
            time.sleep(settle_s)
            js, img = r.observe()
            frames[name] = as_hwc(img)
            states[name] = js
            Image.fromarray(frames[name]).save(f"{out_dir}/{tag}_{name}.png")
            print(f"  [{tag}] {name}: frame={frames[name].shape}", flush=True)
    finally:
        r.close()
    return frames, states


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=0, help="unused; cases are fixed")
    ap.add_argument("--out-dir", default="/scratch/kp0374/fork_match")
    ap.add_argument("--viewer-dist", default=os.environ.get("NGL_DIST"),
                    help="viewer build to serve; default = ngllib's packaged one")
    ap.add_argument("--start-url", default=None, help="default: ngllib config.json")
    ap.add_argument("--settle-s", type=float, default=4.0)
    ap.add_argument("--no-mask", action="store_true", help="skip ngllib's UI mask")
    ap.add_argument("--jpeg", action="store_true",
                    help="capture Chrome as JPEG (production default); PNG otherwise, so "
                         "lossy compression does not masquerade as a renderer difference")
    ap.add_argument("--appspot", action="store_true",
                    help="also render the hosted upstream build, to tell a simulator gap "
                         "apart from a fork-vs-appspot one")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from ngllib.chrome import ChromeRenderer
    from ngllib.simulator import SimulatorRenderer
    from ngllib.simulator.pane2d import mask_ui

    # Production capture geometry: both panes, capture_scale 0.5 -> 900x450,
    # which is the frame ngllib's UI mask coordinates are written for.
    layout = dict(window_size=(1800, 900), capture_scale=0.5,
                  left_pane=True, right_pane=True)
    common = dict(**layout)
    if args.start_url:
        common["start_url"] = args.start_url

    fmt = "jpeg" if args.jpeg else "png"
    common["screenshot_format"] = fmt
    cases = states_for(ChromeRenderer(viewer=args.viewer_dist, **common).default_state())
    print(f"cases: {', '.join(cases)} | chrome capture={fmt}", flush=True)

    arms = {}
    arms["fork"] = render(lambda: ChromeRenderer(viewer=args.viewer_dist, **common),
                          "fork", cases, args.settle_s, args.out_dir)
    if args.appspot:
        arms["appspot"] = render(lambda: ChromeRenderer(viewer="hosted", **common), "appspot", cases,
                                 args.settle_s, args.out_dir)
    sim_common = {k: v for k, v in common.items()
                  if k not in ("start_url", "screenshot_format")}
    if args.start_url:
        from ngllib.dataset import DatasetSpec
        sim_common["dataset"] = DatasetSpec.from_start_url(args.start_url)
    arms["sim"] = render(lambda: SimulatorRenderer(**sim_common), "sim", cases,
                         args.settle_s, args.out_dir)

    pairs = [("fork", "sim")] + ([("appspot", "sim"), ("fork", "appspot")] if args.appspot else [])
    for left, right in pairs:
        print()
        print(f"=== {left} vs {right}, production capture (900x450, both panes) ===", flush=True)
        (lf, ls), (rf, rs) = arms[left], arms[right]
        for name in cases:
            ca, sa = lf[name], rf[name]
            if ca.shape != sa.shape:
                print(f"{name}: SHAPE MISMATCH {left}={ca.shape} {right}={sa.shape}")
                continue
            if not args.no_mask:
                ca, sa = as_hwc(mask_ui(ca)), as_hwc(mask_ui(sa))
            mid = ca.shape[1] // 2      # 2D pane left half, 3D right half
            for part, (a, b) in (("frame", (ca, sa)), ("2D", (ca[:, :mid], sa[:, :mid])),
                                 ("3D", (ca[:, mid:], sa[:, mid:]))):
                mc, ms = coloured(a), coloured(b)
                dy, dx, best = best_offset(mc, ms)
                print(f"{name:9s} {part:5s}: identical={float((a == b).all(axis=2).mean()):.4f} "
                      f"mean|diff|={float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()):5.2f} "
                      f"block_ssim={block_ssim(a, b):.4f} colour-IoU={iou(mc, ms):.4f} "
                      f"best-offset=({dy:+d},{dx:+d})->{best:.4f} "
                      f"cover {left[:1]}={mc.mean():.4f} {right[:1]}={ms.mean():.4f}", flush=True)
            worst = sorted(region_table(ca, sa), key=lambda t: t[2])[:4]
            print("   worst tiles (row,col,identical,mean|diff|): "
                  + ", ".join(f"({r},{c},{v:.3f},{m:.1f})" for r, c, v, m in worst), flush=True)
            diff = [k for k in sorted(set(ls[name]) | set(rs[name]))
                    if json.dumps(ls[name].get(k), sort_keys=True)
                    != json.dumps(rs[name].get(k), sort_keys=True)]
            print(f"   json_state fields differing: {diff or 'none'}", flush=True)
    print("FORKMATCH-DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
