"""Which EM_GAIN actually maximises 2D-pane parity?

probe_em_tone fitted a pure gain of 1.0257 on top of the shipping EM_GAIN
(0.978), i.e. ~1.0032, on mean |diff|. That is one objective; the constant is
load-bearing for every existing run, and an older sweep "re-confirmed" 0.978,
so this re-renders the pane at each candidate gain and scores the metrics that
matter -- block-SSIM and segment IoU -- alongside mean |diff|.

Chrome is rendered once; only the simulator re-renders per candidate.

    uv run --no-sync python native/probe_em_gain_sweep.py \
        --gains 0.978,0.99,1.0,1.0032,1.02 --out-dir /scratch/kp0374/em_gain
"""

from __future__ import annotations

import argparse
import copy
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


def states_for(base: dict) -> dict[str, dict]:
    out = {"base": base}
    for name, key, mult in (("zoom_in", "projectionScale", 0.25),
                            ("zoom_out", "projectionScale", 4.0),
                            ("xs_in", "crossSectionScale", 0.5),
                            ("xs_out", "crossSectionScale", 2.0)):
        st = copy.deepcopy(base)
        st[key] = float(base[key]) * mult
        out[name] = st
    return out


def render(make, tag, states, settle_s, out_dir):
    from PIL import Image

    r = make()
    frames = {}
    try:
        r.open()
        for name, st in states.items():
            r.reset_to(st)
            time.sleep(settle_s)
            _, img = r.observe()
            frames[name] = np.asarray(img).astype(np.uint8)
            if out_dir:
                Image.fromarray(frames[name]).save(f"{out_dir}/{tag}_{name}.png")
    finally:
        r.close()
    print(f"  [{tag}] {len(frames)} frames", flush=True)
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", default="0.978,0.99,1.0,1.0032,1.02")
    ap.add_argument("--out-dir", default="/scratch/kp0374/em_gain")
    ap.add_argument("--settle-s", type=float, default=4.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from ngllib.chrome import ChromeRenderer
    from ngllib.simulator import SimulatorRenderer
    from ngllib.simulator.pane2d import EM_GAIN, mask_ui

    layout = dict(window_size=(1800, 900), capture_scale=0.5, left_pane=True, right_pane=True)
    base = ChromeRenderer(screenshot_format="png", **layout).default_state()
    states = states_for(base)
    print(f"cases: {', '.join(states)} | shipping EM_GAIN={EM_GAIN}", flush=True)

    chrome = render(lambda: ChromeRenderer(screenshot_format="png", **layout),
                    "chrome", states, args.settle_s, args.out_dir)

    print(f"\n{'gain':8s} {'mad':>6s} {'block_ssim':>11s} {'colour-IoU':>11s}", flush=True)
    for gain in [g.strip() for g in args.gains.split(",")]:
        os.environ["NGL_NATIVE_EM_GAIN"] = gain
        sim = render(lambda: SimulatorRenderer(**layout), f"sim_{gain}", states,
                     args.settle_s, args.out_dir)
        mads, ssims, ious = [], [], []
        for name in states:
            mid = chrome[name].shape[1] // 2
            cp = mask_ui(chrome[name])[:, :mid]      # 2D pane only
            sp = mask_ui(sim[name])[:, :mid]
            mads.append(float(np.abs(cp.astype(np.int16) - sp.astype(np.int16)).mean()))
            ssims.append(block_ssim(cp, sp))
            ious.append(iou(coloured(cp), coloured(sp)))
        print(f"{gain:8s} {np.mean(mads):6.2f} {np.mean(ssims):11.4f} {np.mean(ious):11.4f}",
              flush=True)
    os.environ.pop("NGL_NATIVE_EM_GAIN", None)
    print("EMGAIN-DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
