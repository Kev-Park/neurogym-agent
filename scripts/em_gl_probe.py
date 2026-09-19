"""Validate the GPU 2D-EM compositor (MeshRenderer.render_em) against the CPU
pane2d.compose_left_parts (the Chrome-calibrated reference). Prints per-case
pixel diff and saves GPU/CPU/diff PNGs for a visual check. Run on a GPU node:

    EM_OUT=<dir> uv run --no-sync python scripts/em_gl_probe.py
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from ngllib.simulator.pane2d import PANE, PANE_H, compose_left_parts
from ngllib.simulator.render3d import MeshRenderer


def make_data():
    # Vertical grayscale gradient EM (PANE_H x PANE), realistic FlyWire-scale ids
    # in a 3x2 block layout so tint regions are obvious.
    em = (np.linspace(0, 255, PANE_H)[:, None] * np.ones(PANE)[None, :]).astype(np.uint8)
    ids = np.zeros((PANE_H, PANE), dtype=np.uint64)
    seg = [np.uint64(720575940600000000 + i * 6133) for i in range(6)]
    bh, bw = PANE_H // 3, PANE // 2
    for bi in range(3):
        for bj in range(2):
            ids[bi * bh:(bi + 1) * bh, bj * bw:(bj + 1) * bw] = seg[bi * 2 + bj]
    return em, ids, seg


def compare(tag, r, em, ids, visible, outdir, gen):
    gpu = r.render_em(em, ids, visible, tile_key=gen)
    cpu = compose_left_parts(em, ids, visible)
    d = np.abs(gpu.astype(np.int32) - cpu.astype(np.int32))
    per_px = d.max(axis=2)
    print(f"[{tag}] shape gpu={gpu.shape} cpu={cpu.shape} | max={d.max()} "
          f"mean={d.mean():.4f} | px>2diff={100*(per_px > 2).mean():.3f}% "
          f"px>16diff={100*(per_px > 16).mean():.3f}%", flush=True)
    Image.fromarray(gpu).save(os.path.join(outdir, f"em_{tag}_gpu.png"))
    Image.fromarray(cpu).save(os.path.join(outdir, f"em_{tag}_cpu.png"))
    Image.fromarray(np.clip(d * 8, 0, 255).astype(np.uint8)).save(
        os.path.join(outdir, f"em_{tag}_diff8x.png"))
    return int(d.max()), float(d.mean())


def main():
    outdir = os.environ.get("EM_OUT", ".")
    os.makedirs(outdir, exist_ok=True)
    r = MeshRenderer(PANE, PANE_H)
    print("GL:", r.ctx.info.get("GL_RENDERER", "?"), flush=True)
    em, ids, seg = make_data()
    results = {}
    results["subset"] = compare("subset", r, em, ids, {int(seg[0]), int(seg[3])}, outdir, 1)
    results["showall"] = compare("showall", r, em, ids, set(), outdir, 2)
    results["noids"] = compare("noids", r, em, None, set(), outdir, 3)
    # PASS if EM+tint parity is tight; the crosshair is a thin center cross so a
    # few edge px may differ by rounding — gate on mean, not a hard max.
    worst_mean = max(m for _, m in results.values())
    ok = worst_mean < 0.5
    print(f"EM-GL-PARITY {'PASS' if ok else 'FAIL'} (worst mean diff={worst_mean:.4f})",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
