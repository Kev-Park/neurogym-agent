"""Parity gate for batched atlas rendering (throughput-scaling).

Renders identical synthetic 3D scenes two ways — a per-env `MeshRenderer.render`
(the baseline path) and the per-process `RenderService` (batched atlas, N scenes
submitted CONCURRENTLY from threads so they land in one batch) — and checks every
batched cell pixel-matches its single-pane render. This validates the atlas
viewport layout + the single-flip readback split (the parts most likely to be
wrong). Also checks the interop (all-VRAM) path if torch/CUDA are available.

Run on a GPU node:
    RB_OUT=<dir> uv run --no-sync python scripts/render_batch_probe.py
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from ngllib.simulator.pane2d import PANE, PANE_H
from ngllib.simulator.render3d import MeshRenderer
from ngllib.simulator.render_service import RenderService


def cube(center, half):
    c = np.asarray(center, dtype="f4")
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
                 dtype="f4") * half + c
    f = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ], dtype="i4")
    return v, f


def make_scenes():
    """A handful of scenes covering: mesh present at several orientations/zooms,
    a plane tile, and an empty selection (no mesh)."""
    pos = [0.0, 0.0, 0.0]
    zoom = 8000.0
    ext = (zoom * 2.0, zoom * 2.0)
    plane = (np.linspace(0, 255, 64)[:, None] * np.ones(64)[None, :]).astype(np.uint8)
    quats = [
        [0.0, 0.0, 0.0, 1.0],
        [0.3826834, 0.0, 0.0, 0.9238795],   # 45deg about x
        [0.0, 0.3826834, 0.0, 0.9238795],   # 45deg about y
        [0.0, 0.0, 0.3826834, 0.9238795],   # 45deg about z
    ]
    scenes = []
    for i, q in enumerate(quats):
        scenes.append(dict(root_id=["1"], position_nm=pos, quat=q,
                           zoom_nm=zoom * (1.0 + 0.1 * i), color=(1.0, 0.4, 0.2),
                           em_tile=plane, em_extent_nm=ext, em_gain=1.0))
    # empty selection (no mesh drawn) + plane only
    scenes.append(dict(root_id=[], position_nm=pos, quat=quats[0], zoom_nm=zoom,
                       color=[], em_tile=plane, em_extent_nm=ext, em_gain=1.0))
    return scenes


def _base_render(mr, sc):
    return mr.render(sc["root_id"], sc["position_nm"], sc["quat"], sc["zoom_nm"],
                     sc["color"], em_tile=sc["em_tile"], em_extent_nm=sc["em_extent_nm"],
                     em_gain=sc["em_gain"])


def main():
    outdir = os.environ.get("RB_OUT", ".")
    os.makedirs(outdir, exist_ok=True)
    scenes = make_scenes()
    n = len(scenes)
    v, f = cube([0, 0, 0], 2500.0)

    base = MeshRenderer(PANE, PANE_H)
    base.load_mesh("1", v, f)
    print("GL:", base.ctx.info.get("GL_RENDERER", "?"), flush=True)
    base_out = [_base_render(base, sc) for sc in scenes]

    # Batched readback: submit all N concurrently so they coalesce into one atlas.
    svc = RenderService(PANE, PANE_H, n, interop=False, max_delay_ms=20.0)
    svc.load_mesh("1", v, f)
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(
            svc.render, sc["root_id"], sc["position_nm"], sc["quat"], sc["zoom_nm"],
            sc["color"], sc["em_tile"], sc["em_extent_nm"], sc["em_gain"])
            for sc in scenes]
        batched = [fu.result() for fu in futs]

    worst = 0.0
    worst_mean = 0.0
    for i, (b, s) in enumerate(zip(base_out, batched)):
        d = np.abs(b.astype(np.int32) - s.astype(np.int32))
        mx, mn = int(d.max()), float(d.mean())
        worst = max(worst, mx); worst_mean = max(worst_mean, mn)
        print(f"[readback scene {i}] max={mx} mean={mn:.4f}", flush=True)
        if i == 0:
            Image.fromarray(b).save(os.path.join(outdir, "rb_base0.png"))
            Image.fromarray(s).save(os.path.join(outdir, "rb_batched0.png"))
    ok = worst <= 2
    print(f"RB-READBACK-PARITY {'PASS' if ok else 'FAIL'} "
          f"(worst max={worst}, worst mean={worst_mean:.4f})", flush=True)

    # Interop (all-VRAM) path, if CUDA is present: cell views -> flip/drop-alpha.
    try:
        import torch  # noqa: F401
        isvc = RenderService(PANE, PANE_H, n, interop=True, max_delay_ms=20.0)
        isvc.load_mesh("1", v, f)
        with ThreadPoolExecutor(max_workers=n) as ex:
            futs = [ex.submit(
                isvc.render, sc["root_id"], sc["position_nm"], sc["quat"],
                sc["zoom_nm"], sc["color"], sc["em_tile"], sc["em_extent_nm"],
                sc["em_gain"]) for sc in scenes]
            iviews = [fu.result() for fu in futs]
        iworst = 0
        for i, (b, t) in enumerate(zip(base_out, iviews)):
            img = torch.flip(t, dims=[0])[:, :, :3].cpu().numpy()  # gl_flip + drop alpha
            d = np.abs(b.astype(np.int32) - img.astype(np.int32))
            iworst = max(iworst, int(d.max()))
            print(f"[interop scene {i}] max={int(d.max())} mean={float(d.mean()):.4f}",
                  flush=True)
        print(f"RB-INTEROP-PARITY {'PASS' if iworst <= 2 else 'FAIL'} (worst max={iworst})",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"interop path skipped/failed: {e}", flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
