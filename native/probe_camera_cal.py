"""Is the 3D camera's scale calibration wrong?

Splitting the 3D pane's silhouette was the useful step. IoU, all drawn 0.719;
MESH pixels only 0.554; SECTION PLANE only 0.710. The plane is the tell: it is
a quad whose extent and pose we control outright, so if our geometry agreed
with Chrome's it would score near 1.0. It does not -- and the mesh and the
plane are drawn through the SAME projection, so an error in the camera degrades
both, which is exactly the pattern.

That points at SCALE_CAL_NM, the nm-per-projectionScale-unit constant
(zoom_nm = projectionScale * SCALE_CAL_NM, currently 4.07). It was calibrated
in the original campaign, but against click parity -- where an error costs a
couple of screen px on a mesh big enough to absorb it -- not against silhouette
overlap, which is far more sensitive to scale.

So sweep it and let the browser frames pick. If a different value clearly wins
on IoU, the constant is simply mis-set and both halves of the 3D gap shrink
together; if 4.07 wins, the camera is right and the residual really is NG's
per-chunk LOD, which is a much bigger thing to take on.

Needs a GPU node.

    uv run --no-sync python native/probe_camera_cal.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 8
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def block_ssim_content(a, b, block: int = 16):
    ga = np.asarray(a, np.float64).mean(axis=2)
    gb = np.asarray(b, np.float64).mean(axis=2)
    h = (ga.shape[0] // block) * block
    w = (ga.shape[1] // block) * block
    A = ga[:h, :w].reshape(h // block, block, w // block, block)
    B = gb[:h, :w].reshape(h // block, block, w // block, block)
    ma, mb = A.mean(axis=(1, 3)), B.mean(axis=(1, 3))
    va, vb = A.var(axis=(1, 3)), B.var(axis=(1, 3))
    cov = (A * B).mean(axis=(1, 3)) - ma * mb
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / (
        (ma ** 2 + mb ** 2 + c1) * (va + vb + c2))
    con = np.maximum(A.max(axis=(1, 3)), B.max(axis=(1, 3))) > 12
    return float(s[con].mean()) if con.any() else float("nan")


def iou(a, b) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--config", default="configs/native_select.yaml")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--scales", default="3.85,3.95,4.07,4.19,4.31,4.45")
    ap.add_argument("--plane-ext", default=None,
                    help="comma list of multipliers on the section plane's "
                         "extent; sweeps that instead of the camera scale")
    args = ap.parse_args()

    from PIL import Image

    from ngllib.simulator import renderer as nenv
    from ngllib.simulator import pane2d
    from ngllib.simulator.pane2d import mask_ui, mask_ui_enabled
    from ngllib_agent.env_build import build_env, load_config

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = "raw"
    cfg["env"].update({"backend": "native", "image_size": None,
                       "left_pane": True, "right_pane": True,
                       "capture_scale": 0.5})
    env = build_env(cfg)
    inner = env.unwrapped
    T, P = pane2d.TOOLBAR, pane2d.PANE
    base = nenv.SCALE_CAL_NM

    scales = [float(x) for x in (args.plane_ext or args.scales).split(",")]
    acc: dict[float, dict[str, list]] = {
        s: {"iou": [], "mesh": [], "plane": [], "ssim": []} for s in scales}

    records = [json.loads(line) for line in
               open(os.path.join(args.pairs_dir, "states.jsonl"))][:args.limit]
    for rec in records:
        fa = os.path.join(args.pairs_dir, "frames", f"{rec['idx']:04d}_a.png")
        if not os.path.exists(fa):
            continue
        ref_full = np.asarray(Image.open(fa))[..., :3]
        if mask_ui_enabled():
            ref_full = mask_ui(ref_full)
        ref = ref_full[T:, P:2 * P]
        mc = ref.mean(axis=2) > 12
        cc = (ref.std(axis=2) > 18) & mc
        gc = mc & ~cc

        line = [f"[{rec['idx']:04d}]"]
        for sc in scales:
            if args.plane_ext:
                # Sweep the PLANE's own extent, not the camera. The camera
                # sweep showed mesh IoU moving 0.442-0.588 with scale while
                # plane IoU barely moved (0.687-0.751), so the plane's
                # mismatch is its quad size, which the camera cannot explain.
                nenv.SCALE_CAL_NM = base
                pane2d.PLANE_EXT_SCALE = sc
            else:
                nenv.SCALE_CAL_NM = sc      # the camera's zoom calibration
            try:
                obs, _ = inner.reset(options={"state": rec["requested_state"]})
            except Exception as e:  # noqa: BLE001
                print(f"[{rec['idx']:04d}] scale {sc}: {e}", flush=True)
                continue
            ours = np.asarray(obs["image"], np.uint8)[T:, P:2 * P]
            mn = ours.mean(axis=2) > 12
            cn = (ours.std(axis=2) > 18) & mn
            gn = mn & ~cn
            acc[sc]["iou"].append(iou(mn, mc))
            acc[sc]["mesh"].append(iou(cn, cc))
            acc[sc]["plane"].append(iou(gn, gc))
            acc[sc]["ssim"].append(block_ssim_content(ours, ref))
            line.append(f"{sc}:{acc[sc]['iou'][-1]:.3f}")
        print("  ".join(line), flush=True)
    nenv.SCALE_CAL_NM = base
    env.close()

    print("\n============== 3D camera scale calibration ==============")
    print(f"{'plane ext x' if args.plane_ext else 'SCALE_CAL_NM':>13}"
          f"{'IoU all':>10}{'IoU mesh':>10}"
          f"{'IoU plane':>11}{'content ssim':>14}")
    best = None
    for sc in scales:
        if not acc[sc]["iou"]:
            continue
        row = tuple(float(np.nanmedian(acc[sc][k]))
                    for k in ("iou", "mesh", "plane", "ssim"))
        ship = 1.0 if args.plane_ext else base
        flag = "  <- ships" if abs(sc - ship) < 1e-9 else ""
        print(f"{sc:>13.2f}{row[0]:>10.3f}{row[1]:>10.3f}{row[2]:>11.3f}"
              f"{row[3]:>14.4f}{flag}")
        if best is None or row[0] > best[1]:
            best = (sc, row[0])
    if best:
        print(f"\nbest IoU at SCALE_CAL_NM={best[0]:.2f} ({best[1]:.3f}); "
              f"shipping {base:.2f}")
    print("\nIf the peak sits away from the shipping value, the camera zoom is "
          "mis-calibrated and BOTH the mesh and the plane are drawn at the "
          "wrong size -- one constant, not NG's per-chunk LOD. If 4.07 is the "
          "peak, the camera is right and the mesh residual is the LOD "
          "difference, which is a far larger thing to take on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
