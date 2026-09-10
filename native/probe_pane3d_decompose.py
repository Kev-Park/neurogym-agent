"""Why is the 3D pane's native-vs-browser block_ssim only ~0.58?

probe_select_dynamics measured 0.884 on the 2D pane and 0.580 on the 3D pane,
flat across every step. Before treating 0.580 as a rendering defect it has to
survive an obvious objection: the 3D pane is MOSTLY BLACK BACKGROUND, and SSIM
over near-uniform blocks is dominated by tiny variance differences, so a low
whole-pane score can be an artefact of the metric rather than a real mismatch.

This scores the pair three ways on the frames both passes already wrote:

  whole pane      reproduces the dynamics number, for continuity
  content blocks  SSIM over 16x16 blocks where EITHER image has real content
                  (anything above the background floor) -- what a viewer sees
  silhouette IoU  mesh pixels only (coloured, not grey plane, not black), the
                  geometric question: is the same shape in the same place

and separately measures the SECTION PLANE, since Chrome applies the
segmentation layer to it and we draw it greyscale -- a known, unimplemented
gap. Plane pixels are grey-ish in ours and coloured in Chrome's under
SHOW_ALL, so their mean channel spread quantifies how much of the residual
that one difference explains.

Runs on frame pairs, no env and no GPU:

    uv run --no-sync python native/probe_pane3d_decompose.py \
        --frame-dir /scratch/kp0374/native_spike/dyn_frames_883323
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np

TOOLBAR = 17
BG = 12.0          # below this mean intensity a pixel is background
COLOURED = 18.0    # channel spread above this is segment colour, not grey EM


def block_ssim_map(a, b, block=16):
    """Per-block SSIM plus each block's peak content, so blocks can be
    filtered by whether there is anything in them."""
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
    content = np.maximum(A.max(axis=(1, 3)), B.max(axis=(1, 3)))
    return s, content


def masks(img):
    """(foreground, coloured, plane) pixel masks for a 3D pane."""
    a = np.asarray(img, np.float64)[..., :3]
    fg = a.mean(axis=2) > BG
    col = a.std(axis=2) > COLOURED
    return fg, col, fg & ~col     # plane/grey content


def iou(m1, m2):
    u = (m1 | m2).sum()
    return float((m1 & m2).sum() / u) if u else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-dir", required=True)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    from PIL import Image

    pat = re.compile(r"^native_(\d+)_(.+)_t(\d+)\.png$")
    pairs = []
    for fn in sorted(os.listdir(args.frame_dir)):
        m = pat.match(fn)
        if not m:
            continue
        other = os.path.join(args.frame_dir,
                             f"browser_{m.group(1)}_{m.group(2)}_t{m.group(3)}.png")
        if os.path.exists(other):
            pairs.append((os.path.join(args.frame_dir, fn), other,
                          f"{m.group(1)}/{m.group(2)}/t{m.group(3)}"))
    pairs = pairs[:args.limit]
    if not pairs:
        print(f"no matched frame pairs under {args.frame_dir}")
        return 1

    whole, content, sil, fg_n, fg_b, col_n, col_b, plane_sat_n, plane_sat_b = (
        [], [], [], [], [], [], [], [], [])
    for pn, pb, _tag in pairs:
        n = np.asarray(Image.open(pn), np.uint8)[TOOLBAR:, 450:900, :3]
        b = np.asarray(Image.open(pb), np.uint8)[TOOLBAR:, 450:900, :3]
        s, c = block_ssim_map(n, b)
        whole.append(float(s.mean()))
        sel = c > BG
        content.append(float(s[sel].mean()) if sel.any() else float("nan"))

        fn_, cn, pln = masks(n)
        fb_, cb, plb = masks(b)
        sil.append(iou(cn, cb))
        fg_n.append(float(fn_.mean())); fg_b.append(float(fb_.mean()))
        col_n.append(float(cn.mean())); col_b.append(float(cb.mean()))
        an = np.asarray(n, np.float64); ab = np.asarray(b, np.float64)
        plane_sat_n.append(float(an.std(axis=2)[pln].mean()) if pln.any() else 0.0)
        plane_sat_b.append(float(ab.std(axis=2)[plb].mean()) if plb.any() else 0.0)

    def med(v):
        return float(np.nanmedian(v))

    print("\n============== 3D pane decomposition ==============")
    print(f"frame pairs                    : {len(pairs)}")
    print(f"block_ssim, whole pane         : {med(whole):.3f}")
    print(f"block_ssim, CONTENT blocks only: {med(content):.3f}")
    print(f"mesh silhouette IoU            : {med(sil):.3f}")
    print(f"foreground fraction  native {med(fg_n):.4f}  browser {med(fg_b):.4f}")
    print(f"coloured fraction    native {med(col_n):.4f}  browser {med(col_b):.4f}")
    print(f"plane channel spread native {med(plane_sat_n):.2f}  "
          f"browser {med(plane_sat_b):.2f}")
    print("\nReading:")
    print("  content-block SSIM >> whole-pane SSIM => the whole-pane number is "
          "mostly empty background and overstates the mismatch.")
    print("  silhouette IoU near 1 => the mesh is the same shape in the same "
          "place, and the residual is shading/colour, not geometry.")
    print("  browser plane spread >> native's => Chrome is colourizing the "
          "section plane and we are drawing it grey, which is the known "
          "unimplemented gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
