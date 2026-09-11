"""Standing visual differences between the simulator and Chrome, with a CEILING.

Every parity number so far has been quoted against nothing: the 2D pane scores
block_ssim ~0.89 against Chrome, but Chrome does not reproduce itself exactly
either -- it streams chunks, so two captures of the SAME state differ. Without
that ceiling, 0.89 could be at the limit or far from it, and the earlier 0.9997
was a GLOBAL ssim, which is a different and much more forgiving metric.

pairs_v1 has two browser frames per state (_a, _b) for exactly this. So:

  ceiling      Chrome a vs Chrome b -- the best any renderer could score
  parity       simulator vs Chrome a
  gap          ceiling - parity, the part that is actually ours

and then attributes the gap across the differences that are known to exist and
are NOT fixed:

  toolbar      Chrome draws coloured layer tabs, a coordinate readout and a
               scale bar in the top strip; we draw black. Always present.
  sharpness    mean |gradient| ratio -- the two-step resample is known to blur
  jpeg         Chrome captures as JPEG and carries chroma noise; we render a
               clean PNG. Re-encoding ours through JPEG says how much of the
               residual is codec rather than render.
  3D content   the 3D pane is mostly black, so score it over content blocks
               too, where a mismatch actually shows

    uv run --no-sync python native/probe_visual_audit.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 12
"""

from __future__ import annotations

import argparse
import io
import json
import os

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def block_stats(a, b, block: int = 16):
    """(mean SSIM, per-block SSIM, per-block peak content)."""
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
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / (
        (ma ** 2 + mb ** 2 + c1) * (va + vb + c2))
    return float(s.mean()), s, np.maximum(A.max(axis=(1, 3)), B.max(axis=(1, 3)))


def ssim(a, b) -> float:
    return block_stats(a, b)[0]


def sharp(g) -> float:
    g = np.asarray(g, np.float64)
    if g.ndim == 3:
        g = g.mean(axis=2)
    gy, gx = np.gradient(g)
    return float(np.hypot(gy, gx).mean())


def as_jpeg(img, quality=85):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, np.uint8)).save(buf, "JPEG",
                                                    quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    from PIL import Image

    from ngllib.simulator import pane2d
    from ngllib.simulator.pane2d import mask_ui, mask_ui_enabled
    from ngllib.simulator.em import Source, unpack_ids, worker_pane_parts

    T, P, PH = pane2d.TOOLBAR, pane2d.PANE, pane2d.PANE_H
    acc: dict[str, list] = {k: [] for k in (
        "ceil2", "par2", "ceil2_nt", "par2_nt", "ceil3", "par3",
        "par3_content", "ceil3_content", "sharp_n", "sharp_c", "par2_jpeg")}

    records = [json.loads(line) for line in
               open(os.path.join(args.pairs_dir, "states.jsonl"))][:args.limit]
    for rec in records:
        st = rec["requested_state"]
        fa = os.path.join(args.pairs_dir, "frames", f"{rec['idx']:04d}_a.png")
        fb = os.path.join(args.pairs_dir, "frames", f"{rec['idx']:04d}_b.png")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        A = np.asarray(Image.open(fa))[..., :3]
        B = np.asarray(Image.open(fb))[..., :3]
        if mask_ui_enabled():
            A, B = mask_ui(A), mask_ui(B)   # stored frames predate the mask

        try:
            em_gray, ids_p, _plane = worker_pane_parts(
                Source.calibrated(args.cache_dir), list(st["position"]),
                float(st["crossSectionScale"]))
        except Exception as e:  # noqa: BLE001
            print(f"[{rec['idx']:04d}] fetch failed: {e}", flush=True)
            continue
        if em_gray is None:
            continue
        canvas = pane2d.compose_left_parts(
            em_gray, unpack_ids(ids_p), (str(st["segments"][0]),))
        if mask_ui_enabled():
            # compose_left_parts returns the 2D pane alone, so mask it inside a
            # full-width frame and take the pane back out.
            _f = np.zeros((450, 900, 3), np.uint8)
            _f[:, :pane2d.PANE] = canvas
            canvas = mask_ui(_f)[:, :pane2d.PANE]

        # --- 2D pane, whole (toolbar included) and EM region only ----------
        a2, b2, n2 = A[:, :P], B[:, :P], canvas
        acc["ceil2"].append(ssim(a2, b2))
        acc["par2"].append(ssim(n2, a2))
        acc["ceil2_nt"].append(ssim(a2[T:], b2[T:]))
        acc["par2_nt"].append(ssim(n2[T:], a2[T:]))
        acc["par2_jpeg"].append(ssim(as_jpeg(n2)[T:], a2[T:]))
        acc["sharp_n"].append(sharp(n2[T:]))
        acc["sharp_c"].append(sharp(a2[T:]))

        # --- 3D pane: whole, and content blocks only ----------------------
        a3, b3 = A[T:, P:2 * P], B[T:, P:2 * P]
        c3, s3, con3 = block_stats(a3, b3)
        acc["ceil3"].append(c3)
        sel = con3 > 12
        acc["ceil3_content"].append(float(s3[sel].mean()) if sel.any()
                                    else float("nan"))
        print(f"[{rec['idx']:04d}] 2D ceil {acc['ceil2'][-1]:.4f} "
              f"parity {acc['par2'][-1]:.4f} | noToolbar ceil "
              f"{acc['ceil2_nt'][-1]:.4f} parity {acc['par2_nt'][-1]:.4f}",
              flush=True)

    if not acc["ceil2"]:
        print("no usable states")
        return 1

    def m(k):
        return float(np.nanmedian(acc[k])) if acc[k] else float("nan")

    print("\n================ visual parity audit ================")
    print(f"states {len(acc['ceil2'])}\n")
    print("2D pane (the one a policy leans on):")
    print(f"  CEILING  Chrome a vs b      : {m('ceil2'):.4f}")
    print(f"  parity   sim vs Chrome      : {m('par2'):.4f}")
    print(f"  gap that is ours            : {m('ceil2') - m('par2'):+.4f}")
    print(f"  ceiling, toolbar excluded   : {m('ceil2_nt'):.4f}")
    print(f"  parity,  toolbar excluded   : {m('par2_nt'):.4f}")
    print(f"  gap, toolbar excluded       : "
          f"{m('ceil2_nt') - m('par2_nt'):+.4f}")
    print(f"  toolbar costs us            : "
          f"{(m('ceil2_nt') - m('par2_nt')) - (m('ceil2') - m('par2')):+.4f}"
          "  (negative => the black strip HELPS the score, so the real")
    print("                                 difference is understated there)")
    print(f"  parity if we JPEG ours too  : {m('par2_jpeg'):.4f}"
          f"   (delta {m('par2_jpeg') - m('par2_nt'):+.4f})")
    print(f"  sharpness sim / Chrome      : {m('sharp_n'):.2f} / "
          f"{m('sharp_c'):.2f}  ratio {m('sharp_n') / m('sharp_c'):.2f}")
    print("\n3D pane:")
    print(f"  CEILING  Chrome a vs b      : {m('ceil3'):.4f}")
    print(f"  ceiling, content blocks     : {m('ceil3_content'):.4f}")
    print("\nRead the GAP, not the parity number. Chrome does not reproduce "
          "itself either, so the ceiling is what any renderer could score; the "
          "difference between it and our parity is the part that is actually "
          "ours to answer for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
