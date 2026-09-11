"""3D pane parity against Chrome, measured through the production path.

The visual audit established the ceiling -- Chrome reproduces itself at
block_ssim 1.0000 on settled frames -- and put the 2D pane at 0.8990. It could
not score the 3D pane, which needs GL, so the only figure for it is 0.845 from
the August campaign, predating the section-plane colourization and the
multi-mesh render.

That number matters more than the 2D one: the 3D pane carries the task signal,
and a policy that leans on it is leaning on whatever differs. So render it the
way training does -- NativeEnvironment, right pane, reset to the collected
state -- and score against the browser frame, whole-pane and over content
blocks (the pane is mostly black, and a whole-pane average hides a mismatch in
the part that is actually drawn).

Needs a GPU node for EGL.

    uv run --no-sync python native/probe_pane3d_parity.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 --limit 12
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def block_stats(a, b, block: int = 16):
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
    return s, np.maximum(A.max(axis=(1, 3)), B.max(axis=(1, 3)))


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

    T, P = pane2d.TOOLBAR, pane2d.PANE
    whole, content, fg_n, fg_c = [], [], [], []
    iou, inten, hue = [], [], []
    records = [json.loads(line) for line in
               open(os.path.join(args.pairs_dir, "states.jsonl"))][:args.limit]
    for rec in records:
        fa = os.path.join(args.pairs_dir, "frames", f"{rec['idx']:04d}_a.png")
        if not os.path.exists(fa):
            continue
        st = rec["requested_state"]
        # No task_info: the env derives it via provider.task_info_from_state.
        # Passing {"segment_id": ...} alone drops z_max/z_min and the reward
        # factory raises at reset.
        try:
            obs, _ = inner.reset(options={"state": st})
        except Exception as e:  # noqa: BLE001
            print(f"[{rec['idx']:04d}] reset failed: {e}", flush=True)
            continue
        ours = np.asarray(obs["image"], np.uint8)
        ref = np.asarray(Image.open(fa))[..., :3]
        if mask_ui_enabled():
            ref = mask_ui(ref)      # stored frames predate the mask
        ours = ours[T:, P:2 * P, :3]
        ref = ref[T:, P:2 * P, :3]

        s, con = block_stats(ours, ref)
        sel = con > 12
        w_ = float(s.mean())
        c_ = float(s[sel].mean()) if sel.any() else float("nan")
        whole.append(w_)
        content.append(c_)
        fg_n.append(float((ours.mean(axis=2) > 12).mean()))
        fg_c.append(float((ref.mean(axis=2) > 12).mean()))

        # WHAT differs: shape, shading, or colour? These have very different
        # consequences -- a silhouette mismatch means the geometry or camera
        # is off, while matching shape with different brightness is only a
        # lighting model, and a hue mismatch would mean the segment-colour
        # hash disagrees with NG's.
        mn = ours.mean(axis=2) > 12
        mc = ref.mean(axis=2) > 12
        u = (mn | mc).sum()
        iou.append(float((mn & mc).sum() / u) if u else float("nan"))
        both = mn & mc
        if both.sum() > 64:
            a = ours[both].astype(np.float64)
            b = ref[both].astype(np.float64)
            inten.append(float(a.mean() / b.mean()) if b.mean() else
                         float("nan"))
            # hue proxy: which channel dominates, per pixel
            hue.append(float((a.argmax(axis=1) == b.argmax(axis=1)).mean()))
        print(f"[{rec['idx']:04d}] whole {w_:.4f}  content {c_:.4f}  "
              f"drawn {fg_n[-1]:.3f}/{fg_c[-1]:.3f}", flush=True)
    env.close()

    if not whole:
        print("no usable states")
        return 1
    print("\n============== 3D pane parity ==============")
    print(f"states                    : {len(whole)}")
    print(f"CEILING (Chrome vs itself): 1.0000")
    print(f"block_ssim, whole pane    : {np.nanmedian(whole):.4f}")
    print(f"block_ssim, content blocks: {np.nanmedian(content):.4f}")
    print(f"fraction drawn  sim/Chrome: {np.median(fg_n):.3f} / "
          f"{np.median(fg_c):.3f}")
    print(f"silhouette IoU            : {np.nanmedian(iou):.3f}")
    if inten:
        print(f"brightness sim/Chrome     : {np.nanmedian(inten):.3f}"
              "   (on pixels BOTH draw)")
        print(f"dominant-channel agree    : {np.nanmedian(hue):.3f}"
              "   (colour hash agreement)")
    print("\nThe 3D pane is mostly black, so the whole-pane figure flatters "
          "it; the content-block one is what a policy looking at the neuron "
          "sees. The August campaign recorded 0.845 whole-pane, before the "
          "section plane was colourized and before multi-segment rendering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
