"""Do the two backends ANIMATE a selection the same way, not just settle to it?

probe_select_parity established that a double-click at the same pixel ends with
the same selected set and the same final panes (72/72). That says nothing about
the frames in between, and the frames in between are what a policy actually
sees: a rollout spends most of its steps mid-stream, not settled.

For each probe click this records N CONSECUTIVE frames from both backends and
compares the trajectories:

  settle step   the first frame after which the pane stops changing
                (block_ssim(frame_i, final) >= --settle-ssim). Chrome streams
                chunks and downloads meshes; the simulator streams tiles and
                loads meshes inline. If those differ, a policy trained on one
                sees the consequence of its click at a different LAG than it
                would in the other -- a sim2real gap invisible to a
                settled-frame comparison.
  per-step SSIM native vs browser at the SAME step index, per pane.
  tint curve    tinted fraction per step, per pane; shows WHAT changes when.

Two modes so each backend gets its own job, as in probe_select_parity.

    uv run --no-sync python native/probe_select_dynamics.py --mode browser \
        --config configs/native_select.yaml --pool eval_d0_v1.parquet \
        --n-states 4 --steps 8 --out /scratch/.../dyn_browser.jsonl \
        --frame-dir /scratch/.../dyn_frames

    uv run --no-sync python native/probe_select_dynamics.py --mode native \
        --config configs/native_select.yaml \
        --browser-jsonl /scratch/.../dyn_browser.jsonl \
        --frame-dir /scratch/.../dyn_frames
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from probe_select_parity import (  # same package dir; shared, not re-derived
    CSS_PANE,
    CSS_TOOLBAR,
    CSS_VIEW_H,
    DBL,
    NOOP,
    TOOLBAR,
    browser_segments,
    dict_action,
    jsonable,
    make_cfg,
    sample_states,
    tint_frac,
)

_CX, _CY = CSS_PANE / 2.0, CSS_TOOLBAR + CSS_VIEW_H / 2.0

# A deselect (centre, usually on the target) and two neighbour selects. Kept
# short: this probe costs N frames per click on both backends.
PROBE_PX = [
    ("centre", _CX, _CY),
    ("right_60", _CX + 60, _CY),
    ("down_200", _CX, _CY + 200),
]


def block_ssim(a: np.ndarray, b: np.ndarray, block: int = 16) -> float:
    """Mean SSIM over non-overlapping blocks (the metric the parity campaign
    validated; a global SSIM is not comparable to those numbers)."""
    ga = np.asarray(a, np.float64).mean(axis=2)
    gb = np.asarray(b, np.float64).mean(axis=2)
    h = (ga.shape[0] // block) * block
    w = (ga.shape[1] // block) * block
    ga = ga[:h, :w].reshape(h // block, block, w // block, block)
    gb = gb[:h, :w].reshape(h // block, block, w // block, block)
    ma, mb = ga.mean(axis=(1, 3)), gb.mean(axis=(1, 3))
    va, vb = ga.var(axis=(1, 3)), gb.var(axis=(1, 3))
    cov = (ga * gb).mean(axis=(1, 3)) - ma * mb
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float((((2 * ma * mb + c1) * (2 * cov + c2))
                  / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())


def panes(image):
    """(2D pane, 3D pane), toolbar rows dropped. Chrome draws coloured layer
    tabs and a scale bar in that strip; the simulator renders it black by
    design, so including it would score a permanent, meaningless difference."""
    a = np.asarray(image, dtype=np.uint8)
    return a[TOOLBAR:, :450, :3], a[TOOLBAR:, 450:900, :3]


def settle_step(frames, ssim_thresh: float) -> int:
    """First index after which every frame matches the LAST one.

    Counted from the click (index 0 is the frame the click itself returned), so
    0 means "changed immediately and stayed", and len-1 means "still moving at
    the end of the window".
    """
    last = frames[-1]
    for i in range(len(frames)):
        if all(block_ssim(f, last) >= ssim_thresh for f in frames[i:]):
            return i
    return len(frames) - 1


def capture_sequence(inner, x, y, steps: int):
    """Frames from the click frame through `steps` following no-op steps."""
    seq = [np.asarray(inner.step(dict_action(x, y, DBL))[0]["image"],
                      dtype=np.uint8)]
    for _ in range(steps):
        seq.append(np.asarray(inner.step(dict_action(_CX, _CY, NOOP))[0]["image"],
                              dtype=np.uint8))
    return seq


def save_seq(out_dir, tag, seq):
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    for i, f in enumerate(seq):
        Image.fromarray(f).save(os.path.join(out_dir, f"{tag}_t{i:02d}.png"))


def metrics(seq, ssim_thresh):
    p2 = [panes(f)[0] for f in seq]
    p3 = [panes(f)[1] for f in seq]
    return {
        "tint2d": [round(tint_frac(f, 0, 450), 5) for f in seq],
        "tint3d": [round(tint_frac(f, 450, 900), 5) for f in seq],
        "settle2d": settle_step(p2, ssim_thresh),
        "settle3d": settle_step(p3, ssim_thresh),
    }


# --------------------------------------------------------------- browser mode
def mode_browser(args) -> int:
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    from ngllib_agent.env_build import build_env

    cfg = make_cfg(args)
    cfg["env"]["backend"] = "browser"
    env = build_env(cfg)
    inner = env.unwrapped

    records = []
    for k, rid, state, ti in sample_states(cfg, args):
        rec = {"idx": k, "root_id": rid, "state": state,
               "task_info": jsonable(ti), "probes": []}
        for name, x, y in PROBE_PX:
            try:
                inner.reset(options={"state": state, "task_info": ti})
                seq = capture_sequence(inner, x, y, args.steps)
                after = sorted(browser_segments(inner))
            except Exception as e:  # noqa: BLE001
                print(f"[{k:02d}/{name}] browser failed: {e}", flush=True)
                continue
            m = metrics(seq, args.settle_ssim)
            rec["probes"].append({"name": name, "xy": [x, y], "after": after,
                                  **m})
            if args.frame_dir:
                save_seq(args.frame_dir, f"browser_{k:02d}_{name}", seq)
            print(f"[{k:02d}/{name:<9}] settle 2d={m['settle2d']} "
                  f"3d={m['settle3d']}  tint2d={m['tint2d']}", flush=True)
        records.append(rec)
        with open(args.out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    env.close()
    print(f"[browser] wrote {len(records)} states -> {args.out}", flush=True)
    return 0


# ---------------------------------------------------------------- native mode
def mode_native(args) -> int:
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    from PIL import Image

    from ngllib_agent.env_build import build_env

    cfg = make_cfg(args)
    cfg["env"]["backend"] = "native"
    env = build_env(cfg)
    inner = env.unwrapped

    records = [json.loads(line) for line in open(args.browser_jsonl)]
    s2, s3 = [], []        # settle step, per backend, per pane
    b2, b3 = [], []
    cross2, cross3 = [], []   # per-step native-vs-browser SSIM
    rows = []

    for rec in records:
        state, ti = rec["state"], rec["task_info"]
        for p in rec["probes"]:
            x, y = p["xy"]
            try:
                inner.reset(options={"state": state, "task_info": ti})
                seq = capture_sequence(inner, x, y, args.steps)
            except Exception as e:  # noqa: BLE001
                print(f"[{rec['idx']:02d}/{p['name']}] native failed: {e}",
                      flush=True)
                continue
            m = metrics(seq, args.settle_ssim)
            if args.frame_dir:
                save_seq(args.frame_dir, f"native_{rec['idx']:02d}_{p['name']}",
                         seq)
                # Frame-by-frame SSIM needs the browser's own frames; they were
                # saved by the browser pass into the same directory.
                bs = []
                for i in range(len(seq)):
                    fp = os.path.join(
                        args.frame_dir,
                        f"browser_{rec['idx']:02d}_{p['name']}_t{i:02d}.png")
                    if not os.path.exists(fp):
                        bs = []
                        break
                    bs.append(np.asarray(Image.open(fp), dtype=np.uint8))
                if bs:
                    c2 = [block_ssim(panes(a)[0], panes(b)[0])
                          for a, b in zip(seq, bs)]
                    c3 = [block_ssim(panes(a)[1], panes(b)[1])
                          for a, b in zip(seq, bs)]
                    cross2.append(c2)
                    cross3.append(c3)
            s2.append(m["settle2d"]); s3.append(m["settle3d"])
            b2.append(p["settle2d"]); b3.append(p["settle3d"])
            rows.append((rec["idx"], p["name"], m, p))
            print(f"[{rec['idx']:02d}/{p['name']:<9}] settle 2d "
                  f"native={m['settle2d']} browser={p['settle2d']}  "
                  f"3d native={m['settle3d']} browser={p['settle3d']}",
                  flush=True)
    env.close()

    if not rows:
        print("no usable probes")
        return 1

    print("\n============== selection dynamics ==============")
    print(f"clicks                    : {len(rows)}   window {args.steps + 1} frames")
    print(f"settle step 2D  native {np.median(s2):.1f}  browser {np.median(b2):.1f}"
          f"   (median; {args.steps} = never settled in the window)")
    print(f"settle step 3D  native {np.median(s3):.1f}  browser {np.median(b3):.1f}")
    if cross2:
        c2 = np.asarray(cross2)
        c3 = np.asarray(cross3)
        print("\nper-step block_ssim, native vs browser at the same step index:")
        print("  step  :  " + "  ".join(f"{i:5d}" for i in range(c2.shape[1])))
        print("  2D    :  " + "  ".join(f"{v:5.3f}" for v in np.median(c2, axis=0)))
        print("  3D    :  " + "  ".join(f"{v:5.3f}" for v in np.median(c3, axis=0)))
        print("  A curve that RISES with step means the two converge but at "
              "different speeds -- the settled frames match while the interim "
              "ones do not, which is exactly what a settled-frame probe misses.")
    print("\ntint curves (median over clicks), fraction of pane tinted:")
    for lab, key in (("2D", "tint2d"), ("3D", "tint3d")):
        nat = np.median([r[2][key] for r in rows], axis=0)
        brw = np.median([r[3][key] for r in rows], axis=0)
        print(f"  {lab} native : " + "  ".join(f"{v:5.3f}" for v in nat))
        print(f"  {lab} browser: " + "  ".join(f"{v:5.3f}" for v in brw))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["browser", "native"], required=True)
    ap.add_argument("--config", default="configs/native_select.yaml")
    ap.add_argument("--pool", default="eval_d0_v1.parquet")
    ap.add_argument("--n-states", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8,
                    help="no-op steps captured after the click")
    ap.add_argument("--settle-ssim", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=None)
    ap.add_argument("--browser-jsonl", default=None)
    ap.add_argument("--frame-dir", default=None)
    args = ap.parse_args()
    if args.mode == "browser":
        if not args.out:
            ap.error("--out required in browser mode")
        return mode_browser(args)
    if not args.browser_jsonl:
        ap.error("--browser-jsonl required in native mode")
    return mode_native(args)


if __name__ == "__main__":
    raise SystemExit(main())
