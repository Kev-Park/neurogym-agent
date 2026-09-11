"""Step-by-step dynamics for EVERY verb, not just select.

probe_select_dynamics covered one action: double-click select, which after the
parts refactor needs no fetch at all and so responds on step 0 in both
backends. That result does NOT generalise. The action space is four verbs over
2048 cells, and the ones that MOVE THE VIEWER still refetch tiles, which is
precisely where the simulator's streaming and Chrome's chunk streaming can
diverge.

Verbs swept here, in the ngllib Dict action space:

  left_click      a state no-op on both (mousedown0 is a drag binding)
  right_click 2D  move-to-mouse-position on the slice view -> POSITION CHANGE
  right_click 3D  move-to-mouse-position via depth pick -> POSITION CHANGE
  double_click    select; the one already measured, kept for continuity
  rotate          delta_orient -> 3D re-render, no fetch
  zoom            delta_proj_scale -> 3D re-render, no fetch
  xs_scale        delta_xs_scale -> 2D EXTENT CHANGE, so a different tile
  translate       delta_pos -> POSITION CHANGE

Response is measured by whole-pane block_ssim against the pre-action frame,
not by tinted area: a position change rewrites the whole pane rather than
adding colour. The threshold is calibrated per backend from idle frames,
because Chrome re-renders streamed chunks between identical steps and the
simulator does not.

    uv run --no-sync python native/probe_action_dynamics.py --mode browser \
        --config configs/native_select.yaml --n-states 3 --steps 60 \
        --obs dino --out /scratch/.../act_browser.jsonl
    uv run --no-sync python native/probe_action_dynamics.py --mode native \
        --config configs/native_select.yaml --obs dino \
        --browser-jsonl /scratch/.../act_browser.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from probe_select_dynamics import block_ssim, make_encoder, panes
from probe_select_parity import jsonable, make_cfg, sample_states, tint_frac

CSS_PANE = 900.0
PANEL_CX, PANEL_CY = 450.0, 449.5      # pane2d.PANEL_*_CLICK
RIGHT_CX = CSS_PANE + PANEL_CX          # same point on the 3D pane


def action(kind: str) -> dict:
    """One ngllib Dict action per verb, all neutral except the field tested."""
    a = {
        "action_type": 0,
        "mouse_xy": np.zeros(2, dtype=np.float32),
        "modifiers": np.zeros(3, dtype=np.int8),
        "delta_pos": np.zeros(3, dtype=np.float32),
        "delta_xs_scale": np.zeros(1, dtype=np.float32),
        "delta_orient": np.zeros(3, dtype=np.float32),
        "delta_proj_scale": np.zeros(1, dtype=np.float32),
    }
    if kind == "left_click":
        a["mouse_xy"] = np.array([PANEL_CX, PANEL_CY], dtype=np.float32)
    elif kind == "right_click_2d":
        a["action_type"] = 1
        a["mouse_xy"] = np.array([PANEL_CX + 200, PANEL_CY + 150],
                                 dtype=np.float32)
    elif kind == "right_click_3d":
        a["action_type"] = 1
        a["mouse_xy"] = np.array([RIGHT_CX, PANEL_CY], dtype=np.float32)
    elif kind == "double_click":
        a["action_type"] = 2
        a["mouse_xy"] = np.array([PANEL_CX + 60, PANEL_CY], dtype=np.float32)
    elif kind == "rotate":
        a["action_type"] = 3
        a["delta_orient"][:] = (0.16, -0.08, 0.0)
    elif kind == "zoom":
        a["action_type"] = 3
        a["delta_proj_scale"][0] = 2000.0
    elif kind == "xs_scale":
        a["action_type"] = 3
        a["delta_xs_scale"][0] = 1.0
    elif kind == "translate":
        a["action_type"] = 3
        a["delta_pos"][:] = (400.0, 300.0, 0.0)
    else:
        raise ValueError(kind)
    return a


VERBS = ["left_click", "right_click_2d", "right_click_3d", "double_click",
         "rotate", "zoom", "xs_scale", "translate"]

NOOP = action("left_click")


def frame_of(inner, obs, encoder=None):
    img = np.asarray(obs["image"], dtype=np.uint8)
    if encoder is not None:
        mid = img.shape[1] // 2
        encoder.encode([img[:, :mid], img[:, mid:]])
    return img


def idle_floor(inner, encoder, n=4):
    """(pre-frame, per-pane idle SSIM floor) from repeated no-ops.

    Chrome keeps re-rendering streamed chunks between identical steps, so its
    frames differ from each other even with nothing happening. Comparing both
    backends against one fixed SSIM threshold would call that a response.
    """
    seq = [frame_of(inner, inner.step(NOOP)[0], encoder) for _ in range(n)]
    p2 = [panes(f)[0] for f in seq]
    p3 = [panes(f)[1] for f in seq]
    f2 = min(block_ssim(a, seq_last) for a, seq_last in zip(p2[:-1], p2[1:]))
    f3 = min(block_ssim(a, seq_last) for a, seq_last in zip(p3[:-1], p3[1:]))
    return seq[-1], f2, f3


def response(frames, pre, floor, margin=0.01):
    """First frame that differs from `pre` by more than idle variation."""
    thresh = min(floor - margin, 0.999)
    for i, f in enumerate(frames):
        if block_ssim(f, pre) < thresh:
            return i
    return len(frames)


def tint_response(frames, base, x0, x1, floor=0.002):
    """First frame whose tinted AREA moves, as a second opinion on `response`.

    Whole-pane SSIM is the right signal for a position change, which rewrites
    the pane, but it is blind to a selection: a newly tinted segment covers
    ~3% of the pane and never moves block_ssim past 0.99. The first per-verb
    sweep scored double_click as "no response in 60 steps" on BOTH backends for
    exactly that reason -- a metric artefact, not agreement. Reporting both
    keeps a verb from looking matched when neither measure can see it.
    """
    for i, f in enumerate(frames):
        if abs(tint_frac(f, x0, x1) - base) > floor:
            return i
    return len(frames)


def run_sequence(inner, state, ti, verb, steps, encoder, n_moves=6):
    """Response step of SUCCESSIVE moves, not just the first after a reset.

    run_verb resets, acts once and measures -- the worst case for any cache,
    because the chunk LRU is cold for wherever the action lands. A rollout does
    not look like that: it moves repeatedly around one neighbourhood, so by the
    third or fourth move the chunks are largely resident. If later moves in a
    sequence respond much faster than the first, the single-shot number
    overstates the gap a policy actually experiences.
    """
    inner.reset(options={"state": state, "task_info": ti})
    pre, f2, f3 = idle_floor(inner, encoder)
    out = []
    for _ in range(n_moves):
        q2 = panes(pre)[0]
        seq = [frame_of(inner, inner.step(action(verb))[0], encoder)]
        for _ in range(steps):
            seq.append(frame_of(inner, inner.step(NOOP)[0], encoder))
        out.append(response([panes(f)[0] for f in seq], q2, f2))
        pre = seq[-1]
    return out


def run_verb(inner, state, ti, verb, steps, encoder):
    inner.reset(options={"state": state, "task_info": ti})
    pre, f2, f3 = idle_floor(inner, encoder)
    q2, q3 = panes(pre)
    t0 = time.monotonic()
    seq = [frame_of(inner, inner.step(action(verb))[0], encoder)]
    for _ in range(steps):
        seq.append(frame_of(inner, inner.step(NOOP)[0], encoder))
    secs = (time.monotonic() - t0) / (steps + 1)
    p2 = [panes(f)[0] for f in seq]
    p3 = [panes(f)[1] for f in seq]
    return {
        "verb": verb,
        "resp2d": response(p2, q2, f2),
        "resp3d": response(p3, q3, f3),
        "tresp2d": tint_response(seq, tint_frac(pre, 0, 450), 0, 450),
        "tresp3d": tint_response(seq, tint_frac(pre, 450, 900), 450, 900),
        "floor2d": round(f2, 4),
        "floor3d": round(f3, 4),
        "sec_per_step": round(secs, 4),
    }


def mode_browser(args) -> int:
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    from ngllib_agent.env_build import build_env

    cfg = make_cfg(args)
    cfg["env"]["backend"] = "browser"
    env = build_env(cfg)
    inner = env.unwrapped
    encoder = make_encoder(cfg)

    records = []
    for k, rid, state, ti in sample_states(cfg, args):
        rec = {"idx": k, "root_id": rid, "state": state,
               "task_info": jsonable(ti), "verbs": []}
        for verb in VERBS:
            try:
                m = run_verb(inner, state, ti, verb, args.steps, encoder)
                if args.sequence:
                    m["seq2d"] = run_sequence(inner, state, ti, verb,
                                              args.steps, encoder,
                                              args.sequence)
            except Exception as e:  # noqa: BLE001
                print(f"[{k:02d}/{verb}] browser failed: {e}", flush=True)
                continue
            rec["verbs"].append(m)
            print(f"[{k:02d}/{verb:<15}] ssim 2d={m['resp2d']:>4} "
                  f"3d={m['resp3d']:>4}  tint 2d={m['tresp2d']:>4} "
                  f"3d={m['tresp3d']:>4}  {m['sec_per_step']:.3f}s/step",
                  flush=True)
        records.append(rec)
        with open(args.out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    env.close()
    print(f"[browser] wrote {len(records)} states -> {args.out}", flush=True)
    return 0


def mode_native(args) -> int:
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    from ngllib_agent.env_build import build_env

    cfg = make_cfg(args)
    cfg["env"]["backend"] = "native"
    env = build_env(cfg)
    inner = env.unwrapped
    encoder = make_encoder(cfg)

    records = [json.loads(line) for line in open(args.browser_jsonl)]
    rows = []
    for rec in records:
        state, ti = rec["state"], rec["task_info"]
        for b in rec["verbs"]:
            try:
                m = run_verb(inner, state, ti, b["verb"], args.steps, encoder)
                if args.sequence:
                    m["seq2d"] = run_sequence(inner, state, ti, b["verb"],
                                              args.steps, encoder,
                                              args.sequence)
            except Exception as e:  # noqa: BLE001
                print(f"[{rec['idx']:02d}/{b['verb']}] native failed: {e}",
                      flush=True)
                continue
            rows.append((b["verb"], m, b))
            print(f"[{rec['idx']:02d}/{b['verb']:<15}] "
                  f"2d native={m['resp2d']:>4} browser={b['resp2d']:>4}   "
                  f"3d native={m['resp3d']:>4} browser={b['resp3d']:>4}",
                  flush=True)
    env.close()

    if not rows:
        print("no usable rows")
        return 1

    win = args.steps + 1
    print("\n============== per-verb response step ==============")
    print(f"window {win} frames    ({win} = no response inside it)")
    print("ssim = whole-pane change; tint = tinted-area change (sees "
          "selections that ssim cannot)")
    print(f"{'verb':<16}{'2Dssim n':>9}{'2Dssim b':>9}{'3Dssim n':>9}"
          f"{'3Dssim b':>9}{'2Dtint n':>9}{'2Dtint b':>9}{'3Dtint n':>9}"
          f"{'3Dtint b':>9}")
    for verb in VERBS:
        sel = [(m, b) for v, m, b in rows if v == verb]
        if not sel:
            continue
        def med(key, which):
            return float(np.median([x[which][key] for x in sel]))
        print(f"{verb:<16}{med('resp2d', 0):>9.1f}{med('resp2d', 1):>9.1f}"
              f"{med('resp3d', 0):>9.1f}{med('resp3d', 1):>9.1f}"
              f"{med('tresp2d', 0):>9.1f}{med('tresp2d', 1):>9.1f}"
              f"{med('tresp3d', 0):>9.1f}{med('tresp3d', 1):>9.1f}")
    seqs = [(m, b) for _v, m, b in rows if "seq2d" in m and "seq2d" in b]
    if seqs:
        print("\n2D response over SUCCESSIVE moves (move index -> step):")
        n = min(len(m["seq2d"]) for m, _b in seqs)
        nat = [float(np.median([m["seq2d"][i] for m, _b in seqs]))
               for i in range(n)]
        brw = [float(np.median([b["seq2d"][i] for _m, b in seqs]))
               for i in range(n)]
        print("  native : " + "  ".join(f"{v:5.1f}" for v in nat))
        print("  browser: " + "  ".join(f"{v:5.1f}" for v in brw))
        print("  A native curve that FALLS with move index means the chunk "
              "cache is warming in situ, so the single-shot number is a "
              "cold-start artefact rather than what a rollout sees.")
    print("\nA verb whose simulator response is LOWER than Chrome's reacts too "
          "fast (no fetch where Chrome needs one); HIGHER means it lags. "
          "Position-changing verbs are the ones to watch: they refetch tiles, "
          "unlike select, which the parts refactor made fetch-free.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["browser", "native"], required=True)
    ap.add_argument("--config", default="configs/native_select.yaml")
    ap.add_argument("--pool", default="eval_d0_v1.parquet")
    ap.add_argument("--n-states", type=int, default=3)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--obs", choices=["raw", "dino"], default="dino")
    ap.add_argument("--sequence", type=int, default=0,
                    help="measure N successive moves per state instead of one")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=None)
    ap.add_argument("--browser-jsonl", default=None)
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
