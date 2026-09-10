"""Double-click (NG `select`) I/O parity: Chrome vs simulator.

The question is not pixel fidelity but INPUT/OUTPUT equivalence: given the same
state and the same click pixel, do both backends end up with the same set of
selected segments, and does the change show up in both panes?

NG binds `at:dblclick0` to `select`, which TOGGLES the segment under the cursor
in and out of the selected set. So from a reset state whose selection is
{target}:
  - a double-click ON the target (pane centre -- the viewer is placed on one of
    its skeleton nodes) must DESELECT it, leaving {},
  - a double-click on a neighbouring segment must ADD it, leaving {target, x},
  - a double-click on background must change nothing.
All three must agree between backends; that is the parity claim.

Two modes, like input_parity.py, so the browser and the simulator each run in
their own job:

    # GPU node, browser
    uv run --no-sync python native/probe_select_parity.py --mode browser \
        --config configs/native_select.yaml --pool eval_d0_v1.parquet \
        --n-states 8 --out /scratch/kp0374/native_spike/select_browser.jsonl

    # GPU node, simulator
    uv run --no-sync python native/probe_select_parity.py --mode native \
        --config configs/native_select.yaml \
        --browser-jsonl /scratch/kp0374/native_spike/select_browser.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# Probe pixels in CSS coords on the LEFT (2D EM) pane: x in [0,900),
# y in [33,900). The centre sits on the target neuron by construction; the
# ring samples neighbours; the corners are usually background.
CSS_PANE, CSS_TOOLBAR, CSS_VIEW_H = 900.0, 33.0, 867.0
_CX, _CY = CSS_PANE / 2.0, CSS_TOOLBAR + CSS_VIEW_H / 2.0
PROBE_PX = [
    ("centre", _CX, _CY),
    ("right_60", _CX + 60, _CY),
    ("left_60", _CX - 60, _CY),
    ("down_60", _CX, _CY + 60),
    ("up_60", _CX, _CY - 60),
    ("right_200", _CX + 200, _CY),
    ("down_200", _CX, _CY + 200),
    ("corner_tl", 40.0, CSS_TOOLBAR + 40.0),
    ("corner_br", CSS_PANE - 40.0, CSS_TOOLBAR + CSS_VIEW_H - 40.0),
]

DBL = 2  # ngllib Dict action_type for double_click
NOOP = 0  # left_click: a state no-op on both backends, used to let panes settle


def dict_action(x: float, y: float, action_type: int = DBL) -> dict:
    """Bare ngllib Dict action -- stepped on `env.unwrapped` so the probe can
    address arbitrary pixels instead of grid cells."""
    return {
        "action_type": action_type,
        "mouse_xy": np.array([x, y], dtype=np.float32),
        "modifiers": np.zeros(3, dtype=np.int8),
        "delta_pos": np.zeros(3, dtype=np.float32),
        "delta_xs_scale": np.zeros(1, dtype=np.float32),
        "delta_orient": np.zeros(3, dtype=np.float32),
        "delta_proj_scale": np.zeros(1, dtype=np.float32),
    }


def tint_frac(image, x0: int, x1: int) -> float:
    """Fraction of a pane's pixels carrying segment colour (not grey EM).

    The 2D pane is greyscale EM plus a 0.5-alpha segment tint, so channel
    spread is a direct read of "is a segment drawn here" -- the OUTPUT half of
    the parity claim.
    """
    a = np.asarray(image, dtype=np.float32)[:, x0:x1, :3]
    return float((a.std(axis=2) > 6).mean())


def diagnose_pick(inner, state, x_css, y_css, browser_id, native_id):
    """Locate the browser's answer inside the simulator's own label tile.

    A disagreement has two very different causes and this separates them:
      - the browser's id IS in the tile, a few px from the click => the pick
        differs by registration or mip quantisation (a near miss),
      - the browser's id is NOWHERE in the tile => the two are reading
        different data or the CSS->world mapping is wrong (a real bug).
    Returns (distance_px or None, browser_id_present_in_tile).
    """
    from ngllib.native import pane2d
    from ngllib.native.em import EMTiles

    em = EMTiles(getattr(inner, "_cache_dir", None))
    pos = np.asarray(state["position"], np.float64) * pane2d.VOXEL_NM
    xs = float(state["crossSectionScale"])
    ext = pane2d.pane_extents_nm(xs)
    shifted = pane2d.shifted_fetch_center_nm(pos, ext)
    ids = em.label_ids(shifted, ext[0], ext[1],
                       (pane2d.PANE, pane2d.PANE_H))
    if ids is None:
        return None, False
    col = x_css * pane2d.PANE / CSS_PANE
    row = (y_css - CSS_TOOLBAR) * pane2d.PANE_H / CSS_VIEW_H
    hit = np.argwhere(ids == int(browser_id))
    if hit.size == 0:
        return None, False
    d = np.maximum(np.abs(hit[:, 0] - row), np.abs(hit[:, 1] - col))
    return float(d.min()), True


def settle(inner, steps: int):
    """Step no-ops so the panes catch up before the visual is measured.

    The simulator streams its 2D pane (atomic mode: a step shows the last
    COMPLETED canvas), and Chrome streams chunks, so neither reflects a
    selection in the very same step that made it. Applied identically to both
    so the comparison stays symmetric.
    """
    obs = None
    for _ in range(steps):
        obs = inner.step(dict_action(0.0, 0.0, NOOP))[0]
    return obs


def save_frames(out_dir: str, tag: str, before, after) -> None:
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    for suffix, img in (("before", before), ("after", after)):
        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(
            os.path.join(out_dir, f"{tag}_{suffix}.png"))


def make_cfg(args):
    from ngllib_agent.env_build import load_config

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = "raw"
    cfg["env"].update({"image_size": None, "left_pane": True,
                       "right_pane": True, "capture_scale": 0.5})
    return cfg


def sample_states(cfg, args):
    import pyarrow.parquet as pq

    from ngllib_agent.providers import FlywireSkeletonProvider

    ec = cfg["env"]
    psr = ec.get("projection_scale_range")
    provider = FlywireSkeletonProvider(
        ec["parquet_path"],
        projection_scale_range=tuple(psr) if psr else None)
    pool = pq.read_table(args.pool).to_pylist()
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(pool), size=args.n_states, replace=False)
    out = []
    for k, i in enumerate(idx):
        rid = str(pool[int(i)]["root_id"])
        state, ti = provider(rng, {"segment_id": rid})
        out.append((k, rid, state, ti))
    return out


def jsonable(obj):
    """task_info carries numpy scalars/arrays from the provider; the replay
    pass must reset with the SAME task_info, so it has to survive JSON."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def browser_segments(env) -> list[str]:
    """Selected list from the live NG viewer state, VERBATIM.

    Chrome keeps segments on the segmentation LAYER, not at the top level like
    the simulator's flat state. The "!" prefix is preserved: NG tracks
    `selectedSegments` (everything listed) separately from `visibleSegments`
    (the unprefixed subset), and `select` toggles VISIBILITY -- deselecting
    rewrites an entry to "!<id>" instead of removing it. Stripping the prefix
    would make a toggle-off look like no change at all.
    """
    st = env._get_json_state()
    segs: list[str] = []
    for layer in st.get("layers", []):
        for s in layer.get("segments", []) or []:
            segs.append(str(s))
    return segs


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
                obs, _ = inner.reset(options={"state": state, "task_info": ti})
                before = sorted(browser_segments(inner))
                t_before = tint_frac(obs["image"], 0, 450)
                r_before = tint_frac(obs["image"], 450, 900)
                obs2 = inner.step(dict_action(x, y))[0]
                obs2 = settle(inner, args.settle_steps) or obs2
                after = sorted(browser_segments(inner))
                t_after = tint_frac(obs2["image"], 0, 450)
                r_after = tint_frac(obs2["image"], 450, 900)
            except Exception as e:  # noqa: BLE001
                print(f"[{k:02d}/{name}] browser failed: {e}", flush=True)
                continue
            if args.frame_dir:
                save_frames(args.frame_dir, f"browser_{k:02d}_{name}",
                            obs["image"], obs2["image"])
            rec["probes"].append(
                {"name": name, "xy": [x, y], "before": before, "after": after,
                 "tint_before": t_before, "tint_after": t_after,
                 "r3_before": r_before, "r3_after": r_after})
            print(f"[{k:02d}/{name:<10}] {before} -> {after}  "
                  f"tint2d {t_before:.4f}->{t_after:.4f}  "
                  f"tint3d {r_before:.4f}->{r_after:.4f}", flush=True)
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
    from ngllib_agent.env_build import build_env

    cfg = make_cfg(args)
    cfg["env"]["backend"] = "native"
    env = build_env(cfg)
    inner = env.unwrapped

    records = [json.loads(line) for line in open(args.browser_jsonl)]
    agree = total = 0
    tint_dir_ok = tint_dir_n = r3_dir_ok = 0
    disagreements = []

    for rec in records:
        state = rec["state"]
        ti = rec["task_info"]
        for p in rec["probes"]:
            x, y = p["xy"]
            try:
                obs, _ = inner.reset(options={"state": state, "task_info": ti})
                t_before = tint_frac(obs["image"], 0, 450)
                r_before = tint_frac(obs["image"], 450, 900)
                obs2 = inner.step(dict_action(x, y))[0]
                obs2 = settle(inner, args.settle_steps) or obs2
                after = sorted(str(s) for s in inner._json_state["segments"])
                t_after = tint_frac(obs2["image"], 0, 450)
                r_after = tint_frac(obs2["image"], 450, 900)
            except Exception as e:  # noqa: BLE001
                print(f"[{rec['idx']:02d}/{p['name']}] native failed: {e}",
                      flush=True)
                continue
            if args.frame_dir:
                save_frames(args.frame_dir, f"native_{rec['idx']:02d}_{p['name']}",
                            obs["image"], obs2["image"])
            total += 1
            ok = after == sorted(p["after"])
            agree += ok
            if not ok:
                d = {"idx": rec["idx"], "probe": p["name"],
                     "browser": p["after"], "native": after}
                if args.diagnose:
                    new_b = [i for i in p["after"] if i not in p["before"]]
                    new_n = [i for i in after if i not in p["before"]]
                    if new_b:
                        d["dist_px"], d["present"] = diagnose_pick(
                            inner, state, x, y, new_b[0],
                            new_n[0] if new_n else None)
                disagreements.append(d)
            # Direction of the visual change must match even where the exact
            # tint area cannot (the panes are not pixel-identical).
            tint_dir_n += 1
            tint_dir_ok += int(np.sign(p["tint_after"] - p["tint_before"])
                               == np.sign(t_after - t_before))
            r3_dir_ok += int(np.sign(p["r3_after"] - p["r3_before"])
                             == np.sign(r_after - r_before))
            verdict = "OK  " if ok else "DIFF"
            print(f"[{rec['idx']:02d}/{p['name']:<10}] {verdict} "
                  f"browser={p['after']} native={after}  "
                  f"tint2d {t_before:.4f}->{t_after:.4f}", flush=True)
    env.close()

    print("\n============== double-click select parity ==============")
    print(f"probes                         : {total}")
    if total:
        print(f"selected-set agreement         : {agree}/{total} "
              f"({100.0 * agree / total:.1f}%)")
        print(f"2D tint changed the same way   : {tint_dir_ok}/{tint_dir_n}")
        print(f"3D pane changed the same way   : {r3_dir_ok}/{tint_dir_n}")
    for d in disagreements[:20]:
        extra = ""
        if "present" in d:
            extra = (f"  [browser id {d['dist_px']:.0f}px away in our tile]"
                     if d["present"] else "  [browser id ABSENT from our tile]")
        print(f"  DIFF state {d['idx']} {d['probe']}: "
              f"browser={d['browser']} native={d['native']}{extra}")
    near = [d["dist_px"] for d in disagreements
            if d.get("present") and d.get("dist_px") is not None]
    absent = sum(1 for d in disagreements if "present" in d and not d["present"])
    if near or absent:
        print(f"\ndisagreements diagnosed      : {len(near) + absent}")
        print(f"  browser id present in tile : {len(near)} "
              f"(median {np.median(near):.0f} px from the click)"
              if near else "  browser id present in tile : 0")
        print(f"  browser id absent from tile: {absent}")
        print("  A few px => registration/mip quantisation. Absent => the "
              "mapping or the data source is wrong.")
    print("\nAgreement is the parity claim: identical pixel -> identical "
          "selected set. The tint lines confirm the change is REFLECTED in "
          "both panes rather than only in the state dict.")
    return 0 if total and agree == total else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["browser", "native"], required=True)
    ap.add_argument("--config", default="configs/native_select.yaml")
    ap.add_argument("--pool", default="eval_d0_v1.parquet")
    ap.add_argument("--n-states", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=None)
    ap.add_argument("--browser-jsonl", default=None)
    ap.add_argument("--diagnose", action="store_true",
                    help="on a disagreement, locate the browser id in our tile")
    ap.add_argument("--settle-steps", type=int, default=3,
                    help="no-op steps after the click, so streamed panes catch up")
    ap.add_argument("--frame-dir", default=None,
                    help="dump before/after frames per probe for inspection")
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
