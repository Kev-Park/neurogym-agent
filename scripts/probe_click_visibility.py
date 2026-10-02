"""Can a 3D-pane right-click hit something the policy could NOT see?

Rolls a policy on holdout starts (simulator backend) and, BEFORE every 3D-pane
right-click is applied, classifies what is under the clicked pixel by
re-rendering depth in three variants with the renderer's own GL state:

  mesh        meshes only (what _draw_meshes put on screen)
  plane_drawn the EM section plane AS DRAWN in the observed frame
              (tiles["plane"] present, extent tiles["ext"] * PLANE_EXT_SCALE)
  plane_pick  the plane AS PICKED (always drawn, extent from the CURRENT
              crossSectionScale -- what _pick_3d_world actually tests)

and records whether the observed frame was visibly empty around that pixel,
whether any visible segment's mesh was still loading, and whether the click
moved the camera. Classes per click:

  MESH        a drawn mesh is under the cursor
  PLANE       the drawn plane is under the cursor
  INVISIBLE   only the pick-time plane is: clickable, but not on screen
  MISS        nothing pickable

Mirrors _pick_3d_world's pixel mapping and 7x7 radius search.

    uv run --no-sync python scripts/probe_click_visibility.py \
        --config configs/native_zmaxleft_v7_eval.yaml \
        --state-pkl /scratch/kp0374/checkpoints/zmaxleft-v7/ckpt_000450.pkl \
        --holdout /scratch/kp0374/neurogym-agent/eval_zmaxleft_v1.parquet --n 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from eval_zmaxleft import StatePklPolicy, state_from_row  # noqa: E402

FAR = 0.9999


def _window(depth, ix, iy, h, w):
    y0, y1 = max(0, iy - 3), min(h, iy + 4)
    x0, x1 = max(0, ix - 3), min(w, ix + 4)
    return bool((depth[y0:y1, x0:x1] < FAR).any())


def classify(rr, x_css, y_css, frame):
    from ngllib.simulator import pane2d as pane2d_mod
    from ngllib.simulator import renderer as R
    from ngllib import state as S

    st = rr._state
    pos_nm = np.asarray(st["position"], dtype=np.float64) * rr._voxel_nm
    quat = st["projectionOrientation"]
    zoom_nm = float(st["projectionScale"]) * R.SCALE_CAL_NM
    ids = S.visible_segments(st["segments"])
    cs = rr.layout.capture_scale
    ix = int(round(x_css * cs - R.PANE))
    iy = int(round(y_css * cs - R.TOOLBAR_3D))
    h, w = R.PANE_H_3D, R.PANE
    if not (0 <= ix < w and 0 <= iy < h):
        return None
    mr = rr._renderer
    d_mesh, _, _ = mr.pick_depth(ids, pos_nm, quat, zoom_nm)
    mesh = _window(d_mesh, ix, iy, h, w)
    tiles = rr._tiles or {}
    plane_drawn = False
    if tiles.get("plane") is not None and tiles.get("ext") is not None:
        ext = (tiles["ext"][0] * pane2d_mod.PLANE_EXT_SCALE,
               tiles["ext"][1] * pane2d_mod.PLANE_EXT_SCALE)
        d_pd, _, _ = mr.pick_depth([], pos_nm, quat, zoom_nm, plane_extent_nm=ext)
        plane_drawn = _window(d_pd, ix, iy, h, w)
    ext_pick = R.pane_extents_nm(st["crossSectionScale"], rr._canonical_nm)
    d_pp, _, _ = mr.pick_depth([], pos_nm, quat, zoom_nm, plane_extent_nm=ext_pick)
    plane_pick = _window(d_pp, ix, iy, h, w)

    # The observed frame: [left pane | right pane], right pane composited
    # TOOLBAR_3D rows down -> frame row = iy + TOOLBAR_3D.
    off = frame.shape[1] - R.PANE
    fy = iy + R.TOOLBAR_3D
    patch = frame[max(0, fy - 3):fy + 4, off + max(0, ix - 3):off + ix + 4]
    looks_empty = bool(patch.size) and int(patch.max()) <= 12

    loading = bool(rr._mesh_futs) or any(not mr.has_mesh(r) for r in ids)
    if mesh:
        cls = "MESH"
    elif plane_drawn:
        cls = "PLANE"
    elif plane_pick:
        cls = "INVISIBLE"
    else:
        cls = "MISS"
    return {"cls": cls, "looks_empty": looks_empty, "loading": loading,
            "plane_tile_missing": tiles.get("plane") is None,
            "ext_mismatch": tiles.get("ext") is not None and not np.allclose(
                np.asarray(tiles["ext"]) * pane2d_mod.PLANE_EXT_SCALE, ext_pick, rtol=1e-3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--state-pkl", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="click_visibility.json")
    args = ap.parse_args()

    import gymnasium as gym
    import torch

    from ngllib_agent.env_build import action_spec_from_config, build_env, load_config
    from ngllib_agent.wrappers.action import cell_to_pixel

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = "dino"
    cfg.setdefault("env", {})["max_episode_steps"] = args.max_steps
    env = build_env(cfg)

    frames: list[np.ndarray] = []
    w = env
    while True:
        inner = getattr(w, "env", None)
        sp = getattr(inner, "observation_space", None)
        if hasattr(w, "observation") and isinstance(sp, gym.spaces.Dict) and "image" in sp.spaces:
            break
        w = inner
    orig = w.observation

    def tapped(obs):
        frames.append(np.asarray(obs["image"], dtype=np.uint8))
        return orig(obs)

    w.observation = tapped
    rr = env.unwrapped._renderer
    spec = action_spec_from_config(cfg["action"])
    torch.manual_seed(args.seed)
    policy = StatePklPolicy(args.state_pkl, env, cfg.get("model", {}), stochastic=True)

    rows = pq.read_table(args.holdout).to_pylist()
    rng = np.random.default_rng(args.seed)
    picks = [rows[i] for i in rng.choice(len(rows), size=min(args.n, len(rows)), replace=False)]

    records = []
    for row in picks:
        state, info = state_from_row(row)
        frames.clear()
        obs, _ = env.reset(options={"state": state, "task_info": info})
        ep = Counter()
        for _ in range(args.max_steps):
            a = np.asarray(policy.act(obs)).ravel()
            rec = None
            if int(a[0]) == 0:
                col = int(a[1]) % spec.grid_cols
                if not (spec.pane_x0 < 900.0 and col < spec.grid_cols // 2):
                    x, y = cell_to_pixel(int(a[1]), spec)
                    rec = classify(rr, x, y, frames[-1])
            pos_prev = np.asarray(obs["pos_state"][:3]).copy()
            obs, _, term, trunc, _ = env.step(a)
            if rec is not None:
                rec["moved"] = not np.allclose(np.asarray(obs["pos_state"][:3]), pos_prev)
                rec["idx"] = int(row["idx"])
                records.append(rec)
                ep[rec["cls"]] += 1
            if term or trunc:
                break
        print(f"[clickvis] idx={int(row['idx'])} " + " ".join(
            f"{k}={ep[k]}" for k in ("MESH", "PLANE", "INVISIBLE", "MISS")), flush=True)

    def tab(sub, label):
        c = Counter(r["cls"] for r in sub)
        mv = Counter(r["cls"] for r in sub if r["moved"])
        n = len(sub) or 1
        print(f"[clickvis] {label}: n={len(sub)} " + " ".join(
            f"{k}={c[k]} ({100 * c[k] / n:.0f}%, moved {mv[k]})"
            for k in ("MESH", "PLANE", "INVISIBLE", "MISS")), flush=True)

    tab(records, "ALL")
    tab([r for r in records if r["loading"]], "mesh-LOADING steps")
    tab([r for r in records if not r["loading"]], "settled steps")
    empty_moved = [r for r in records if r["looks_empty"] and r["moved"]]
    print(f"[clickvis] clicks on visibly-EMPTY pixels that MOVED the camera: "
          f"{len(empty_moved)} / {sum(r['looks_empty'] for r in records)} empty-pixel clicks; "
          f"by class {dict(Counter(r['cls'] for r in empty_moved))}", flush=True)
    inv = [r for r in records if r["cls"] == "INVISIBLE"]
    print(f"[clickvis] INVISIBLE causes: plane tile missing {sum(r['plane_tile_missing'] for r in inv)}, "
          f"extent mismatch {sum(r['ext_mismatch'] for r in inv)} (of {len(inv)})", flush=True)
    with open(args.out, "w") as f:
        json.dump(records, f)
    print("CLICKVIS-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
