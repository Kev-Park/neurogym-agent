"""Are missed clicks made while meshes are on screen, and when the 3D pane is
empty, is that a rendering failure or are there genuinely no meshes to show?

Rolls a policy on holdout starts and, at EVERY step, classifies the 3D pane
using the renderer's own GL state and camera:

  mesh_px     mesh pixels in the whole pane (depth buffer, meshes only)
  pane state  MESH        mesh pixels on screen
              NO_SELECT   nothing selected
              LOADING     selected, but no selected mesh has arrived yet
              OFF_VIEW    meshes loaded, but no vertex projects into the view
              NOT_DRAWN   meshes loaded AND vertices project into the view, yet
                          no mesh pixel was drawn -> a rendering problem
              EVICTED     a selected mesh had been loaded but was LRU-evicted
                          from the GPU slot pool (counted alongside the above)

Vertices are captured by wrapping MeshRenderer.load_mesh (a subsample per
segment) and projected with the renderer's own view/proj matrices.
For every 3D right-click it also records whether the click hit anything and
the pane state at that step.

    NGL_NATIVE_VAO_LRU_MB=128 uv run --no-sync python scripts/probe_empty_screens.py \
        --config configs/native_zmaxleft_v7_eval.yaml --state-pkl <ckpt> --holdout <parquet>
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from eval_zmaxleft import StatePklPolicy, state_from_row  # noqa: E402
from probe_click_visibility import classify  # noqa: E402

FAR = 0.9999
MAX_VERTS = 4000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--state-pkl", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="empty_screens.json")
    args = ap.parse_args()

    import gymnasium as gym
    import torch

    from ngllib import state as S
    from ngllib.simulator import renderer as R
    from ngllib_agent.env_build import action_spec_from_config, build_env, load_config
    from ngllib_agent.wrappers.action import cell_to_pixel

    from ngllib.simulator.render3d import MeshRenderer

    # The MeshRenderer is created lazily at the first reset, so the vertex
    # capture is patched onto the class before anything is built.
    verts: dict[str, np.ndarray] = {}
    ever_loaded: set[str] = set()
    orig_load = MeshRenderer.load_mesh

    def load_mesh(self, root_id, vertices_nm, faces, normals=None, replace=False):
        v = np.asarray(vertices_nm, dtype=np.float64)
        if len(v) > MAX_VERTS:
            v = v[np.random.default_rng(0).choice(len(v), MAX_VERTS, replace=False)]
        verts[str(root_id)] = v
        ever_loaded.add(str(root_id))
        return orig_load(self, root_id, vertices_nm, faces, normals=normals, replace=replace)

    MeshRenderer.load_mesh = load_mesh

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

    def pane_state():
        mr = rr._renderer
        st = rr._state
        ids = S.visible_segments(st["segments"])
        pos_nm = np.asarray(st["position"], dtype=np.float64) * rr._voxel_nm
        quat = st["projectionOrientation"]
        zoom_nm = float(st["projectionScale"]) * R.SCALE_CAL_NM
        loaded = [r for r in ids if mr.has_mesh(r)]
        evicted = [r for r in ids if not mr.has_mesh(r) and r in ever_loaded]
        depth, view, proj = mr.pick_depth(ids, pos_nm, quat, zoom_nm)
        # displayed rows only (the composite drops the last rows)
        mesh_px = int((depth[:R.PANE_H_3D] < FAR).sum())
        in_view = 0
        for r in loaded:
            v = verts.get(r)
            if v is None:
                continue
            clip = (proj @ view @ np.c_[v, np.ones(len(v))].T).T
            wv = clip[:, 3]
            ok = wv > 1e-9
            ndc = clip[ok, :3] / wv[ok, None]
            inside = ((np.abs(ndc[:, 0]) <= 1) & (np.abs(ndc[:, 1]) <= 1)
                      & (ndc[:, 2] >= -1) & (ndc[:, 2] <= 1))
            in_view += int(inside.sum())
        if mesh_px > 0:
            cls = "MESH"
        elif not ids:
            cls = "NO_SELECT"
        elif not loaded:
            cls = "LOADING"
        elif in_view == 0:
            cls = "OFF_VIEW"
        else:
            cls = "NOT_DRAWN"
        return {"pane": cls, "mesh_px": mesh_px, "n_sel": len(ids), "n_loaded": len(loaded),
                "n_evicted": len(evicted), "verts_in_view": in_view,
                "in_flight": bool(rr._mesh_futs)}

    spec = action_spec_from_config(cfg["action"])
    torch.manual_seed(args.seed)
    policy = StatePklPolicy(args.state_pkl, env, cfg.get("model", {}), stochastic=True)

    rows = pq.read_table(args.holdout).to_pylist()
    rng = np.random.default_rng(args.seed)
    picks = [rows[i] for i in rng.choice(len(rows), size=min(args.n, len(rows)), replace=False)]

    steps, clicks = [], []
    for i, row in enumerate(picks):
        state, info = state_from_row(row)
        frames.clear()
        obs, _ = env.reset(options={"state": state, "task_info": info})
        if i == 0:
            print(f"[empty] mesh slot pool: {rr._renderer._n_slots} slots "
                  f"(NGL_NATIVE_VAO_LRU_MB={os.environ.get('NGL_NATIVE_VAO_LRU_MB', 'unset')})",
                  flush=True)
        for _ in range(args.max_steps):
            ps = pane_state()
            ps["idx"] = int(row["idx"])
            steps.append(ps)
            a = np.asarray(policy.act(obs)).ravel()
            rec = None
            if int(a[0]) == 0:
                col = int(a[1]) % spec.grid_cols
                if not (spec.pane_x0 < 900.0 and col < spec.grid_cols // 2):
                    x, y = cell_to_pixel(int(a[1]), spec)
                    rec = classify(rr, x, y, frames[-1])
            obs, _, term, trunc, _ = env.step(a)
            if rec is not None:
                rec.update({"pane": ps["pane"], "mesh_px": ps["mesh_px"], "idx": ps["idx"]})
                clicks.append(rec)
            if term or trunc:
                break
        c = collections.Counter(s["pane"] for s in steps if s["idx"] == int(row["idx"]))
        print(f"[empty] idx={int(row['idx'])} steps={sum(c.values())} " + " ".join(
            f"{k}={c[k]}" for k in ("MESH", "NO_SELECT", "LOADING", "OFF_VIEW", "NOT_DRAWN")), flush=True)

    n = len(steps)
    c = collections.Counter(s["pane"] for s in steps)
    print(f"[empty] === pane state over all {n} steps ===", flush=True)
    for k in ("MESH", "NO_SELECT", "LOADING", "OFF_VIEW", "NOT_DRAWN"):
        print(f"[empty]   {k:<10} {c[k]:>6} ({100 * c[k] / max(1, n):.1f}%)", flush=True)
    ev = [s for s in steps if s["n_evicted"] > 0]
    print(f"[empty] steps with a selected mesh EVICTED from the GPU slot pool: {len(ev)} "
          f"({100 * len(ev) / max(1, n):.1f}%); of those, pane had no mesh pixels: "
          f"{sum(s['pane'] != 'MESH' for s in ev)}", flush=True)
    sel = [s["n_sel"] for s in steps]
    print(f"[empty] selected segments per step: mean {np.mean(sel):.1f}, max {max(sel)}", flush=True)

    misses = [r for r in clicks if r["cls"] == "MISS"]
    cm = collections.Counter(r["pane"] for r in misses)
    print(f"[empty] === {len(misses)} missed 3D right-clicks (of {len(clicks)}) by pane state ===", flush=True)
    for k in ("MESH", "NO_SELECT", "LOADING", "OFF_VIEW", "NOT_DRAWN"):
        print(f"[empty]   {k:<10} {cm[k]:>5} ({100 * cm[k] / max(1, len(misses)):.0f}%)", flush=True)
    mp = [r["mesh_px"] for r in misses if r["pane"] == "MESH"]
    if mp:
        print(f"[empty]   misses with mesh on screen: mesh pixels median {int(np.median(mp))} "
              f"(pane {R.PANE_H_3D * R.PANE} px)", flush=True)
    with open(args.out, "w") as f:
        json.dump({"steps": steps, "clicks": clicks}, f)
    print("EMPTY-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
