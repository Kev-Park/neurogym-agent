"""Does the trained CNN keep the information the task needs? (+ aim lift)

Rolls the spatial (patch-token) policy on holdout starts and, every `--stride`
steps, records:
  inputs     raw DINO patch tokens per pane (the CNN's input)
  trained    the trained CNN's per-pane feature map (C x 16 x 16) and the
             shared vector g every head reads
  random     the same CNN architecture at random init (floor)
  labels     from the simulator's own state:
               occ3d[16x16]  a 3D right-click at that token's cell would hit a
                             drawn MESH (renderer depth, pick's 7x7 window)
               hop2d[16x16]  the 2D-pane cell centre lies on a segment that is
                             not currently selected (a double-click target;
                             approximate pixel mapping)
               headroom      max loaded-mesh z-max minus current z (voxels)
               loading       a visible segment's mesh is still in flight
  policy     the click-cell distribution, for AIM LIFT = (policy mass on
             occupied 3D cells) / (occupied fraction of 3D cells)

Then fits linear probes (logistic / ridge, per-location probes shared across
locations) on episode-split train/test and prints held-out AUC / R^2 for
inputs vs trained vs random. Trained ~= inputs >> random  => the CNN keeps the
information; trained << inputs => it discards some.

    uv run --no-sync python scripts/probe_representation.py \
        --config configs/native_zmaxleft_v7_eval.yaml \
        --state-pkl /scratch/kp0374/checkpoints/zmaxleft-v7/ckpt_000450.pkl \
        --holdout /scratch/kp0374/neurogym-agent/eval_zmaxleft_v1.parquet --n 16
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from eval_zmaxleft import StatePklPolicy, state_from_row  # noqa: E402

FAR = 0.9999


def occ3d_labels(rr, spec, g_rows, g_cols):
    """[g_rows, g_cols] bool: would a click at each 3D cell hit a drawn mesh."""
    from ngllib import state as S
    from ngllib.simulator import renderer as R
    from ngllib_agent.wrappers.action import cell_to_pixel

    st = rr._state
    ids = S.visible_segments(st["segments"])
    pos_nm = np.asarray(st["position"], dtype=np.float64) * rr._voxel_nm
    zoom_nm = float(st["projectionScale"]) * R.SCALE_CAL_NM
    depth, _, _ = rr._renderer.pick_depth(ids, pos_nm, st["projectionOrientation"], zoom_nm)
    hit = depth < FAR
    h, w = R.PANE_H_3D, R.PANE
    cs = rr.layout.capture_scale
    out = np.zeros((g_rows, g_cols), dtype=bool)
    for r in range(g_rows):
        for c in range(g_cols):
            cell = r * spec.grid_cols + (spec.grid_cols - g_cols) + c   # 3D half
            x, y = cell_to_pixel(cell, spec)
            ix, iy = int(round(x * cs - R.PANE)), int(round(y * cs - R.TOOLBAR_3D))
            if 0 <= ix < w and 0 <= iy < h:
                out[r, c] = hit[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4].any()
    return out


def hop2d_labels(rr, g_rows, g_cols):
    """[g_rows, g_cols] bool: the 2D cell centre is on a not-selected segment.
    Maps cell centres uniformly onto the id map (approximate)."""
    from ngllib import state as S

    tiles = rr._tiles or {}
    ids = tiles.get("ids")
    out = np.zeros((g_rows, g_cols), dtype=bool)
    if ids is None:
        return out
    vis = {int(v) for v in S.visible_segments(rr._state["segments"])}
    H, W = ids.shape[:2]
    for r in range(g_rows):
        for c in range(g_cols):
            v = int(ids[int((r + 0.5) * H / g_rows), int((c + 0.5) * W / g_cols)])
            out[r, c] = v != 0 and v not in vis
    return out


def auc(y, s):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logistic(Xtr, ytr, Xte, steps=400, l2=1e-4):
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=dev)
    Xte = torch.as_tensor(Xte, dtype=torch.float32, device=dev)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    y = torch.as_tensor(ytr, dtype=torch.float32, device=dev)
    pos = y.mean().clamp(1e-3, 1 - 1e-3)
    wt = torch.where(y > 0, 0.5 / pos, 0.5 / (1 - pos))
    lin = torch.nn.Linear(Xtr.shape[1], 1).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            lin(Xtr).squeeze(-1), y, weight=wt) + l2 * lin.weight.pow(2).sum())
        loss.backward(); opt.step()
    with torch.no_grad():
        return lin(Xte).squeeze(-1).cpu().numpy()


def fit_ridge(Xtr, ytr, Xte, yte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (ytr - ym))
    pred = B @ w + ym
    return 1 - ((pred - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--state-pkl", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from ray.rllib.core.columns import Columns

    from ngllib import state as S
    from ngllib_agent.env_build import action_spec_from_config, build_env, load_config
    from ngllib_agent.policies import HierarchicalPPOModule

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = "dino"
    cfg.setdefault("env", {})["max_episode_steps"] = args.max_steps
    env = build_env(cfg)
    rr = env.unwrapped._renderer
    spec = action_spec_from_config(cfg["action"])
    torch.manual_seed(args.seed)
    pol = StatePklPolicy(args.state_pkl, env, cfg.get("model", {}), stochastic=True)
    m = pol.module.eval()
    torch.manual_seed(12345)
    rnd = HierarchicalPPOModule(observation_space=env.observation_space,
                                action_space=env.action_space,
                                model_config=cfg.get("model", {})).eval()
    n_panes, G, _ = m._grid
    n_verb = int(spec.nvec()[0])
    n_cell = spec.num_cells

    rows = pq.read_table(args.holdout).to_pylist()
    rng = np.random.default_rng(args.seed)
    picks = [rows[i] for i in rng.choice(len(rows), size=min(args.n, len(rows)), replace=False)]

    D = {k: [] for k in ("ep", "tok", "fmap", "rfmap", "g", "rg", "occ", "hop",
                         "headroom", "loading", "pocc", "focc")}
    for ei, row in enumerate(picks):
        state, info = state_from_row(row)
        obs, _ = env.reset(options={"state": state, "task_info": info})
        for t in range(args.max_steps):
            batch = {Columns.OBS: {k: torch.from_numpy(np.asarray(v)).unsqueeze(0)
                                   for k, v in obs.items()}}
            with torch.no_grad():
                g, fmap = m._features(batch)
                logits = m._logits(g, fmap)
            if t % args.stride == 0:
                with torch.no_grad():
                    rg, rfmap = rnd._features(batch)
                occ = occ3d_labels(rr, spec, G, G)
                cell_p = torch.softmax(logits[0, n_verb:n_verb + n_cell].float(), -1)
                cell_p = cell_p.reshape(G, n_panes * G)[:, (n_panes - 1) * G:].numpy()
                cell_p = cell_p / max(cell_p.sum(), 1e-12)
                st = rr._state
                ids = S.visible_segments(st["segments"])
                zmaxes = [rr.mesh_zmax(r) for r in ids]
                zmaxes = [z for z in zmaxes if z is not None]
                D["ep"].append(ei)
                D["tok"].append(np.asarray(obs["patch_features"], np.float16))
                D["fmap"].append(fmap.reshape(n_panes, -1, G, G).float().numpy().astype(np.float16))
                D["rfmap"].append(rfmap.reshape(n_panes, -1, G, G).float().numpy().astype(np.float16))
                D["g"].append(g[0].float().numpy()); D["rg"].append(rg[0].float().numpy())
                D["occ"].append(occ)
                D["hop"].append(hop2d_labels(rr, G, G) if n_panes == 2 else np.zeros((G, G), bool))
                D["headroom"].append((max(zmaxes) - float(st["position"][2])) if zmaxes else np.nan)
                D["loading"].append(bool(rr._mesh_futs) or any(not rr._renderer.has_mesh(r) for r in ids))
                D["pocc"].append(float(cell_p[occ].sum())); D["focc"].append(float(occ.mean()))
            a = pol.dist_cls.from_logits(logits).sample().squeeze(0).cpu().numpy()
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        print(f"[repr] episode {ei + 1}/{len(picks)} idx={int(row['idx'])} samples={len(D['ep'])}", flush=True)

    ep = np.array(D["ep"]); test = ep >= int(round(len(picks) * 0.75)); train = ~test
    tok = np.stack(D["tok"]).astype(np.float32)           # (N, P, G, G, 384)
    fmap = np.stack(D["fmap"]).astype(np.float32)          # (N, P, C, G, G)
    rfmap = np.stack(D["rfmap"]).astype(np.float32)
    occ = np.stack(D["occ"]); hop = np.stack(D["hop"])
    p3 = n_panes - 1

    def per_loc(feats_nlocd, labels_nloc, name, what):
        X = feats_nlocd; y = labels_nloc
        s = fit_logistic(X[train].reshape(-1, X.shape[-1]), y[train].reshape(-1),
                         X[test].reshape(-1, X.shape[-1]))
        print(f"[repr] {what:<22} {name:<8} held-out AUC {auc(y[test].reshape(-1), s):.3f} "
              f"(pos rate {y[test].mean():.2f})", flush=True)

    tok3 = tok[:, p3].reshape(len(tok), G * G, -1)
    fm3 = fmap[:, p3].transpose(0, 2, 3, 1).reshape(len(fmap), G * G, -1)
    rf3 = rfmap[:, p3].transpose(0, 2, 3, 1).reshape(len(rfmap), G * G, -1)
    occ_l = occ.reshape(len(occ), G * G)
    print("[repr] === per-location probes: 3D cell occupancy (a click there hits a mesh) ===")
    per_loc(tok3, occ_l, "inputs", "occ3d")
    per_loc(fm3, occ_l, "trained", "occ3d")
    per_loc(rf3, occ_l, "random", "occ3d")
    if n_panes == 2:
        tok2 = tok[:, 0].reshape(len(tok), G * G, -1)
        fm2 = fmap[:, 0].transpose(0, 2, 3, 1).reshape(len(fmap), G * G, -1)
        rf2 = rfmap[:, 0].transpose(0, 2, 3, 1).reshape(len(rfmap), G * G, -1)
        hop_l = hop.reshape(len(hop), G * G)
        print("[repr] === per-location probes: 2D hop target (unselected segment) ===")
        per_loc(tok2, hop_l, "inputs", "hop2d")
        per_loc(fm2, hop_l, "trained", "hop2d")
        per_loc(rf2, hop_l, "random", "hop2d")

    print("[repr] === global probes from the shared vector g (vs mean-pooled tokens) ===")
    gv, rgv = np.stack(D["g"]), np.stack(D["rg"])
    pooled = tok.mean(axis=(2, 3)).reshape(len(tok), -1)
    head = np.array(D["headroom"], dtype=np.float64); okh = ~np.isnan(head)
    load = np.array(D["loading"], bool)
    for name, X in (("inputs", pooled), ("trained", gv), ("random", rgv)):
        r2 = fit_ridge(X[train & okh], head[train & okh], X[test & okh], head[test & okh])
        s = fit_logistic(X[train], load[train].astype(np.float32), X[test])
        print(f"[repr] {name:<8} headroom R^2 {r2:.3f} | mesh-loading AUC {auc(load[test], s):.3f}", flush=True)

    pocc, focc = np.array(D["pocc"]), np.array(D["focc"])
    ok = (focc > 0) & (focc < 1)
    print("[repr] === aim lift (3D pane): policy click mass on occupied cells / occupied fraction ===")
    print(f"[repr] steps with partial occupancy: {ok.sum()} / {len(focc)}; "
          f"mean occupied fraction {focc[ok].mean():.3f}; mean policy mass on occupied {pocc[ok].mean():.3f}; "
          f"lift {pocc[ok].mean() / focc[ok].mean():.2f}x (median per-step {np.median(pocc[ok] / focc[ok]):.2f}x)", flush=True)
    for flag, nm in ((True, "loading"), (False, "settled")):
        k = ok & (load == flag)
        if k.any():
            print(f"[repr]   {nm:<8} n={k.sum()} occ frac {focc[k].mean():.3f} policy mass {pocc[k].mean():.3f} "
                  f"lift {pocc[k].mean() / focc[k].mean():.2f}x", flush=True)
    print("REPR-DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
