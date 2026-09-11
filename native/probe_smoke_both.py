"""Gate 3 (cluster smoke) + the shared state rules, both backends in one run.

Builds the SAME config through build_env with env.backend chrome and
simulator, resets each, steps a fixed script that touches every verb the
policy can emit (right-click, rotate, zoom, select), and checks:

  - observation_space / action_space are EQUAL across backends;
  - every obs is inside the space, with identical keys, shapes and dtypes;
  - info carries the same keys and json_state the same five NglState fields;
  - the shared zoom rule: a zoom that would reach projectionScale == 0 keeps
    the previous value on BOTH backends, and a legal zoom applies on both;
  - per-step wall time, so a regression in either path is visible.

    uv run --no-sync python native/probe_smoke_both.py --config configs/native_select.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def _script(spec):
    """[verb, cell, rx, ry, rz, zoom] actions: right-click centre of the 3D
    pane, rotate +x, zoom out, select at the 2D pane centre, rotate -y, zoom in."""
    mid = spec.rotation_bins_per_axis // 2
    zmid = spec.zoom_bins // 2
    cell_3d = (spec.grid_rows // 2) * spec.grid_cols + (3 * spec.grid_cols) // 4
    cell_2d = (spec.grid_rows // 2) * spec.grid_cols + spec.grid_cols // 4
    return [
        [0, cell_3d, mid, mid, mid, zmid],          # right-click (move-to) in 3D
        [1, 0, mid + 2, mid, mid, zmid],            # rotate +x
        [2, 0, mid, mid, mid, zmid + 2],            # zoom out
        [3, cell_2d, mid, mid, mid, zmid],          # select in the 2D pane
        [1, 0, mid, mid - 2, mid, zmid],            # rotate -y
        [2, 0, mid, mid, mid, zmid - 1],            # zoom in
        [0, cell_2d, mid, mid, mid, zmid],          # right-click in 2D
        [3, cell_2d, mid, mid, mid, zmid],          # select again (toggle back)
    ]


def _describe(obs):
    return {k: (tuple(v.shape), str(v.dtype)) for k, v in obs.items()}


def run_backend(cfg, backend, n_steps):
    from ngllib_agent.env_build import build_env

    cfg = json.loads(json.dumps(cfg))
    cfg["env"]["backend"] = backend
    env = build_env(cfg)
    spec = env.unwrapped if hasattr(env.unwrapped, "_action_spec") else None
    # MultiDiscreteActionWrapper sits right above the base env.
    w = env
    while not hasattr(w, "_action_spec"):
        w = w.env
    spec = w._action_spec
    t0 = time.monotonic()
    obs, info = env.reset(seed=0)
    t_reset = time.monotonic() - t0
    assert env.observation_space.contains(obs), f"{backend}: reset obs outside space"
    out = {"obs_space": str(env.observation_space), "act_space": str(env.action_space),
           "obs_desc": _describe(obs), "info_keys": sorted(info),
           "json_keys": sorted(info["json_state"]), "reset_s": t_reset, "step_s": [],
           "json_states": [info["json_state"]]}
    script = _script(spec)
    for i in range(n_steps):
        a = np.asarray(script[i % len(script)])
        t0 = time.monotonic()
        obs, r, term, trunc, info = env.step(a)
        out["step_s"].append(time.monotonic() - t0)
        assert env.observation_space.contains(obs), f"{backend}: step {i} obs outside space"
        assert _describe(obs) == out["obs_desc"], f"{backend}: obs shape drifted at step {i}"
        out["json_states"].append(info["json_state"])
        if term or trunc:
            obs, info = env.reset()

    # Shared zoom rule, driven through the public API: start at ps=1000, zoom
    # by -1000 (bin 0 at zoom_step 500 is -2000, so use zoom_step arithmetic).
    base = env.unwrapped
    st = dict(info["json_state"])
    st["projectionScale"] = 2 * spec.zoom_step          # 1000 at the default step
    obs, info = env.reset(options={"state": st, "task_info": info["task_info"]})
    zero_bin = spec.zoom_bins // 2 - 2                   # delta = -2 * zoom_step = -1000
    mid = spec.rotation_bins_per_axis // 2
    obs, *_ = env.step(np.asarray([2, 0, mid, mid, mid, zero_bin]))
    out["zoom_to_zero_ps"] = float(obs["proj_scale"][0])
    obs, *_ = env.step(np.asarray([2, 0, mid, mid, mid, spec.zoom_bins // 2 - 1]))
    out["zoom_legal_ps"] = float(obs["proj_scale"][0])
    env.close()
    del base
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/native_select.yaml")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--backends", default="chrome,simulator")
    args = ap.parse_args()

    from ngllib_agent.env_build import load_config

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = "raw"
    cfg["env"].update({"left_pane": True, "right_pane": True, "image_size": None,
                       "capture_scale": 0.5})

    results = {}
    for b in args.backends.split(","):
        print(f"\n=== {b} ===", flush=True)
        results[b] = run_backend(cfg, b, args.steps)
        r = results[b]
        print(f"reset {r['reset_s']:.2f}s  step median {np.median(r['step_s']) * 1000:.0f}ms "
              f"p90 {np.percentile(r['step_s'], 90) * 1000:.0f}ms", flush=True)
        print(f"obs: {r['obs_desc']}", flush=True)
        print(f"zoom->0 keeps ps: {r['zoom_to_zero_ps']}   legal zoom ps: {r['zoom_legal_ps']}",
              flush=True)

    ok = True
    if len(results) == 2:
        a, b = results.values()
        for k in ("obs_space", "act_space", "obs_desc", "info_keys", "json_keys"):
            same = a[k] == b[k]
            ok &= same
            print(f"{k:10s} identical across backends: {same}", flush=True)
        for k in ("zoom_to_zero_ps", "zoom_legal_ps"):
            same = abs(a[k] - b[k]) < 1e-6
            ok &= same
            print(f"{k:16s} {a[k]} vs {b[k]}  agree: {same}", flush=True)
        ok &= a["zoom_to_zero_ps"] == 1000.0 and b["zoom_to_zero_ps"] == 1000.0
        ok &= a["zoom_legal_ps"] == 500.0 and b["zoom_legal_ps"] == 500.0
    print("\nGATE3-SMOKE", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
