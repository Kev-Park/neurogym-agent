"""Input parity: browser (Chrome+NG) vs native, empirically.

The env's full input surface is MultiDiscrete [verb, cell, rx, ry, rz, zoom]:
  verb 0  right-click on a 3D-pane cell -> NG "move-to-mouse-position"
          (mousedown2 binding): position := picked surface point, no-op on
          background.
  verb 1  rotate: ngllib state edit — euler += delta, euler->quat roundtrip.
  verb 2  zoom:   projectionScale = min(500000, ps + delta).
(Keyboard input is not part of the action space; NG keyboard bindings are
out of scope until the action space grows.)

--mode browser  (MAIN venv, GPU node): for N provider-sampled states, reset
    the browser env and apply each probe action from a fresh reset; record
    observed post-reset and post-action viewer state. Ground truth JSONL.
--mode native   (SPIKE venv, GPU node): replay the JSONL. Clicks: render the
    mesh depth buffer at the observed reset state, unproject the clicked
    pixel (exact px, and nearest-hit within a 3px pick radius), predict the
    new position; compare to browser in voxels. Rotate/zoom: ngllib's exact
    arithmetic (geom.py file-loaded); compare numerically.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import sys

import numpy as np

PANE, TOOLBAR = 450, 17
PANE_H = PANE - TOOLBAR
VOXEL_NM = np.array([4.0, 4.0, 40.0])

CLICK_CELLS = [528, 264, 792, 0, 1023]  # (r16,c16),(r8,c8),(r24,c24),corners
ROT_ACTION = [1, 0, 5, 2, 4, 4]         # euler delta (+0.08, -0.16, 0)
ZOOM_ACTIONS = [[2, 0, 4, 4, 4, 8], [2, 0, 4, 4, 4, 0]]  # +2000 / -2000


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def obs_state(obs):
    return {
        "position": np.asarray(obs["position"]).tolist(),
        "orientation": np.asarray(obs["orientation"]).tolist(),
        "xs_scale": float(np.asarray(obs["xs_scale"]).ravel()[0]),
        "proj_scale": float(np.asarray(obs["proj_scale"]).ravel()[0]),
    }


def mode_browser(args):
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    import pyarrow.parquet as pq

    from ngllib_agent.env_build import build_env, load_config
    from ngllib_agent.providers import FlywireSkeletonProvider

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = "raw"  # 84x84 obs is fine: only state numbers matter
    env = build_env(cfg)

    class T(Exception):
        pass

    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(T()))
    ec = cfg["env"]
    psr = ec.get("projection_scale_range")
    provider = FlywireSkeletonProvider(
        ec["parquet_path"],
        projection_scale_range=tuple(psr) if psr else None)
    pool = pq.read_table(args.pool).to_pylist()
    rng = np.random.default_rng(11)
    idx = rng.choice(len(pool), size=args.n_states, replace=False)

    probes = ([{"kind": "click", "action": [0, c, 4, 4, 4, 4]}
               for c in CLICK_CELLS]
              + [{"kind": "rotate", "action": ROT_ACTION}]
              + [{"kind": "zoom", "action": z} for z in ZOOM_ACTIONS])

    records = []
    for k, i in enumerate(idx):
        rid = str(pool[int(i)]["root_id"])
        state, ti = provider(rng, {"segment_id": rid})
        rec = {"idx": k, "root_id": rid, "requested_state": state,
               "probes": []}
        for p in probes:
            signal.alarm(420)
            try:
                obs, _ = env.reset(options={"state": state, "task_info": ti})
                pre = obs_state(obs)
                obs2, _, _, _, _ = env.step(np.array(p["action"]))
                post = obs_state(obs2)
            except T:
                print(f"[input] state {k} probe wedged; env rebuild",
                      flush=True)
                signal.alarm(30)
                try:
                    env.close()
                except Exception:
                    pass
                finally:
                    signal.alarm(0)
                env = build_env(cfg)
                continue
            finally:
                signal.alarm(0)
            rec["probes"].append({**p, "pre": pre, "post": post})
        records.append(rec)
        print(f"[input] state {k + 1}/{args.n_states} ({len(rec['probes'])} probes)",
              flush=True)
        with open(args.out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    env.close()
    print(f"[input] wrote {len(records)} states -> {args.out}", flush=True)
    return 0


def mode_native(args):
    import moderngl

    NGL = "/scratch/kp0374/wt/neurogym-native/src/ngllib"
    camera = _load("cam", f"{NGL}/native/camera.py")
    geom = _load("geom", f"{NGL}/utils/geom.py")
    from cloudvolume import CloudVolume

    ctx = moderngl.create_context(standalone=True, backend="egl")
    ctx.enable(moderngl.DEPTH_TEST)
    color = ctx.texture((PANE, PANE_H), 4)
    depth = ctx.depth_texture((PANE, PANE_H))
    fbo = ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
    prog = ctx.program(
        vertex_shader="""#version 330
            uniform mat4 mvp; in vec3 pos;
            void main() { gl_Position = mvp * vec4(pos, 1.0); }""",
        fragment_shader="""#version 330
            out vec4 f; void main() { f = vec4(1.0); }""")

    vol = CloudVolume("precomputed://gs://flywire_v141_m783", use_https=True,
                      cache="/scratch/kp0374/native_spike/cv_cache",
                      progress=False)
    vaos = {}

    def load_mesh(rid):
        if rid in vaos:
            return vaos[rid]
        m = vol.mesh.get(int(rid))
        mesh = m[int(rid)] if hasattr(m, "get") or isinstance(m, dict) else m
        v = np.asarray(mesh.vertices, dtype="f4")
        f = np.asarray(mesh.faces, dtype="i4")
        vbo = ctx.buffer(v.tobytes())
        ibo = ctx.buffer(f.tobytes())
        vaos[rid] = ctx.vertex_array(prog, [(vbo, "3f", "pos")],
                                     index_buffer=ibo)
        return vaos[rid]

    records = [json.loads(l) for l in open(args.browser_jsonl)]
    click_err, radius_err, noop_ok, noop_n = [], [], 0, 0
    click_along, click_lat = [], []
    rot_err, zoom_err = [], []

    for rec in records:
        for p in rec["probes"]:
            pre = p["pre"]
            if p["kind"] == "rotate":
                d = [0.08, -0.16, 0.0]
                pred = np.asarray(geom.quaternion_to_euler(
                    geom.euler_to_quaternion(
                        [pre["orientation"][i] + d[i] for i in range(3)])))
                err = np.abs(pred - np.asarray(p["post"]["orientation"]))
                err = np.minimum(err, 2 * np.pi - err)  # wraparound
                rot_err.append(float(err.max()))
                continue
            if p["kind"] == "zoom":
                dz = (p["action"][5] - 4) * 500.0
                pred = min(500_000.0, pre["proj_scale"] + dz)
                zoom_err.append(abs(pred - p["post"]["proj_scale"]))
                continue
            # click: depth-unproject at the observed pre state
            pos_nm = np.asarray(pre["position"]) * VOXEL_NM
            quat = rec["requested_state"]["projectionOrientation"]
            zoom_nm = pre["proj_scale"] * 4.07
            view, proj = camera.projection_camera(
                pos_nm, quat, zoom_nm, PANE, PANE_H)
            mvp = (proj @ view).astype("f4")
            fbo.use()
            fbo.clear(0, 0, 0, 1)
            prog["mvp"].write(mvp.T.copy().tobytes())
            load_mesh(rec["root_id"]).render(mode=4)
            draw = np.frombuffer(depth.read(), dtype="f4").reshape(
                PANE_H, PANE)[::-1]
            cell = p["action"][1]
            row, col = cell // 32, cell % 32
            x_css = 900 + (col + 0.5) * (900 / 32)
            y_css = (row + 0.5) * (900 / 32)
            fx = x_css * 0.5 - PANE
            fy = y_css * 0.5 - TOOLBAR
            ix, iy = int(round(fx)), int(round(fy))

            def unproject(px, py, dval):
                ndc = np.array([2 * (px + 0.5) / PANE - 1,
                                1 - 2 * (py + 0.5) / PANE_H,
                                2 * dval - 1, 1.0])
                w = np.linalg.inv(proj @ view) @ ndc
                return (w[:3] / w[3]) / VOXEL_NM

            browser_moved = (np.linalg.norm(
                np.asarray(p["post"]["position"])
                - np.asarray(pre["position"])) > 0.5)
            # view direction in voxel space, for error decomposition
            vdir_nm = np.linalg.inv(view)[:3, 2]
            vdir_vox = (vdir_nm / VOXEL_NM)
            vdir_vox /= np.linalg.norm(vdir_vox) + 1e-9
            d0 = draw[iy, ix] if 0 <= iy < PANE_H and 0 <= ix < PANE else 1.0
            # pick radius 3px: nearest (front-most) hit in the window
            y0, y1 = max(0, iy - 3), min(PANE_H, iy + 4)
            x0, x1 = max(0, ix - 3), min(PANE, ix + 4)
            win = draw[y0:y1, x0:x1]
            hit = win < 0.9999
            if d0 < 0.9999:
                pr = unproject(ix, iy, d0)
                dvec = pr - np.asarray(p["post"]["position"])
                err = np.linalg.norm(dvec)
                along = float(abs(dvec @ vdir_vox))
                lateral = float(np.sqrt(max(0.0, err ** 2 - along ** 2)))
                if browser_moved:
                    click_err.append(float(err))
                    click_along.append(along)
                    click_lat.append(lateral)
            elif hit.any():
                yy, xx = np.unravel_index(np.argmin(win), win.shape)
                pr = unproject(x0 + xx, y0 + yy, win[yy, xx])
                err = np.linalg.norm(pr - np.asarray(p["post"]["position"]))
                if browser_moved:
                    radius_err.append(float(err))
                else:
                    noop_n += 1
            else:
                noop_n += 1
                noop_ok += int(not browser_moved)

    def stats(name, arr):
        if not arr:
            print(f"[parity] {name}: no samples", flush=True)
            return
        a = np.array(arr)
        print(f"[parity] {name}: n={len(a)} median {np.median(a):.2f} "
              f"p90 {np.percentile(a, 90):.2f} max {a.max():.2f}", flush=True)

    stats("click position error (voxels, exact px)", click_err)
    stats("  - along view axis", click_along)
    stats("  - lateral (screen)", click_lat)
    stats("click position error (voxels, 3px radius)", radius_err)
    print(f"[parity] background no-op agreement: {noop_ok}/{noop_n}", flush=True)
    stats("rotate euler error (rad)", rot_err)
    stats("zoom proj_scale error", zoom_err)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["browser", "native"], required=True)
    ap.add_argument("--config", default="configs/ppo_zmax_navigate.yaml")
    ap.add_argument("--pool", default="eval_d0_v1.parquet")
    ap.add_argument("--n-states", type=int, default=40)
    ap.add_argument("--out", default="/scratch/kp0374/native_spike/input_browser.jsonl")
    ap.add_argument("--browser-jsonl",
                    default="/scratch/kp0374/native_spike/input_browser.jsonl")
    args = ap.parse_args()
    return mode_browser(args) if args.mode == "browser" else mode_native(args)


if __name__ == "__main__":
    sys.exit(main())
