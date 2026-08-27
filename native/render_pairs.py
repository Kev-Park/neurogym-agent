"""Render collected states natively and score parity vs browser frames.

Phase A (this script): GEOMETRY parity of the 3D pane. For each collected
state: fetch the segment mesh (CloudVolume, cached), render it offscreen
(moderngl EGL) with the ported NG camera, and compare the mesh SILHOUETTE
against the browser frame's right pane via IoU — silhouettes isolate the
camera model from shading/color, which calibrate later (Phase B).

The two flagged unknowns in ngllib.native.camera (canonical-space units and
depth-range) are resolved empirically: a grid search over the projection-
scale calibration factor on the first K states picks the value maximizing
mean IoU; the rest of the set is scored at that value.

Spike venv + worktrees:
    /scratch/kp0374/native_spike/.venv/bin/python render_pairs.py \
        --pairs-dir /scratch/kp0374/native_spike/pairs_v1 \
        --out-dir  /scratch/kp0374/native_spike/render_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/scratch/kp0374/wt/neurogym-native/src")
from ngllib.native.camera import projection_camera  # noqa: E402
from ngllib.native.colors import segment_color  # noqa: E402

VOXEL_NM = np.array([4.0, 4.0, 40.0])
PANE = 450  # capture_scale 0.5 of the 900px pane


class MeshRenderer:
    def __init__(self):
        import moderngl

        self.ctx = moderngl.create_context(standalone=True, backend="egl")
        print(f"[render] GL_RENDERER = {self.ctx.info['GL_RENDERER']}",
              flush=True)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.fbo = self.ctx.simple_framebuffer((PANE, PANE), components=4)
        self.prog = self.ctx.program(
            vertex_shader="""#version 330
                uniform mat4 mvp;
                in vec3 pos; in vec3 nrm;
                out vec3 v_nrm;
                void main() {
                    gl_Position = mvp * vec4(pos, 1.0);
                    v_nrm = nrm;
                }""",
            fragment_shader="""#version 330
                uniform vec3 color;
                in vec3 v_nrm;
                out vec4 frag;
                void main() {
                    float l = 0.3 + 0.7 * abs(normalize(v_nrm).z);
                    frag = vec4(color * l, 1.0);
                }""",
        )
        self._vaos = {}

    def load_mesh(self, rid: str, vertices_nm, faces):
        v = np.asarray(vertices_nm, dtype="f4")
        f = np.asarray(faces, dtype="i4")
        tri = v[f.reshape(-1)]
        e1 = v[f[:, 1]] - v[f[:, 0]]
        e2 = v[f[:, 2]] - v[f[:, 0]]
        n = np.cross(e1, e2)
        n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        nrm = np.repeat(n, 3, axis=0).astype("f4")
        vbo = self.ctx.buffer(np.hstack([tri, nrm]).astype("f4").tobytes())
        self._vaos[rid] = (self.ctx.simple_vertex_array(
            self.prog, vbo, "pos", "nrm"), len(tri))

    def render(self, rid: str, position_nm, quat, zoom_nm) -> np.ndarray:
        view, proj = projection_camera(position_nm, quat, zoom_nm, PANE, PANE)
        mvp = (proj @ view).astype("f4")
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.prog["mvp"].write(mvp.T.copy().tobytes())  # column-major
        self.prog["color"].value = segment_color(int(rid))
        vao, _ = self._vaos[rid]
        vao.render(mode=4)
        px = np.frombuffer(self.fbo.read(components=4), dtype=np.uint8)
        return px.reshape(PANE, PANE, 4)[::-1, :, :3]  # GL origin flip


def silhouette(img_rgb: np.ndarray, thresh: int = 12) -> np.ndarray:
    return img_rgb.max(axis=2) > thresh


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calib-states", type=int, default=12)
    ap.add_argument("--scale-grid", default="2.0,3.0,4.0,5.0,6.0,8.0",
                    help="Candidate nm-per-projectionScale-unit factors.")
    args = ap.parse_args()

    from cloudvolume import CloudVolume

    records = [json.loads(l) for l in
               open(os.path.join(args.pairs_dir, "states.jsonl"))]
    if args.limit:
        records = records[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    vol = CloudVolume("precomputed://gs://flywire_v141_m783", use_https=True,
                      cache="/scratch/kp0374/native_spike/cv_cache",
                      progress=False)
    rend = MeshRenderer()
    for rid in sorted({r["root_id"] for r in records}):
        m = vol.mesh.get(int(rid))
        mesh = m[int(rid)] if hasattr(m, "get") or isinstance(m, dict) else m
        rend.load_mesh(rid, mesh.vertices, mesh.faces)
        print(f"[render] mesh {rid}: {len(mesh.vertices)} verts", flush=True)

    def browser_pane(rec):
        img = np.asarray(Image.open(os.path.join(
            args.pairs_dir, "frames", f"{rec['idx']:04d}_a.png")))
        return img[:, img.shape[1] // 2:, :3]  # right pane

    def native_pane(rec, scale_cal):
        st = rec["requested_state"]
        pos_nm = np.asarray(rec["observed_a"]["position"]) * VOXEL_NM
        quat = st["projectionOrientation"]
        zoom_nm = rec["observed_a"]["proj_scale"] * scale_cal
        return rend.render(rec["root_id"], pos_nm, quat, zoom_nm)

    # ---- calibration: pick the units factor by silhouette IoU -------------
    grid = [float(x) for x in args.scale_grid.split(",")]
    calib = records[: args.calib_states]
    best = None
    for sc in grid:
        vals = [iou(silhouette(native_pane(r, sc)),
                    silhouette(browser_pane(r))) for r in calib]
        mean = float(np.mean(vals))
        print(f"[calib] scale_cal={sc}: mean IoU {mean:.3f}", flush=True)
        if best is None or mean > best[1]:
            best = (sc, mean)
    sc, _ = best
    print(f"[calib] chosen scale_cal={sc}", flush=True)

    # ---- full scoring + side-by-sides ------------------------------------
    metrics = []
    for k, rec in enumerate(records):
        nat = native_pane(rec, sc)
        bro = browser_pane(rec)
        v = iou(silhouette(nat), silhouette(bro))
        metrics.append({"idx": rec["idx"], "root_id": rec["root_id"],
                        "iou": round(v, 4)})
        if k < 24:
            side = np.concatenate([bro, nat], axis=1)
            Image.fromarray(side).save(
                os.path.join(args.out_dir, f"pair_{rec['idx']:04d}.png"))
    ious = np.array([m["iou"] for m in metrics])
    print(f"[render] silhouette IoU over {len(ious)} states: "
          f"mean {ious.mean():.3f} median {np.median(ious):.3f} "
          f"p10 {np.percentile(ious, 10):.3f} min {ious.min():.3f}",
          flush=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"scale_cal": sc, "per_state": metrics}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
