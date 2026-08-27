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

# Load the ngllib.native modules directly by file: a package import would
# execute ngllib/__init__ (gymnasium/playwright), which the spike venv
# intentionally does not carry.
import importlib.util  # noqa: E402

_NGL_NATIVE = "/scratch/kp0374/wt/neurogym-native/src/ngllib/native"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


projection_camera = _load("ngl_native_camera",
                          f"{_NGL_NATIVE}/camera.py").projection_camera
segment_color = _load("ngl_native_colors",
                      f"{_NGL_NATIVE}/colors.py").segment_color

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
        """Indexed draw with smooth per-vertex normals.

        VRAM discipline (native-renderer design constraint): buffers are
        explicit, sized exactly (verts*24B + faces*12B — ~3x smaller than
        tri-soup), shared across all envs in the process, and releasable
        (`vbo.release()`) under an LRU byte budget in the real renderer —
        no growth-until-restart, unlike Chrome's opaque GPU caches.
        """
        v = np.asarray(vertices_nm, dtype="f4")
        f = np.asarray(faces, dtype="i4")
        e1 = v[f[:, 1]] - v[f[:, 0]]
        e2 = v[f[:, 2]] - v[f[:, 0]]
        fn = np.cross(e1, e2)
        vn = np.zeros_like(v)
        for k in range(3):
            np.add.at(vn, f[:, k], fn)
        vn /= (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9)
        vbo = self.ctx.buffer(np.hstack([v, vn.astype("f4")]).tobytes())
        ibo = self.ctx.buffer(f.tobytes())
        vao = self.ctx.vertex_array(
            self.prog, [(vbo, "3f 3f", "pos", "nrm")], index_buffer=ibo)
        self._vaos[rid] = (vao, vbo.size + ibo.size)

    def render(self, rid: str, position_nm, quat, zoom_nm,
               conj: bool = False) -> np.ndarray:
        q = list(quat)
        if conj:
            q = [-q[0], -q[1], -q[2], q[3]]
        view, proj = projection_camera(position_nm, q, zoom_nm, PANE, PANE)
        mvp = (proj @ view).astype("f4")
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.prog["mvp"].write(mvp.T.copy().tobytes())  # column-major
        self.prog["color"].value = segment_color(int(rid))
        vao, _ = self._vaos[rid]
        vao.render(mode=4)
        px = np.frombuffer(self.fbo.read(components=4), dtype=np.uint8)
        return px.reshape(PANE, PANE, 4)[::-1, :, :3]  # GL origin flip


def color_silhouette(img_rgb: np.ndarray, expected_rgb, margin: int = 16,
                     tol: float = 0.35) -> np.ndarray:
    """Mask of pixels whose chromaticity matches the segment color.

    Chromaticity (channel proportions) is shading-invariant for our diffuse
    model (color * luminance), and rejects the browser pane's grey overlays
    (toolbar, section plane, chips) and colored axis lines. `margin` rows are
    cleared top/bottom to drop NG's toolbar strip and bottom chip.
    """
    img = img_rgb.astype(np.float32)
    lum = img.sum(axis=2)
    bright = lum > 40.0
    chroma = img / (lum[..., None] + 1e-6)
    exp = np.asarray(expected_rgb, dtype=np.float32)
    exp = exp / (exp.sum() + 1e-6)
    dist = np.abs(chroma - exp[None, None, :]).sum(axis=2)
    mask = bright & (dist < tol)
    mask[:margin] = False
    mask[-margin:] = False
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def centroid_delta(a: np.ndarray, b: np.ndarray):
    """(dy, dx) between mask centroids, in px; None if either mask empty."""
    if not a.any() or not b.any():
        return None
    ca = np.argwhere(a).mean(axis=0)
    cb = np.argwhere(b).mean(axis=0)
    return (float(ca[0] - cb[0]), float(ca[1] - cb[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calib-states", type=int, default=12)
    ap.add_argument("--scale-grid",
                    default="3.0,3.5,4.0,4.5,5.0,5.5,6.0,7.0",
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

    def native_pane(rec, scale_cal, conj=False):
        st = rec["requested_state"]
        pos_nm = np.asarray(rec["observed_a"]["position"]) * VOXEL_NM
        quat = st["projectionOrientation"]
        zoom_nm = rec["observed_a"]["proj_scale"] * scale_cal
        return rend.render(rec["root_id"], pos_nm, quat, zoom_nm, conj=conj)

    def masks(rec, scale_cal, conj):
        col = segment_color(int(rec["root_id"]))
        nat = color_silhouette(native_pane(rec, scale_cal, conj), col, margin=0)
        bro = color_silhouette(browser_pane(rec), col, margin=16)
        return nat, bro

    # ---- calibration: units factor x quaternion convention ---------------
    grid = [float(x) for x in args.scale_grid.split(",")]
    calib = records[: args.calib_states]
    best = None
    for conj in (False, True):
        for sc in grid:
            vals = [iou(*masks(r, sc, conj)) for r in calib]
            mean = float(np.mean(vals))
            print(f"[calib] conj={conj} scale_cal={sc}: mean IoU {mean:.3f}",
                  flush=True)
            if best is None or mean > best[2]:
                best = (conj, sc, mean)
    conj, sc, _ = best
    print(f"[calib] chosen conj={conj} scale_cal={sc}", flush=True)

    # ---- full scoring + side-by-sides ------------------------------------
    metrics = []
    deltas = []
    for k, rec in enumerate(records):
        nat, bro = masks(rec, sc, conj)
        v = iou(nat, bro)
        d = centroid_delta(nat, bro)
        if d:
            deltas.append(d)
        metrics.append({"idx": rec["idx"], "root_id": rec["root_id"],
                        "iou": round(v, 4),
                        "centroid_dydx": [round(x, 1) for x in d] if d else None})
        if k < 24:
            side = np.concatenate([native_pane(rec, sc, conj),
                                   browser_pane(rec)], axis=1)
            Image.fromarray(side).save(
                os.path.join(args.out_dir, f"pair_{rec['idx']:04d}.png"))
    ious = np.array([m["iou"] for m in metrics])
    print(f"[render] color-silhouette IoU over {len(ious)} states: "
          f"mean {ious.mean():.3f} median {np.median(ious):.3f} "
          f"p10 {np.percentile(ious, 10):.3f} min {ious.min():.3f}",
          flush=True)
    if deltas:
        dd = np.array(deltas)
        print(f"[render] centroid delta (native-browser) px: "
              f"dy median {np.median(dd[:, 0]):+.1f}  "
              f"dx median {np.median(dd[:, 1]):+.1f}", flush=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"scale_cal": sc, "conj": conj, "per_state": metrics},
                  f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
