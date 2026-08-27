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
PANE = 450        # capture_scale 0.5 of the 900px pane width
TOOLBAR = 17      # NG top toolbar (~33 CSS px) at capture scale 0.5
PANE_H = PANE - TOOLBAR  # the 3D panel's true captured height (433)


class MeshRenderer:
    """3D-projection pane: mesh + axis lines + EM section plane.

    Parity sources (google/neuroglancer master, 2026-08-27):
    - mesh/frontend.ts + perspective_view/panel.ts: Gouraud lighting
      factor = |dot(n, l)| * 0.8 + 0.2, light = -(R(q) @ z) (headlight).
    - axes_lines.ts + panel.ts: three lines through position, colors pure
      R/G/B alpha 0.5, width 1px, half-length = zoom * min(w,h)/h/4.
    - panel.ts drawSliceViews: the cross-section plane is the EM slice
      textured onto its plane, modulated by the same lighting factor.
    """

    def __init__(self):
        import moderngl

        self._moderngl = moderngl
        self.ctx = moderngl.create_context(standalone=True, backend="egl")
        print(f"[render] GL_RENDERER = {self.ctx.info['GL_RENDERER']}",
              flush=True)
        self.ctx.enable(moderngl.DEPTH_TEST)
        # Render at the 3D panel's true captured geometry (450 x 433 below
        # the toolbar) so the optical center and aspect match the browser.
        self.fbo = self.ctx.simple_framebuffer((PANE, PANE_H), components=4)
        self.prog = self.ctx.program(
            vertex_shader="""#version 330
                uniform mat4 mvp;
                uniform vec4 light;   // xyz dir (pre-scaled 0.8), w ambient
                in vec3 pos; in vec3 nrm;
                out float v_l;
                void main() {
                    gl_Position = mvp * vec4(pos, 1.0);
                    v_l = abs(dot(normalize(nrm), light.xyz)) + light.w;
                }""",
            fragment_shader="""#version 330
                uniform vec3 color;
                in float v_l;
                out vec4 frag;
                void main() { frag = vec4(color * v_l, 1.0); }""",
        )
        self.line_prog = self.ctx.program(
            vertex_shader="""#version 330
                uniform mat4 mvp;
                in vec3 pos; in vec4 col;
                out vec4 v_c;
                void main() {
                    gl_Position = mvp * vec4(pos, 1.0);
                    v_c = col;
                }""",
            fragment_shader="""#version 330
                in vec4 v_c; out vec4 frag;
                void main() { frag = v_c; }""",
        )
        self.plane_prog = self.ctx.program(
            vertex_shader="""#version 330
                uniform mat4 mvp;
                in vec3 pos; in vec2 uv;
                out vec2 v_uv;
                void main() {
                    gl_Position = mvp * vec4(pos, 1.0);
                    v_uv = uv;
                }""",
            fragment_shader="""#version 330
                uniform sampler2D em;
                uniform float lfac;
                uniform float alpha;
                in vec2 v_uv; out vec4 frag;
                void main() {
                    float g = texture(em, v_uv).r * lfac;
                    frag = vec4(g, g, g, alpha);
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

    @staticmethod
    def _rot(q):
        x, y, z, w = q
        n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
        x, y, z, w = x / n, y / n, z / n, w / n
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def render(self, rid: str, position_nm, quat, zoom_nm,
               conj: bool = False, em_tile=None, em_extent_nm=None,
               overlays: bool = True) -> np.ndarray:
        q = list(quat)
        if conj:
            q = [-q[0], -q[1], -q[2], q[3]]
        view, proj = projection_camera(position_nm, q, zoom_nm, PANE, PANE_H)
        mvp = (proj @ view).astype("f4")
        mvp_b = mvp.T.copy().tobytes()  # column-major
        pos = np.asarray(position_nm, dtype="f4")
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)

        # Lighting: headlight along -(R(q) @ z), NG panel.ts.
        ldir = -(self._rot(q) @ np.array([0.0, 0.0, 1.0]))
        ldir /= np.linalg.norm(ldir) + 1e-9
        light = (*(ldir * 0.8), 0.2)

        self.prog["mvp"].write(mvp_b)
        self.prog["light"].value = tuple(float(v) for v in light)
        self.prog["color"].value = segment_color(int(rid))
        vao, _ = self._vaos[rid]
        vao.render(mode=4)

        # Section plane: NG's quad is the CROSS-SECTION VIEWPORT's square
        # (crossSectionScale x 900px x 4nm per side — small vs the projection
        # view), rendered bright and opaque with normal depth. Iteration 6's
        # "occlusion regression" was the quad being frustum-sized, not its
        # opacity.
        if overlays and em_tile is not None:
            tex = self.ctx.texture(em_tile.shape[::-1], 1,
                                   np.ascontiguousarray(em_tile).tobytes())
            tex.use(0)
            h = em_extent_nm / 2.0
            quad = np.array([
                pos[0] - h, pos[1] - h, pos[2], 0, 0,
                pos[0] + h, pos[1] - h, pos[2], 1, 0,
                pos[0] - h, pos[1] + h, pos[2], 0, 1,
                pos[0] + h, pos[1] + h, pos[2], 1, 1,
            ], dtype="f4")
            vbo = self.ctx.buffer(quad.tobytes())
            vao = self.ctx.vertex_array(
                self.plane_prog, [(vbo, "3f 2f", "pos", "uv")])
            self.plane_prog["mvp"].write(mvp_b)
            # plane normal = +z; NG: factor = ambient + |dot(l, n)| * 0.8
            self.plane_prog["lfac"].value = float(0.2 + abs(ldir[2]) * 0.8)
            self.plane_prog["alpha"].value = 1.0
            vao.render(mode=5)  # TRIANGLE_STRIP
            vao.release(); vbo.release(); tex.release()

        # Axis lines: half-length = zoom * min(w,h)/h / 4 (panel.ts).
        if overlays:
            al = zoom_nm * (min(PANE, PANE_H) / PANE_H) / 4.0
            self.ctx.enable(self._moderngl.BLEND)
            verts = []
            for i, col in enumerate([(1, 0, 0, 0.5), (0, 1, 0, 0.5),
                                     (0, 0, 1, 0.5)]):
                a = np.zeros(3); a[i] = al
                verts += [*(pos - a), *col, *(pos + a), *col]
            vbo = self.ctx.buffer(np.array(verts, dtype="f4").tobytes())
            vao = self.ctx.vertex_array(
                self.line_prog, [(vbo, "3f 4f", "pos", "col")])
            self.line_prog["mvp"].write(mvp_b)
            vao.render(mode=1)  # LINES
            vao.release(); vbo.release()
            self.ctx.disable(self._moderngl.BLEND)

        px = np.frombuffer(self.fbo.read(components=4), dtype=np.uint8)
        pane = px.reshape(PANE_H, PANE, 4)[::-1, :, :3]  # GL origin flip
        # Paste below a black toolbar strip: same 450x450 layout as browser.
        out = np.zeros((PANE, PANE, 3), dtype=np.uint8)
        out[TOOLBAR:, :] = pane
        return out


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
    if margin > 0:  # note: mask[-0:] would clear the WHOLE array
        mask[:margin] = False
        mask[-margin:] = False
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


class EMTiles:
    """z-slice tiles around a position from the public FlyWire EM volume.

    Real data starts at mip1 (8x8x40nm; mip0 is a placeholder). The mip is
    chosen per request so the tile stays <= max_px, mirroring NG's use of
    coarse mips for the 3D section plane.
    """

    URL = "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14"
    RES_XY = [8, 16, 32, 64, 128]  # nm, mips 1..5

    def __init__(self, cache_dir):
        from cloudvolume import CloudVolume

        self._vols = {}
        self._CloudVolume = CloudVolume
        self._cache = cache_dir

    def _vol(self, mip):
        if mip not in self._vols:
            self._vols[mip] = self._CloudVolume(
                self.URL, mip=mip, use_https=True, cache=self._cache,
                progress=False, fill_missing=True, bounded=False)
        return self._vols[mip]

    def tile(self, pos_nm, extent_nm, max_px=512):
        mip = 1
        for i, r in enumerate(self.RES_XY):
            mip = i + 1
            if extent_nm / r <= max_px:
                break
        res = self.RES_XY[mip - 1]
        vol = self._vol(mip)
        cx, cy = int(pos_nm[0] / res), int(pos_nm[1] / res)
        z = int(pos_nm[2] / 40.0)
        half = int(extent_nm / res / 2)
        cut = vol[cx - half:cx + half, cy - half:cy + half, z:z + 1]
        img = np.asarray(cut)[:, :, 0, 0].T.astype(np.uint8)  # row=y, col=x
        return img


def block_ssim(a: np.ndarray, b: np.ndarray, block: int = 16) -> float:
    """Mean SSIM over non-overlapping blocks of the grayscale images."""
    ga = a.mean(axis=2).astype(np.float64)
    gb = b.mean(axis=2).astype(np.float64)
    H = (ga.shape[0] // block) * block
    W = (ga.shape[1] // block) * block
    ga = ga[:H, :W].reshape(H // block, block, W // block, block)
    gb = gb[:H, :W].reshape(H // block, block, W // block, block)
    ma, mb = ga.mean(axis=(1, 3)), gb.mean(axis=(1, 3))
    va, vb = ga.var(axis=(1, 3)), gb.var(axis=(1, 3))
    cov = (ga * gb).mean(axis=(1, 3)) - ma * mb
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / (
        (ma ** 2 + mb ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


def dilate(mask: np.ndarray, r: int = 2) -> np.ndarray:
    """Binary dilation by r px (separable max filter, numpy-only)."""
    out = mask.copy()
    for _ in range(r):
        out[1:] |= out[:-1]
        out[:-1] |= out[1:]
        out[:, 1:] |= out[:, :-1]
        out[:, :-1] |= out[:, 1:]
    return out


def tol_iou(a: np.ndarray, b: np.ndarray, r: int = 2) -> float:
    """Tolerance IoU: fraction of each mask within r px of the other —
    robust to antialiasing/shading edge noise on thin filaments."""
    if not a.any() or not b.any():
        return 0.0
    da, db = dilate(a, r), dilate(b, r)
    hit_a = np.logical_and(a, db).sum() / a.sum()
    hit_b = np.logical_and(b, da).sum() / b.sum()
    return float(min(hit_a, hit_b))


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

    em = EMTiles("/scratch/kp0374/native_spike/cv_cache")

    def native_pane(rec, scale_cal, conj=False, overlays=True):
        st = rec["requested_state"]
        pos_nm = np.asarray(rec["observed_a"]["position"]) * VOXEL_NM
        quat = st["projectionOrientation"]
        zoom_nm = rec["observed_a"]["proj_scale"] * scale_cal
        tile = ext = None
        if overlays:
            # NG slice quad side = crossSectionScale x 900 CSS px x 4nm.
            ext = rec["observed_a"]["xs_scale"] * 900.0 * 4.0
            try:
                tile = em.tile(pos_nm, ext, max_px=1024)
            except Exception as e:
                print(f"[render] EM tile fail ({e}); plane skipped", flush=True)
        return rend.render(rec["root_id"], pos_nm, quat, zoom_nm, conj=conj,
                           em_tile=tile, em_extent_nm=ext, overlays=overlays)

    def masks(rec, scale_cal, conj):
        # Calibration compares MESH-ONLY silhouettes (overlays off) — the
        # color mask excludes plane/axes on the browser side anyway.
        col = segment_color(int(rec["root_id"]))
        nat = color_silhouette(
            native_pane(rec, scale_cal, conj, overlays=False), col, margin=0)
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
    # Golden-section-ish refine around the grid winner (grid steps are ~12%
    # — too coarse: thin filaments halve IoU on a few-% scale error).
    lo, hi = sc * 0.85, sc * 1.15
    for _ in range(5):
        cands = np.linspace(lo, hi, 5)
        vals = [(float(np.mean([iou(*masks(r, s, conj)) for r in calib])), s)
                for s in cands]
        vals.sort(reverse=True)
        sc = vals[0][1]
        span = (hi - lo) / 4
        lo, hi = sc - span / 2, sc + span / 2
    print(f"[calib] chosen conj={conj} scale_cal={sc:.3f} (refined)", flush=True)

    # ---- full scoring + side-by-sides ------------------------------------
    metrics = []
    deltas = []
    for k, rec in enumerate(records):
        nat, bro = masks(rec, sc, conj)
        v = iou(nat, bro)
        tv = tol_iou(nat, bro, r=2)
        d = centroid_delta(nat, bro)
        if d:
            deltas.append(d)
        full_nat = native_pane(rec, sc, conj, overlays=True)
        full_bro = browser_pane(rec)
        ss = block_ssim(full_nat[TOOLBAR:], full_bro[TOOLBAR:])
        metrics.append({"idx": rec["idx"], "root_id": rec["root_id"],
                        "iou": round(v, 4), "tol_iou": round(tv, 4),
                        "ssim": round(ss, 4),
                        "centroid_dydx": [round(x, 1) for x in d] if d else None})
        if k < 24:
            side = np.concatenate([full_nat, full_bro], axis=1)
            Image.fromarray(side).save(
                os.path.join(args.out_dir, f"pair_{rec['idx']:04d}.png"))
    ious = np.array([m["iou"] for m in metrics])
    tious = np.array([m["tol_iou"] for m in metrics])
    print(f"[render] color-silhouette IoU over {len(ious)} states: "
          f"mean {ious.mean():.3f} median {np.median(ious):.3f} "
          f"p10 {np.percentile(ious, 10):.3f} min {ious.min():.3f}",
          flush=True)
    print(f"[render] tolerance-IoU (2px): mean {tious.mean():.3f} "
          f"median {np.median(tious):.3f} p10 {np.percentile(tious, 10):.3f}",
          flush=True)
    ssims = np.array([m["ssim"] for m in metrics])
    print(f"[render] full-pane block-SSIM (overlays on): mean {ssims.mean():.3f} "
          f"median {np.median(ssims):.3f} p10 {np.percentile(ssims, 10):.3f}",
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
