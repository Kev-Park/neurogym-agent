"""Feasibility probe for the native renderer stack, on one GPU node.

Validates, in order:
  1. moderngl EGL headless context on the NVIDIA stack (vendor string must
     be NVIDIA, not llvmpipe/softpipe — the SwiftShader trap, native
     edition) + render/readback throughput at pane size.
  2. CloudVolume EM cutout from the public FlyWire precomputed source.
  3. CloudVolume sharded-Draco mesh fetch for eval-pool root_ids.

Runs in the SPIKE venv (/scratch/kp0374/native_spike/.venv) — separate from
the training venv by design.
"""

from __future__ import annotations

import sys
import time

import numpy as np

PANE = 900


def probe_egl() -> bool:
    import moderngl

    ctx = moderngl.create_context(standalone=True, backend="egl")
    info = ctx.info
    print(f"[egl] GL_VENDOR   = {info['GL_VENDOR']}", flush=True)
    print(f"[egl] GL_RENDERER = {info['GL_RENDERER']}", flush=True)
    print(f"[egl] GL_VERSION  = {info['GL_VERSION']}", flush=True)
    hw = "NVIDIA" in info["GL_VENDOR"].upper()
    if not hw:
        print("[egl] FAIL: software renderer — the native SwiftShader trap",
              flush=True)
        return False

    fbo = ctx.simple_framebuffer((PANE, PANE), components=4)
    fbo.use()
    prog = ctx.program(
        vertex_shader="""#version 330
            in vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }""",
        fragment_shader="""#version 330
            out vec4 c; void main(){ c = vec4(0.2, 0.8, 0.3, 1.0); }""",
    )
    vbo = ctx.buffer(np.array([-1, -1, 1, -1, 0, 1], dtype="f4").tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, "p")
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    vao.render(mode=4)  # TRIANGLES
    px = np.frombuffer(fbo.read(components=4), dtype=np.uint8)
    print(f"[egl] triangle render: nonzero px = {(px > 0).sum()}", flush=True)

    n = 500
    t0 = time.monotonic()
    for _ in range(n):
        fbo.clear(0.1, 0.1, 0.1, 1.0)
        vao.render(mode=4)
        fbo.read(components=4)
    dt = time.monotonic() - t0
    print(f"[egl] {n} render+readback @ {PANE}x{PANE}: "
          f"{n / dt:.0f} fps ({1e3 * dt / n:.2f} ms/frame)", flush=True)
    return (px > 0).sum() > 0


def probe_em() -> bool:
    from cloudvolume import CloudVolume

    t0 = time.monotonic()
    vol = CloudVolume(
        "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14",
        mip=0, use_https=True, cache="/scratch/kp0374/native_spike/cv_cache",
        progress=False)
    cut = vol[143700:144212, 60820:61332, 192:193]  # 512^2 @ default position
    print(f"[em] cutout {cut.shape} dtype={cut.dtype} "
          f"mean={float(np.mean(cut)):.1f} in {time.monotonic() - t0:.1f}s",
          flush=True)
    return cut.size > 0 and float(np.std(cut)) > 1.0


def probe_mesh(root_ids) -> bool:
    from cloudvolume import CloudVolume

    vol = CloudVolume(
        "precomputed://gs://flywire_v141_m783", mip=0, use_https=True,
        cache="/scratch/kp0374/native_spike/cv_cache", progress=False)
    ok = 0
    for rid in root_ids:
        t0 = time.monotonic()
        try:
            m = vol.mesh.get(int(rid))
            mesh = m[int(rid)] if isinstance(m, dict) else m
            print(f"[mesh] {rid}: {len(mesh.vertices)} verts, "
                  f"{len(mesh.faces)} faces in {time.monotonic() - t0:.1f}s",
                  flush=True)
            ok += 1
        except Exception as e:
            print(f"[mesh] {rid}: FAIL {type(e).__name__}: {e}", flush=True)
    return ok == len(root_ids)


def main() -> int:
    results = {}
    for name, fn in [("egl", probe_egl), ("em", probe_em),
                     ("mesh", lambda: probe_mesh(
                         ["720575940623044103",   # default-URL segment
                          "720575940611949205",   # eval pool q1
                          "720575940632192676"])),  # eval pool q4 (3.6M nm)
                     ]:
        try:
            results[name] = fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = False
        print(f"[probe] {name}: {'PASS' if results[name] else 'FAIL'}",
              flush=True)
    print(f"[probe] SUMMARY: {results}", flush=True)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
