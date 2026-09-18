"""Phase-2b de-risk probe: moderngl EGL texture -> CUDA (GL interop) -> torch
CUDA tensor -> CUDA IPC to a child process.

Validates, in isolated stages, the mechanism the fully-in-VRAM DINO server needs,
BEFORE integrating into ngllib/RLlib. Run on a GPU node:

    uv run --no-sync python scripts/glcuda_probe.py

Each stage prints PASS/FAIL so infeasibility is localized (esp. whether
cudaGraphicsGLRegisterImage works against moderngl's standalone EGL context, and
whether cuda-python's runtime ABI matches torch's bundled cu124 libcudart).
"""
from __future__ import annotations

import sys

import numpy as np

GL_TEXTURE_2D = 0x0DE1
W, H = 64, 48  # non-square to catch row/stride bugs


def _rt():
    from cuda.bindings import runtime as rt
    return rt


def _chk(ret, what):
    """cuda-python runtime calls return (err, *out); raise on nonzero err."""
    rt = _rt()
    if isinstance(ret, (tuple, list)):
        err, out = ret[0], tuple(ret[1:])
    else:
        err, out = ret, ()
    if int(err) != 0:
        _, name = rt.cudaGetErrorString(err)
        raise RuntimeError(f"{what}: cuda err {int(err)} {name}")
    return out


def _child(entry):
    """Runs in a spawned process: read the shared CUDA tensor, print checksum."""
    import torch  # noqa
    t = entry  # torch reconstructs the CUDA tensor via its IPC reduction
    torch.cuda.synchronize()
    s = int(t.sum().item())
    c00 = t[0, 0].tolist()
    print(f"CHILD: got tensor shape={tuple(t.shape)} sum={s} px00={c00}", flush=True)


def main():
    import torch
    rt = _rt()

    # ---- STAGE 1: moderngl EGL render + CPU readback (reference) ----
    import moderngl
    ctx = moderngl.create_context(standalone=True, backend="egl")
    tex = ctx.texture((W, H), 4)
    fbo = ctx.framebuffer(color_attachments=[tex])
    fbo.use()
    fbo.clear(0.2, 0.4, 0.8, 1.0)
    cpu = np.frombuffer(fbo.read(components=4), np.uint8).reshape(H, W, 4).copy()
    print(f"STAGE1 PASS moderngl render+readback: px00={cpu[0,0].tolist()} glo={tex.glo}",
          flush=True)

    # ---- STAGE 2: torch CUDA context + register GL texture ----
    torch.cuda.init()
    dev = torch.device("cuda:0")
    dst = torch.empty((H, W, 4), dtype=torch.uint8, device=dev)

    flags = rt.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly
    (resource,) = _chk(rt.cudaGraphicsGLRegisterImage(tex.glo, GL_TEXTURE_2D, flags),
                       "cudaGraphicsGLRegisterImage")
    print("STAGE2 PASS registered GL image with CUDA", flush=True)

    # ---- STAGE 3: map -> copy cudaArray to linear torch buffer -> compare ----
    # cuda-python takes the resource directly (not a list) when count=1.
    _chk(rt.cudaGraphicsMapResources(1, resource, 0), "MapResources")
    (arr,) = _chk(rt.cudaGraphicsSubResourceGetMappedArray(resource, 0, 0),
                  "GetMappedArray")
    _chk(rt.cudaMemcpy2DFromArray(
        dst.data_ptr(), W * 4, arr, 0, 0, W * 4, H,
        rt.cudaMemcpyKind.cudaMemcpyDeviceToDevice), "Memcpy2DFromArray")
    _chk(rt.cudaGraphicsUnmapResources(1, resource, 0), "Unmap")
    torch.cuda.synchronize()
    got = dst.cpu().numpy()
    match = bool(np.array_equal(got, cpu))
    print(f"STAGE3 {'PASS' if match else 'FAIL'} GL->CUDA->torch matches CPU readback "
          f"(match={match}, got px00={got[0,0].tolist()})", flush=True)
    if not match:
        print("STAGE3 FAIL: aborting IPC test", flush=True)
        return 1

    # ---- STAGE 4: CUDA IPC of the torch tensor to a spawned child ----
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    p = mp.Process(target=_child, args=(dst,))
    p.start()
    p.join(30)
    ok = (p.exitcode == 0)
    print(f"STAGE4 {'PASS' if ok else 'FAIL'} CUDA-IPC to child (exitcode={p.exitcode}); "
          f"parent sum={int(dst.sum().item())}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
