"""Is CPU JPEG-decode the ~40-sps host-side wall, and does GPU decode remove it?

Captures ONE real Neuroglancer JPEG frame (then closes the browser) and probes
the post-capture path with NO browser stepping in the loop:

  (1) single-worker stage timing — PIL decode, decode+resize(224) — absolute ms.
  (2) CPU-decode FLOOD (the memory-bandwidth confirmation, analog of the capture
      flood): P separate PROCESSES each tight-loop decode+resize. Aggregate
      decodes/s vs P=1,2,4,8. If it PLATEAUS (per-proc drops), the resource is
      shared (host memory bandwidth) — the smoking gun for a mem-bw wall. If it
      scales ~linearly with P, it's per-core compute, not memory. Compare the
      plateau to the ~40 sps full-step ceiling: if ≈40, decode IS the wall.
  (3) GPU decode (nvJPEG via torchvision.io.decode_jpeg on CUDA) + GPU resize —
      single-stream throughput. If ≫ the CPU plateau, moving decode to the GPU
      would lift the wall.

    uv run --no-sync python scripts/probe_decode.py
"""

from __future__ import annotations

import base64
import io
import multiprocessing as mp
import time

import numpy as np


def cpu_decode_once(jpeg, size=224):
    from PIL import Image
    im = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((size, size))
    return np.asarray(im)


def cpu_worker(jpeg, T, q):
    deadline = time.time() + T
    c = 0
    while time.time() < deadline:
        cpu_decode_once(jpeg)
        c += 1
    q.put(c)


def med(fn, n=50):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def main() -> int:
    import ngllib
    from ngllib_agent.env_build import load_config
    from ngllib_agent.providers import FlywireSkeletonProvider

    cfg = load_config("configs/ppo_zmax_navigate.yaml")
    ec = cfg["env"]
    base = ngllib.Environment(
        headless=True, renderer="gpu", orientation="euler",
        left_pane=True, right_pane=True, window_size=(1800, 900),
        reset_state_provider=FlywireSkeletonProvider(ec["parquet_path"]),
    )
    base.reset(seed=0)
    page = base.page
    cdp = page.context.new_cdp_session(page)
    res = cdp.send("Page.captureScreenshot", {"format": "jpeg", "quality": 85})
    jpeg = base64.b64decode(res["data"])
    base.close()
    print(f"[decode] frame jpeg_bytes={len(jpeg)}", flush=True)

    # (1) single-worker stage timing
    from PIL import Image
    m_dec = med(lambda: np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB")))
    m_full = med(lambda: cpu_decode_once(jpeg))
    print(f"[decode] CPU PIL decode: {m_dec:.1f}ms ({1000/m_dec:.0f}/s)  "
          f"decode+resize224: {m_full:.1f}ms ({1000/m_full:.0f}/s)", flush=True)

    # (2) CPU-decode flood across PROCESSES (memory-bandwidth confirmation)
    ctx = mp.get_context("spawn")
    T = 8.0
    for P in [1, 2, 4, 8]:
        q = ctx.Queue()
        procs = [ctx.Process(target=cpu_worker, args=(jpeg, T, q)) for _ in range(P)]
        for p in procs:
            p.start()
        counts = [q.get() for _ in range(P)]
        for p in procs:
            p.join()
        tot = sum(counts)
        print(f"[decode] CPU-flood P={P}: aggregate={tot/T:.0f} dec/s  "
              f"per_proc={tot/T/P:.0f}", flush=True)

    # (3) GPU decode (nvJPEG)
    try:
        import torch
        from torchvision.io import decode_jpeg
        jt = torch.from_numpy(np.frombuffer(jpeg, dtype=np.uint8).copy())
        for _ in range(5):  # warm nvJPEG + CUDA
            decode_jpeg(jt, device="cuda")
        torch.cuda.synchronize()

        def gpu_dec():
            img = decode_jpeg(jt, device="cuda").float()
            torch.nn.functional.interpolate(img.unsqueeze(0), size=(224, 224),
                                             mode="bilinear", align_corners=False)
            torch.cuda.synchronize()

        m_gpu = med(gpu_dec, n=50)
        print(f"[decode] GPU nvJPEG decode+resize224: {m_gpu:.1f}ms "
              f"({1000/m_gpu:.0f}/s)", flush=True)
    except Exception as e:
        print(f"[decode] GPU decode FAILED: {type(e).__name__}: {str(e)[:140]}",
              flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
