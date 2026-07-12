"""Phase-1 test of the optimized obs architecture (no browsers).

The ~40 wall is GIL-bound per-env orchestration, not any single stage. This
tests whether restructuring the obs path removes it, by feeding M identical
pre-captured JPEG frames through two pipelines and measuring obs/sec (= the
env-steps/sec the obs path allows):

  A (current): M THREADS, each does CPU PIL-decode -> split panes ->
     encoder.encode([left,right]) (batch 2, own CPU->GPU transfer). All the
     decode + numpy prep is GIL-held and contends across threads.
  B (optimized): ONE thread does batched GPU nvJPEG-decode of all M frames ->
     split on GPU -> ONE DINO forward over 2M panes -> single GPU->CPU transfer.
     Implements opt (1) GPU decode + (3) batched DINO; removes the per-env
     GIL-held work.

If B >> A the vector-level obs rewrite is worth building (Phase 2: real loop +
browsers + process count). If B ~= A, the hypothesis is wrong.

    uv run --no-sync python scripts/probe_obs_arch.py <M> <seconds>
"""

from __future__ import annotations

import base64
import io
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.io import decode_jpeg

import ngllib
from ngllib_agent.env_build import load_config
from ngllib_agent.obs.dino_encoder import get_dino_encoder
from ngllib_agent.providers import FlywireSkeletonProvider


def capture_frame(cfg):
    ec = cfg["env"]
    base = ngllib.Environment(
        headless=True, renderer="gpu", orientation="euler",
        left_pane=True, right_pane=True, window_size=(1800, 900),
        reset_state_provider=FlywireSkeletonProvider(ec["parquet_path"]),
    )
    base.reset(seed=0)
    cdp = base.page.context.new_cdp_session(base.page)
    res = cdp.send("Page.captureScreenshot", {"format": "jpeg", "quality": 85})
    jpeg = base64.b64decode(res["data"])
    base.close()
    return jpeg


def obs_A_once(enc, jpeg):
    img = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))  # (H, W, 3)
    mid = img.shape[1] // 2
    enc.encode([img[:, :mid], img[:, mid:]])  # per-env batch 2


def main() -> int:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    cfg = load_config("configs/ppo_zmax_navigate.yaml")
    jpeg = capture_frame(cfg)
    print(f"[obsarch] M={M} frame_bytes={len(jpeg)}", flush=True)

    enc = get_dino_encoder()
    obs_A_once(enc, jpeg)  # warm

    # ---- Mode A: M threads, per-env CPU decode + DINO(batch 2) ----
    counts = [0] * M
    deadline = time.time() + T

    def wa(i):
        c = 0
        while time.time() < deadline:
            obs_A_once(enc, jpeg)
            c += 1
        counts[i] = c

    ths = [threading.Thread(target=wa, args=(i,)) for i in range(M)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    a_rate = sum(counts) / (time.time() - t0)
    print(f"[obsarch] A per-env (CPU decode + DINO batch2, {M} threads): "
          f"{a_rate:.0f} obs/s", flush=True)

    # ---- Mode B: batched GPU decode + one DINO forward of 2M panes ----
    model, dev = enc.model, enc.device
    mean, std, insz = enc._mean, enc._std, enc.input_size
    jt = torch.from_numpy(np.frombuffer(jpeg, np.uint8).copy())

    def obs_B_batch():
        imgs = [decode_jpeg(jt, device="cuda") for _ in range(M)]  # (3,H,W) uint8 on GPU
        panes = []
        for im in imgs:
            w = im.shape[2] // 2
            panes.append(im[:, :, :w])
            panes.append(im[:, :, w:])
        batch = torch.stack([p.float() for p in panes])            # (2M,3,H,W/2)
        batch = F.interpolate(batch, size=(insz, insz), mode="bilinear",
                              align_corners=False).div_(255.0).sub_(mean).div_(std)
        with torch.no_grad():
            feats = model(batch)
        return feats.detach().cpu().numpy()                        # single transfer

    obs_B_batch()  # warm
    n = 0
    t0 = time.time()
    while time.time() - t0 < T:
        obs_B_batch()
        n += M
    b_rate = n / (time.time() - t0)
    print(f"[obsarch] B batched (GPU decode + DINO batch{2*M}, 1 thread): "
          f"{b_rate:.0f} obs/s", flush=True)
    print(f"[obsarch] RESULT M={M} speedup B/A = {b_rate/a_rate:.1f}x  "
          f"(A={a_rate:.0f}, B={b_rate:.0f} obs/s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
