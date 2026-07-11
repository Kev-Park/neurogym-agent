"""Per-node process-scaling + GPU-scaling probe for the readback ceiling.

Launches N concurrent probe_throughput.py subprocesses (each a ThreadedVectorEnv
of M browsers) with per-process CUDA_VISIBLE_DEVICES pinning, and reports the
AGGREGATE sps across the node. Separates the two levers cleanly:

  (A) GPU-scaling — 1 process/GPU x M=32 over {1,2,4} GPUs. If aggregate scales
      ~linearly, the per-node ceiling was RLlib coordination, not the readback
      path, and adding GPUs (each hosting its own browsers) genuinely helps.
  (B) Process-packing on ONE GPU at a FIXED 32 browsers — 1xM32 vs 2xM16 vs
      4xM8. Same GPU, same browser count, more OS processes. If multi-process
      beats single-process threading, the single Python process / GIL was a
      limit and escaping it (multiprocess EnvRunners) helps even without more
      GPUs. If flat, threading already saturates what one GPU's readback allows.

VRAM note: ~0.6GB/Chrome caps a 24GB 3090 at ~32 browsers, so config A puts 32
browsers on each GPU (fits) and config B keeps the one-GPU total at 32.

    uv run --no-sync python scripts/probe_procscale.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
THRU = os.path.join(HERE, "probe_throughput.py")
NSTEPS = 100
_SPS = re.compile(r"sps=([0-9.]+)")


# Slurm hands us the allocated GPUs via CUDA_VISIBLE_DEVICES (0-based within the
# allocation under ConstrainDevices, or physical ids otherwise). Configs below
# use LOGICAL gpu indices 0..3; translate each onto the real allocated device so
# pinning is correct regardless of which physical GPUs Slurm gave us.
_ALLOC = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0,1,2,3").split(",")


def run_config(name, procs):
    """procs: list of (logical_gpu, M). Launch all concurrently, sum sps."""
    children = []
    for gpu, M in procs:
        dev = _ALLOC[gpu % len(_ALLOC)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(dev))
        p = subprocess.Popen(
            [sys.executable, THRU, str(M), str(NSTEPS)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        children.append((dev, M, p))

    total, detail = 0.0, []
    for dev, M, p in children:
        out, _ = p.communicate()
        sps = None
        for line in out.splitlines():
            if "RESULT" in line:
                m = _SPS.search(line)
                if m:
                    sps = float(m.group(1))
        if sps is None:
            detail.append(f"dev{dev}/M{M}=FAIL")
        else:
            total += sps
            detail.append(f"dev{dev}/M{M}={sps:.1f}")
    print(f"[procscale] {name}: AGGREGATE={total:.1f} sps  ({'  '.join(detail)})",
          flush=True)


def main() -> int:
    ngpu = len(_ALLOC)
    print(f"[procscale] allocated GPUs: {_ALLOC} (n={ngpu})", flush=True)
    configs = []
    # (A) GPU-scaling: k processes, one M=32 per GPU, for k = 1..ngpu.
    for k in range(1, ngpu + 1):
        configs.append((f"A{k}_{k}gpu_{k}x32", [(g, 32) for g in range(k)]))
    # (B) Process-packing on ONE GPU at a fixed 32 browsers.
    configs.append(("B2_1gpu_2x16", [(0, 16), (0, 16)]))
    configs.append(("B4_1gpu_4x8", [(0, 8)] * 4))
    for name, procs in configs:
        run_config(name, procs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
