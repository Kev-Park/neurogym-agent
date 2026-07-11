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


def run_config(name, procs):
    """procs: list of (cuda_visible_devices, M). Launch all concurrently, sum sps."""
    children = []
    for cuda, M in procs:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cuda))
        p = subprocess.Popen(
            [sys.executable, THRU, str(M), str(NSTEPS)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        children.append((cuda, M, p))

    total, detail = 0.0, []
    for cuda, M, p in children:
        out, _ = p.communicate()
        sps = None
        for line in out.splitlines():
            if "RESULT" in line:
                m = _SPS.search(line)
                if m:
                    sps = float(m.group(1))
        if sps is None:
            detail.append(f"gpu{cuda}/M{M}=FAIL")
        else:
            total += sps
            detail.append(f"gpu{cuda}/M{M}={sps:.1f}")
    print(f"[procscale] {name}: AGGREGATE={total:.1f} sps  ({'  '.join(detail)})",
          flush=True)


def main() -> int:
    configs = [
        ("A1_1gpu_1x32", [(0, 32)]),
        ("A2_2gpu_2x32", [(0, 32), (1, 32)]),
        ("A4_4gpu_4x32", [(0, 32), (1, 32), (2, 32), (3, 32)]),
        ("B2_1gpu_2x16", [(0, 16), (0, 16)]),
        ("B4_1gpu_4x8",  [(0, 8), (0, 8), (0, 8), (0, 8)]),
    ]
    for name, procs in configs:
        run_config(name, procs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
