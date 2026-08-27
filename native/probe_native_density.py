"""Single-node density probe: how many native envs per EnvRunner process?

Mirrors the production runner topology (one process, M sticky-thread envs via
ThreadedVectorEnv, one shared DINO) and measures aggregate steps/sec as M
scales. A node hosts 2 runner processes in production, so node throughput
~= 2x the best single-process number (CPU permitting).

Actions are uniform action_space samples (1/3 clicks — the expensive verb:
each click moves the position and forces EM/label tile refetches).

    uv run --no-sync python native/probe_native_density.py \
        --config configs/native.yaml --threads-list 4,8,16,32 --obs dino
"""

from __future__ import annotations

import argparse
import subprocess
import time

import numpy as np


def gpu_mem() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        return out.replace("\n", " | ")
    except Exception as e:
        return f"nvidia-smi fail: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/native.yaml")
    ap.add_argument("--threads-list", default="4,8,16,32")
    ap.add_argument("--secs", type=float, default=120.0)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--obs", choices=["pos", "dino"], default="dino")
    args = ap.parse_args()

    from ngllib_agent.env_build import load_config, make_env_creator

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = args.obs

    for m in [int(x) for x in args.threads_list.split(",")]:
        creator = make_env_creator(cfg, vector_mode="threads")
        t0 = time.monotonic()
        venv = creator({"num_envs": m})
        build_s = time.monotonic() - t0
        t0 = time.monotonic()
        venv.reset(seed=1)
        reset_s = time.monotonic() - t0

        def sample():
            return np.stack([venv.single_action_space.sample()
                             for _ in range(m)])

        for _ in range(args.warmup_steps):
            venv.step(sample())
        n = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.secs:
            venv.step(sample())
            n += 1
        el = time.monotonic() - t0
        sps = m * n / el
        print(f"DENSITY M={m} obs={args.obs} sps={sps:.1f} "
              f"per_env={sps / m:.2f} vec_steps={n} secs={el:.0f} "
              f"build_s={build_s:.0f} reset_s={reset_s:.0f} "
              f"vram: {gpu_mem()}", flush=True)
        venv.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
