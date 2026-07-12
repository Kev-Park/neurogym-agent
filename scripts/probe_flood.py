"""Flood the screenshot-readback path to measure its raw ceiling directly.

Full-step throughput (~40 sps/GPU) bundles capture + DINO + state-gather +
decode. This strips all of that: build M browsers on ONE GPU with a real
Neuroglancer frame loaded, then each browser tight-loops raw CDP
`Page.captureScreenshot` (no decode, no action, no DINO) from its own thread.
Aggregate captures/sec at the plateau = the pure readback/encode/transport
ceiling.
  - lands ≈ full-step ~40  => capture IS the wall (no headroom w/o async).
  - lands ≫ 40             => full-step rate is bound by the OTHER per-step work
                              (DINO/decode/GIL) and there's headroom to reclaim.

Single-process/threaded: captureScreenshot blocks on CDP I/O (releasing the GIL),
so threads overlap on the readback; the base64 result is discarded (≈no GIL-held
decode), so this is close to the true readback ceiling as seen from one process.

    uv run --no-sync python scripts/probe_flood.py <M> <seconds>
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np

from ngllib_agent.env_build import load_config, make_env_creator


def main() -> int:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

    cfg = load_config("configs/ppo_zmax_navigate.yaml")
    cfg.setdefault("obs", {})["mode"] = "dino"
    venv = make_env_creator(cfg, vector_mode="threads")({"num_envs": M})

    def acts():
        return np.stack([venv.single_action_space.sample() for _ in range(M)])

    # reliable warmup past the cold-start herd (same pattern as probe_throughput)
    for attempt in range(5):
        try:
            venv.reset(seed=0)
            for _ in range(4):
                venv.step(acts())
            break
        except Exception as e:
            print(f"[flood] warmup {attempt} failed: {type(e).__name__}: "
                  f"{str(e)[:80]}; retry", flush=True)
            time.sleep(5)
    else:
        print(f"[flood] RESULT M={M} caps_s=FAILED (warmup)", flush=True)
        venv.close()
        return 1

    # one dedicated CDP session per browser page
    cdps = []
    for sub in venv.envs:
        pg = sub.unwrapped.page
        cdps.append(pg.context.new_cdp_session(pg))

    counts = [0] * M
    deadline = time.time() + T

    def flood(i, cdp):
        c = 0
        while time.time() < deadline:
            try:
                cdp.send("Page.captureScreenshot", {"format": "jpeg", "quality": 85})
                c += 1
            except Exception:
                break
        counts[i] = c

    threads = [threading.Thread(target=flood, args=(i, c)) for i, c in enumerate(cdps)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.time() - t0

    total = sum(counts)
    print(f"[flood] RESULT M={M} caps_s={total/dt:.1f} per_browser={total/dt/M:.2f} "
          f"(caps={total} in {dt:.0f}s)", flush=True)
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
