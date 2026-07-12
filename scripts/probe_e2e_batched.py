"""Phase-2 end-to-end: does batched vector-level DINO beat per-env DINO in the
REAL loop (browsers + action + capture + decode)?

  mode=baseline : obs.mode=dino -> per-env DINO inside each sub-env step (current
                  production arch, ~40 sps/GPU).
  mode=proto    : obs.mode=raw  -> sub-envs return raw native two-pane image +
                  state; after venv.step, ONE batched DINO over all M frames
                  (2M panes). Tests opt (3) batched DINO. (opt (1) GPU-decode
                  deferred — needs an ngllib raw-JPEG mode; decode stays per-env
                  CPU here.)

    uv run --no-sync python scripts/probe_e2e_batched.py <M> <baseline|proto>
"""

from __future__ import annotations

import sys
import time

import numpy as np

from ngllib_agent.env_build import load_config, make_env_creator


def main() -> int:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    mode = sys.argv[2] if len(sys.argv) > 2 else "proto"
    N = 100

    cfg = load_config("configs/ppo_zmax_navigate.yaml")
    if mode == "baseline":
        cfg.setdefault("obs", {})["mode"] = "dino"  # env_build forces panes/native res
    elif mode == "proto":
        cfg.setdefault("obs", {})["mode"] = "raw"
        # raw mode passes env cfg through -> force the DINO input shape ourselves
        cfg["env"]["left_pane"] = True
        cfg["env"]["right_pane"] = True
        cfg["env"]["image_size"] = None
    else:
        print(f"[e2e] bad mode {mode!r}", flush=True)
        return 2

    venv = make_env_creator(cfg, vector_mode="threads")({"num_envs": M})

    enc = None
    if mode == "proto":
        from ngllib_agent.obs.dino_encoder import get_dino_encoder
        enc = get_dino_encoder()

    def acts():
        return np.stack([venv.single_action_space.sample() for _ in range(M)])

    def process(obs):
        # proto: one batched DINO forward over all M frames' 2 panes.
        if mode != "proto":
            return
        panes = []
        for im in obs["image"]:            # (M, H, W, 3)
            mid = im.shape[1] // 2
            panes.append(im[:, :mid])
            panes.append(im[:, mid:])
        enc.encode(panes)                  # (2M, D) — single batched forward

    for attempt in range(5):
        try:
            obs, _ = venv.reset(seed=0)
            for _ in range(6):
                obs = venv.step(acts())[0]
                process(obs)
            break
        except Exception as e:
            print(f"[e2e] warmup {attempt} failed: {type(e).__name__}: "
                  f"{str(e)[:80]}; retry", flush=True)
            time.sleep(5)
    else:
        print(f"[e2e] RESULT M={M} mode={mode} sps=FAILED (warmup)", flush=True)
        venv.close()
        return 1

    steps = 0
    t0 = time.time()
    for _ in range(N):
        obs = venv.step(acts())[0]
        process(obs)
        steps += M
    dt = time.time() - t0
    print(f"[e2e] RESULT M={M} mode={mode} sps={steps/dt:.1f} "
          f"(per_env={steps/dt/M:.2f})", flush=True)
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
