"""GPU-time split probe: RENDER (2-pane + many-mesh raster) vs DINO (2x ViT).

The zmax-left co-pane workload (native_zmaxleft_v5.yaml, both panes + mesh
accumulation) runs GPU-COMPUTE-bound (100% util, VRAM ~98%) — the opposite of
the right-pane single-mesh regime, which is GIL/CPU-bound. This probe attributes
that GPU time to its two consumers so the sharpest lever is obvious:

  RENDER  = inner env.step()  — SimulatorRenderer GL raster of BOTH panes with
            the episode's accumulated meshes, plus the CPU read-back (glFinish).
  DINO    = DinoObservationWrapper.observation() — the frozen ViT-S/14 forward
            over the 2 panes (224x224 each).

Each phase is wall-timed with a torch.cuda.synchronize() barrier so the number
is that phase's own GPU-inclusive cost, measured on ONE env in ONE process (no
MPS contention) — the clean per-env FLOP split, which is what decides between
cutting render (LOD/mesh cap) and cutting DINO (fp16/smaller ViT/lower res).

Realistic mesh load: driven by the trained v5 policy (StatePklPolicy), stochastic
like an EnvRunner, warm-started so the accumulated-mesh state matches training.

Usage (GPU node, inside the venv):
  uv run --no-sync python native/probe_gpu_split.py \
      --config configs/native_zmaxleft_v5.yaml \
      --state-pkl /scratch/kp0374/checkpoints/zmaxleft-v5/ckpt_000490.pkl \
      --warmup 150 --measure 250
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import numpy as np

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))


def _find_dino_wrapper(env):
    """Walk the .env chain to the DinoObservationWrapper (has ._encoder)."""
    e = env
    while e is not None:
        if hasattr(e, "_encoder") and hasattr(e, "observation"):
            return e
        e = getattr(e, "env", None)
    raise RuntimeError("no DinoObservationWrapper (._encoder) in the env stack")


def _mesh_count(inner, info) -> int | None:
    """Best-effort count of loaded/visible segments for correlation (never fatal)."""
    for key in ("num_segments", "n_segments", "visible_segments", "loaded_meshes"):
        if isinstance(info, dict) and key in info:
            v = info[key]
            return int(v if np.isscalar(v) else len(v))
    base = getattr(inner, "unwrapped", inner)
    for attr in ("segments", "visible_segments", "loaded_segments"):
        v = getattr(base, attr, None)
        if v is not None:
            try:
                return len(v)
            except TypeError:
                pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/native_zmaxleft_v5.yaml")
    ap.add_argument("--state-pkl", required=True, help="v5 ckpt_*.pkl for the driver policy")
    ap.add_argument("--warmup", type=int, default=150, help="steps to accumulate meshes before measuring")
    ap.add_argument("--measure", type=int, default=250, help="steps in the timed window")
    ap.add_argument("--max-steps", type=int, default=500, help="per-episode cap before manual reset")
    ap.add_argument("--torch-seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.torch_seed)

    from eval_zmaxleft import StatePklPolicy  # trained-policy driver (rebuilds module)
    from ngllib_agent.env_build import build_env, load_config

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = "dino"
    cfg.setdefault("env", {})["max_episode_steps"] = args.max_steps

    full_env = build_env(cfg)
    dino_w = _find_dino_wrapper(full_env)
    inner = dino_w.env  # raw two-pane image obs (below the DINO encode)
    encoder = dino_w._encoder
    cuda = torch.cuda.is_available()

    def sync():
        if cuda:
            torch.cuda.synchronize()

    policy = StatePklPolicy(args.state_pkl, full_env, cfg.get("model", {}), stochastic=True)
    print(f"[gpu-split] config={args.config} pkl={os.path.basename(args.state_pkl)} "
          f"cuda={cuda} warmup={args.warmup} measure={args.measure}", flush=True)

    # --- rollout with phase timing --------------------------------------------
    raw_obs, _ = inner.reset()
    policy_obs = dino_w.observation(raw_obs)
    sync()

    t_render = t_dino = t_policy = t_reset = 0.0
    n_meas = 0
    ep_step = 0
    mesh_hist: Counter = Counter()
    last_report = time.time()
    wall0 = None

    total = args.warmup + args.measure
    i = 0
    while i < total:
        measuring = i >= args.warmup
        if measuring and wall0 is None:
            wall0 = time.time()  # wall start of the measured window (sps)

        tp = time.perf_counter()
        action = policy.act(policy_obs)
        sync()
        dt_policy = time.perf_counter() - tp

        tr = time.perf_counter()
        raw_obs, reward, terminated, truncated, info = inner.step(action)
        sync()
        dt_render = time.perf_counter() - tr

        td = time.perf_counter()
        policy_obs = dino_w.observation(raw_obs)
        sync()
        dt_dino = time.perf_counter() - td

        if measuring:
            t_render += dt_render
            t_dino += dt_dino
            t_policy += dt_policy
            n_meas += 1
            mc = _mesh_count(inner, info)
            if mc is not None:
                mesh_hist[mc] += 1

        ep_step += 1
        i += 1
        if terminated or truncated or ep_step >= args.max_steps:
            trs = time.perf_counter()
            raw_obs, _ = inner.reset()
            policy_obs = dino_w.observation(raw_obs)
            sync()
            if measuring:
                t_reset += time.perf_counter() - trs
            ep_step = 0

        if time.time() - last_report > 10:
            phase = "MEAS" if measuring else "warm"
            print(f"[gpu-split] {phase} step {i}/{total} "
                  f"render={dt_render*1e3:.0f}ms dino={dt_dino*1e3:.0f}ms", flush=True)
            last_report = time.time()

    wall = time.time() - wall0 if wall0 else 0.0
    step_ms = lambda tot: 1e3 * tot / n_meas if n_meas else 0.0
    r_ms, d_ms, p_ms = step_ms(t_render), step_ms(t_dino), step_ms(t_policy)
    gpu_ms = r_ms + d_ms  # the two GPU-bound phases
    render_pct = 100 * r_ms / gpu_ms if gpu_ms else 0.0
    dino_pct = 100 * d_ms / gpu_ms if gpu_ms else 0.0

    print("\n[gpu-split] ===== RESULT (single env, no MPS contention) =====", flush=True)
    print(f"[gpu-split] measured_steps={n_meas} sps_single_env={n_meas/wall:.2f} "
          f"(wall={wall:.0f}s)", flush=True)
    print(f"[gpu-split] per-step  RENDER={r_ms:.1f}ms  DINO={d_ms:.1f}ms  "
          f"policy={p_ms:.1f}ms  reset_amortized={1e3*t_reset/n_meas:.1f}ms", flush=True)
    print(f"[gpu-split] GPU SPLIT  render={render_pct:.1f}%  dino={dino_pct:.1f}%  "
          f"(of render+dino={gpu_ms:.1f}ms)", flush=True)
    print(f"[gpu-split] render:dino ratio = {r_ms/d_ms:.2f}" if d_ms else "", flush=True)
    if mesh_hist:
        keys = sorted(mesh_hist)
        wmean = sum(k * c for k, c in mesh_hist.items()) / sum(mesh_hist.values())
        print(f"[gpu-split] mesh/seg count in window: min={keys[0]} max={keys[-1]} "
              f"mean={wmean:.1f}", flush=True)
    full_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
