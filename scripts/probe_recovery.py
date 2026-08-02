"""Glitch-recovery A/B probe (Q2) — pure env, no Ray/PPO.

Steps a ThreadedVectorEnv of M browsers under contention and measures how the
ngllib glitch-recovery strategy affects throughput. In a ThreadedVectorEnv every
vector-step joins all M sticky threads, so a glitchy sub-env stalls the whole
step — the within-runner analog of the cross-runner sample barrier. The A/B:

  recovery_mode=escalate  — current: full browser relaunch on repeated glitch
  recovery_mode=in_place  — legacy-style cheap context recycle at the source

Reports aggregate sps PLUS the per-step wall-time distribution and the "stall"
tax (steps far above median = a sub-env mid-recovery), which is exactly the
SPS-relevant quantity the two strategies move.

    uv run --no-sync python scripts/probe_recovery.py <M> <N_steps> <escalate|in_place>
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from ngllib_agent.env_build import load_config, make_env_creator


def main() -> int:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    recovery = sys.argv[3] if len(sys.argv) > 3 else "escalate"
    tag = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

    cfg = load_config("configs/ppo_zmax_navigate.yaml")
    cfg.setdefault("obs", {})["mode"] = "dino"
    cfg.setdefault("env", {})["recovery_mode"] = recovery
    venv = make_env_creator(cfg, vector_mode="threads")({"num_envs": M})

    rng = np.random.default_rng(0)

    def acts():
        return np.stack([venv.single_action_space.sample() for _ in range(M)])

    for attempt in range(5):
        try:
            venv.reset(seed=0)
            for _ in range(8):  # warm all browsers past cold start
                venv.step(acts())
            break
        except Exception as e:
            print(f"[recov] warmup attempt {attempt} failed: {type(e).__name__}: "
                  f"{str(e)[:80]}; retrying", flush=True)
            time.sleep(5)
    else:
        print(f"[recov] RESULT M={M} gpu={tag} mode={recovery} sps=FAILED "
              f"(could not warm up)", flush=True)
        venv.close()
        return 1
    print(f"[recov] M={M} gpu={tag} mode={recovery} warm; measuring {N} "
          f"vector-steps...", flush=True)

    step_dt = np.empty(N, dtype=np.float64)
    t0 = time.time()
    for i in range(N):
        s = time.time()
        venv.step(acts())
        step_dt[i] = time.time() - s
    dt = time.time() - t0
    steps = N * M

    med = float(np.median(step_dt))
    # A "stall" step = one sub-env is mid-recovery, dragging the joined vector
    # step far above the clean median. Threshold at 3x median (robust to the
    # clean-step jitter band).
    stall_thresh = 3.0 * med
    stall_mask = step_dt > stall_thresh
    n_stall = int(stall_mask.sum())
    stall_time = float(step_dt[stall_mask].sum())
    clean_sps = M / med if med > 0 else float("nan")

    print(
        f"[recov] RESULT M={M} gpu={tag} mode={recovery} "
        f"sps={steps/dt:.1f} per_env={steps/dt/M:.2f} "
        f"clean_sps={clean_sps:.1f} "
        f"step_med={med*1000:.0f}ms step_p90={np.percentile(step_dt,90)*1000:.0f}ms "
        f"step_p99={np.percentile(step_dt,99)*1000:.0f}ms step_max={step_dt.max()*1000:.0f}ms "
        f"stalls={n_stall}/{N} stall_time={stall_time:.0f}s "
        f"stall_frac_wall={100*stall_time/dt:.0f}%",
        flush=True,
    )
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
