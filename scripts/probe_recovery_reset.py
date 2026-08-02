"""Reset-path glitch-recovery A/B probe (Q2 variant) — pure env, no Ray/PPO.

Companion to probe_recovery.py (which stresses the STEP path). This one hammers
the RESET/navigation path — repeatedly reset()-ing M browsers under contention,
which is exactly what `recovery_mode` changes: `_navigate_with_retry` uses a
cheap context recycle (in_place) vs a full browser relaunch (escalate) between
attempts. The training storms are reset storms, so this is the path where Q2's
benefit should actually appear.

Each cycle: reset all M envs (re-navigate to fresh neuron URLs via the provider)
+ a few steps to reach a realistic mid-episode state, then reset again. Reports
resets/sec + per-reset wall-time distribution + the reset-stall tax + failures.

    uv run --no-sync python scripts/probe_recovery_reset.py <M> <N_resets> <escalate|in_place>
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from ngllib_agent.env_build import load_config, make_env_creator


def main() -> int:
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    recovery = sys.argv[3] if len(sys.argv) > 3 else "escalate"
    tag = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

    cfg = load_config("configs/ppo_zmax_navigate.yaml")
    cfg.setdefault("obs", {})["mode"] = "dino"
    cfg.setdefault("env", {})["recovery_mode"] = recovery
    venv = make_env_creator(cfg, vector_mode="threads")({"num_envs": M})

    def acts():
        return np.stack([venv.single_action_space.sample() for _ in range(M)])

    # Warm up: one reset + a few steps so first-cycle cold start isn't counted.
    for attempt in range(5):
        try:
            venv.reset(seed=0)
            for _ in range(5):
                venv.step(acts())
            break
        except Exception as e:
            print(f"[reset] warmup attempt {attempt} failed: {type(e).__name__}: "
                  f"{str(e)[:80]}; retrying", flush=True)
            time.sleep(5)
    else:
        print(f"[reset] RESULT M={M} gpu={tag} mode={recovery} resets_per_s=FAILED "
              f"(could not warm up)", flush=True)
        venv.close()
        return 1
    print(f"[reset] M={M} gpu={tag} mode={recovery} warm; measuring {N} "
          f"reset cycles...", flush=True)

    reset_dt = np.empty(N, dtype=np.float64)
    n_fail = 0
    t0 = time.time()
    for i in range(N):
        s = time.time()
        try:
            venv.reset(seed=i + 1)
            # a few steps to land in a realistic mid-episode state before next reset
            for _ in range(3):
                venv.step(acts())
        except Exception as e:
            n_fail += 1
            print(f"[reset] cycle {i} FAILED: {type(e).__name__}: {str(e)[:80]}",
                  flush=True)
        reset_dt[i] = time.time() - s
    dt = time.time() - t0

    med = float(np.median(reset_dt))
    stall_thresh = 3.0 * med
    stall_mask = reset_dt > stall_thresh
    n_stall = int(stall_mask.sum())
    stall_time = float(reset_dt[stall_mask].sum())

    print(
        f"[reset] RESULT M={M} gpu={tag} mode={recovery} "
        f"resets_per_s={N/dt:.2f} cycle_med={med:.1f}s "
        f"cycle_p90={np.percentile(reset_dt,90):.1f}s cycle_p99={np.percentile(reset_dt,99):.1f}s "
        f"cycle_max={reset_dt.max():.1f}s "
        f"stalls={n_stall}/{N} stall_time={stall_time:.0f}s "
        f"stall_frac_wall={100*stall_time/dt:.0f}% fails={n_fail}",
        flush=True,
    )
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
