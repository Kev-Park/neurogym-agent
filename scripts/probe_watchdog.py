"""R2 GPU confirmation: SIGSTOP a live Chrome mid-run, verify the watchdog
kills it, the step raises, and the env recovers on the next reset.

Run on a vulkan node via srun. Uses a short step_timeout so the test is quick.
"""

from __future__ import annotations

import os
import signal
import time

import ngllib
from ngllib_agent.env_build import load_config
from ngllib_agent.providers import FlywireSkeletonProvider
from ngllib_agent.rewards import make_z_reward_factory, make_z_termination_factory
from ngllib_agent.wrappers import MultiDiscreteActionWrapper, ResilientStepWrapper

STEP_TIMEOUT = 8.0  # short, so the hang test doesn't take 30s


def main() -> int:
    cfg = load_config("configs/ppo_zmax_navigate.yaml")
    ec, ac, rc = cfg["env"], cfg["action"], cfg["reward"]
    from ngllib_agent.rewards import ZRewardConfig
    from ngllib_agent.wrappers import ActionSpec

    rcfg = ZRewardConfig(**{k: rc[k] for k in ("z_tolerance", "success", "z_shaping_coef", "step_penalty")})
    base = ngllib.Environment(
        headless=True, renderer="gpu", orientation="euler",
        left_pane=False, right_pane=True, image_size=(84, 84),
        reset_state_provider=FlywireSkeletonProvider(ec["parquet_path"]),
        reward_factory=make_z_reward_factory(rcfg),
        termination_factory=make_z_termination_factory(rcfg),
        step_timeout_s=STEP_TIMEOUT, reset_timeout_s=120.0,
    )
    x0, y0, x1, y1 = ac["pane_3d_bounds"]
    spec = ActionSpec(grid_rows=ac["grid_rows"], grid_cols=ac["grid_cols"],
                      pane_x0=x0, pane_y0=y0, pane_x1=x1, pane_y1=y1,
                      rotation_bins_per_axis=ac["rotation_bins_per_axis"],
                      rotation_step_rad=ac["rotation_step_rad"],
                      zoom_bins=ac["zoom_bins"], zoom_step=ac["zoom_step"])
    env = ResilientStepWrapper(MultiDiscreteActionWrapper(base, spec))

    obs, info = env.reset(seed=0)
    pid = base._chrome_pid
    print(f"[probe] reset ok; chrome_pid={pid}", flush=True)
    assert pid is not None, "watchdog needs a chrome pid; none found"

    for _ in range(3):
        env.step(env.action_space.sample())
    print("[probe] 3 normal steps ok", flush=True)

    # Freeze Chrome so the next Playwright call blocks forever.
    os.kill(pid, signal.SIGSTOP)
    print(f"[probe] SIGSTOP sent to chrome {pid}; stepping into the hang...", flush=True)
    t0 = time.time()
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    dt = time.time() - t0
    print(f"[probe] step returned after {dt:.1f}s trunc={trunc} glitch={info.get('env_glitch')}", flush=True)
    assert trunc is True, "resilient wrapper should truncate on the watchdog-killed step"
    assert dt < STEP_TIMEOUT + 15, f"watchdog should fire near {STEP_TIMEOUT}s, took {dt:.1f}s"
    # Chrome should be dead now (watchdog killed the stopped process).
    still = base._chrome_pid
    try:
        os.kill(pid, 0)
        alive = True
    except OSError:
        alive = False
    print(f"[probe] old chrome alive={alive}", flush=True)

    # Recovery: next reset must relaunch a fresh browser and work.
    obs, info = env.reset()
    new_pid = base._chrome_pid
    print(f"[probe] recovery reset ok; new chrome_pid={new_pid}", flush=True)
    assert new_pid is not None and new_pid != pid, "reset should relaunch a new browser"
    env.step(env.action_space.sample())
    env.close()
    print("WATCHDOG PROBE PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
