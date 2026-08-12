"""M5 functional smoke — warm-context reset-ahead on a real browser (1 GPU).

Runs one env with a short episode cadence (reset every 40 steps, prep from step
20) so several boundaries are crossed quickly, then reads back the event log:
  - were warm contexts prepared and READY before the boundary (warm_ready)?
  - did resets adopt them (reset event warm=true) with navigate_ms ~0?
  - inline (first/fallback) resets for contrast.

    uv run --no-sync python scripts/probe_reset_ahead.py
"""

from __future__ import annotations

import glob
import json
import os

EVDIR = "/tmp/m5_smoke"
os.makedirs(EVDIR, exist_ok=True)
for f in glob.glob(EVDIR + "/ev-*.jsonl"):
    os.remove(f)
os.environ["NGLLIB_EVENT_LOG"] = EVDIR + "/ev-{host}-{pid}.jsonl"

import ngllib  # noqa: E402  (env var must be set before construction)
from ngllib_agent.env_build import load_config  # noqa: E402
from ngllib_agent.providers import FlywireSkeletonProvider  # noqa: E402


def main() -> int:
    cfg = load_config("configs/ppo_zmax_navigate.yaml")
    ec = cfg["env"]
    env = ngllib.Environment(
        headless=True, renderer="gpu", orientation="euler",
        left_pane=True, right_pane=True, window_size=(1800, 900),
        reset_state_provider=FlywireSkeletonProvider(ec["parquet_path"]),
        reset_ahead=True, reset_ahead_after_steps=20,
    )
    env.reset(seed=0)
    for ep in range(4):
        steps = 0
        while steps < 40:
            try:
                env.step(env.action_space.sample())
            except Exception as e:  # transient glitch: keep the smoke going
                print(f"[m5] step glitch ep{ep} step{steps}: {type(e).__name__}", flush=True)
            steps += 1
        env.reset()
        print(f"[m5] boundary {ep + 1} crossed", flush=True)
    env.close()

    resets, ready, prep_failed = [], 0, 0
    for f in glob.glob(EVDIR + "/ev-*.jsonl"):
        for line in open(f, errors="ignore"):
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("evt") == "reset":
                resets.append(e)
            elif e.get("evt") == "warm_ready":
                ready += 1
            elif e.get("evt") == "warm_prep_failed":
                prep_failed += 1

    warm = [r for r in resets if r.get("warm")]
    inline = [r for r in resets if not r.get("warm")]
    wnav = [r.get("navigate_ms") or 0 for r in warm]
    inav = [r.get("navigate_ms") or 0 for r in inline]
    print(f"[m5] RESULT resets={len(resets)} warm_adopted={len(warm)} "
          f"warm_ready_events={ready} prep_failed={prep_failed}", flush=True)
    if warm:
        print(f"[m5]   warm  resets: total_ms med={sorted(r['total_ms'] for r in warm)[len(warm)//2]:.0f} "
              f"navigate_ms max={max(wnav):.0f}", flush=True)
    if inline:
        print(f"[m5]   inline resets: total_ms med={sorted(r['total_ms'] for r in inline)[len(inline)//2]:.0f} "
              f"navigate_ms max={max(inav):.0f}", flush=True)
    # Pass = at least 2 of the 4 boundary resets adopted a warm context and
    # skipped navigation entirely.
    ok = len(warm) >= 2 and (not wnav or max(wnav) < 100.0)
    print(f"[m5] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
