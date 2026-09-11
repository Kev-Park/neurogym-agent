"""Gate 4's open measurement: what does Neuroglancer do with a rejected zoom?

`TrackableZoom.restoreState` runs `verifyFinitePositiveFloat`, so a URL whose
projectionScale is 0 or negative fails to parse (renderer_seam_plan.md 7.1).
The shared state math (ngllib.state.next_projection_scale) keeps the previous
zoom and applies the OTHER components of the edit. Whether Neuroglancer itself
discards only the zoom, or the whole state edit, decides if that rule is
right -- and it cannot be read off the source, so drive Chrome to the boundary
and diff the readback.

Bypasses the environment on purpose (the environment never sends such a URL
any more): takes the ChromeRenderer, navigates to a URL that changes the
position AND zeroes the zoom, and reads the viewer state back.

    uv run --no-sync python native/probe_zoom_boundary.py
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/native_select.yaml")
    ap.add_argument("--settle-s", type=float, default=10.0)
    args = ap.parse_args()

    from ngllib_agent.env_build import build_env, load_config

    cfg = load_config(args.config)
    cfg.setdefault("obs", {})["mode"] = "raw"
    cfg["env"].update({"backend": "chrome", "left_pane": True, "right_pane": True,
                       "image_size": None, "capture_scale": 0.5})
    env = build_env(cfg)
    obs, info = env.reset(seed=3)
    base = env.unwrapped
    chrome = base.renderer
    start = dict(info["json_state"])
    print(f"start: pos={start['position']} ps={start['projectionScale']}", flush=True)

    verdicts = []
    for bad_ps in (0.0, -500.0):
        # Re-seat the known-good state first so each trial starts identically.
        env.reset(options={"state": start, "task_info": info["task_info"]})
        moved = dict(start)
        moved["position"] = [start["position"][0] + 100.0, start["position"][1] + 50.0,
                             start["position"][2]]
        moved["crossSectionScale"] = float(start["crossSectionScale"]) * 1.5
        moved["projectionScale"] = bad_ps
        url = chrome._state_to_url(moved)
        chrome.page.goto(url)
        # First run (job 883827): 2 s after the bad navigation the readback had
        # NO `position` key at all -- the very KeyError('position') glitch the
        # Chrome env used to truncate episodes on. So poll, and print exactly
        # which fields NG exposes as time passes, before judging.
        after = None
        for t in range(int(args.settle_s * 5)):
            time.sleep(0.2)
            try:
                raw = chrome._get_json_state()
            except Exception as e:  # noqa: BLE001
                print(f"ps={bad_ps} t={0.2 * (t + 1):.1f}s: readback failed: {e}", flush=True)
                continue
            have = sorted(k for k in ("position", "crossSectionScale", "projectionScale",
                                      "projectionOrientation") if k in raw)
            if t % 5 == 4 or len(have) == 4:
                print(f"ps={bad_ps} t={0.2 * (t + 1):.1f}s: fields present {have}", flush=True)
            if len(have) == 4:
                after = raw
                break
        if after is None:
            print(f"ps={bad_ps}: viewer state never regained all fields within "
                  f"{args.settle_s}s -- NG's parse failure leaves the state UNREADABLE, "
                  "which the old Chrome env surfaced as a missing-fields glitch and an "
                  "episode truncation, not as any state transition.", flush=True)
            verdicts.append((bad_ps, "UNREADABLE: whole state unavailable after a rejected zoom"))
            continue
        pos_moved = [round(a - b, 3) for a, b in
                     zip(after["position"], start["position"])]
        xs_after = after.get("crossSectionScale")
        ps_after = after.get("projectionScale")
        print(f"ps={bad_ps}: position delta {pos_moved}  crossSectionScale "
              f"{start['crossSectionScale']} -> {xs_after}  projectionScale "
              f"{start['projectionScale']} -> {ps_after}", flush=True)
        if ps_after == start["projectionScale"] and any(abs(d) > 1e-6 for d in pos_moved):
            v = "PARTIAL: zoom rejected, other components applied (matches next_projection_scale)"
        elif ps_after == start["projectionScale"] and not any(abs(d) > 1e-6 for d in pos_moved):
            v = "WHOLE-EDIT rejected: nothing changed (rule should move to apply_state_edit)"
        elif ps_after == bad_ps:
            v = "ACCEPTED non-positive zoom (contradicts the source read; re-examine 7.1)"
        else:
            v = f"OTHER: ps_after={ps_after}"
        print(f"  -> {v}", flush=True)
        verdicts.append((bad_ps, v))
    env.close()
    print("\nZOOM-BOUNDARY", verdicts, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
