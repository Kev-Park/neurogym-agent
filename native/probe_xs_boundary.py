"""What does Neuroglancer do with a rejected 2D zoom (crossSectionScale <= 0)?

The 3D zoom has a measured rule -- a non-positive projectionScale leaves the
whole viewer state unreadable, which is why ngllib.state keeps the previous
value (probe_zoom_boundary, gate 4a). Exposing the 2D pane's zoom as an action
needs the same answer for `crossSectionScale`, and it cannot be read off the
source: `TrackableZoom.restoreState` runs verifyFinitePositiveFloat for both,
but whether NG drops the field, the edit, or the whole state is behaviour.

Also measures the upper end: an action space needs to know whether a very large
crossSectionScale is clamped, accepted, or rejected.

Drives Chrome directly (the environment never sends such a URL) and diffs the
readback, exactly like the 3D probe.

    uv run --no-sync python native/probe_xs_boundary.py
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

FIELDS = ("position", "crossSectionScale", "projectionScale", "projectionOrientation")


def main() -> int:
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
    chrome = env.unwrapped.renderer
    start = dict(info["json_state"])
    xs0 = float(start["crossSectionScale"])
    print(f"start: pos={start['position']} xs={xs0} ps={start['projectionScale']}", flush=True)

    verdicts = []
    for bad_xs in (0.0, -2.0, 1e6):
        env.reset(options={"state": start, "task_info": info["task_info"]})
        # Move position AND change the 3D zoom too, so the readback says whether
        # NG drops just the bad field or the entire edit.
        edited = dict(start)
        edited["position"] = [start["position"][0] + 100.0,
                              start["position"][1] + 50.0, start["position"][2]]
        edited["projectionScale"] = float(start["projectionScale"]) * 1.5
        edited["crossSectionScale"] = bad_xs
        chrome.page.goto(chrome._state_to_url(edited))

        after = None
        for t in range(int(args.settle_s * 5)):
            time.sleep(0.2)
            try:
                raw = chrome._get_json_state()
            except Exception as e:  # noqa: BLE001
                print(f"xs={bad_xs} t={0.2 * (t + 1):.1f}s: readback failed: {e}", flush=True)
                continue
            have = sorted(k for k in FIELDS if k in raw)
            if t % 10 == 9 or len(have) == 4:
                print(f"xs={bad_xs} t={0.2 * (t + 1):.1f}s: fields present {have}", flush=True)
            if len(have) == 4:
                after = raw
                break
        if after is None:
            print(f"xs={bad_xs}: state never regained all fields in {args.settle_s}s "
                  "-> UNREADABLE, same failure mode as a non-positive projectionScale",
                  flush=True)
            verdicts.append((bad_xs, "UNREADABLE"))
            continue
        dpos = [round(a - b, 3) for a, b in zip(after["position"], start["position"])]
        print(f"xs={bad_xs}: pos delta {dpos}  xs {xs0} -> {after['crossSectionScale']}  "
              f"ps {start['projectionScale']} -> {after['projectionScale']}", flush=True)
        if after["crossSectionScale"] == xs0 and any(abs(d) > 1e-6 for d in dpos):
            verdicts.append((bad_xs, "KEEPS-PREVIOUS-XS, rest of the edit applied"))
        elif after["crossSectionScale"] == bad_xs:
            verdicts.append((bad_xs, "ACCEPTED"))
        else:
            verdicts.append((bad_xs, f"OTHER -> {after['crossSectionScale']}"))
    env.close()
    print(f"XS-BOUNDARY {verdicts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
