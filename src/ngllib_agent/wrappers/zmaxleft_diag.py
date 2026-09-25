"""Per-episode diagnostics for the zmax-left free-climb task.

Sits between MultiDiscreteActionWrapper and the obs-mode wrappers, so it sees
the verb-indexed action vector and the RAW ngllib obs (position + the visible
`segments` tuple this branch adds). Emits one compact line per finished episode
(greppable as `[zmaxleft-ep]` in the slurm .out — the failure-diagnosis data
approved 2026-09-24) and mirrors the dict into `info["zmaxleft"]`.

Tracked: dz (best-so-far gain), new distinct segments selected, verb usage,
SHOW_ALL steps (empty visible set), step index of the last new high, and the
distinct-segment count at that moment (were hops involved in the best climb?).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ZmaxLeftDiagWrapper:
    """gymnasium Wrapper (lazy class def keeps gymnasium import off pure tests)."""

    def __new__(cls, env):
        import gymnasium as gym

        class _Impl(gym.Wrapper):
            _ep_seq = 0

            def __init__(self, env):
                super().__init__(env)
                self._reset_track({})

            def _reset_track(self, obs: dict[str, Any]) -> None:
                pos = obs.get("position")
                z = float(np.asarray(pos)[2]) if pos is not None else 0.0
                segs = set(obs.get("segments", ()))
                self._t = {
                    "z0": z, "z_best": z, "steps": 0, "steps_at_best": 0,
                    "seen": segs, "n0": len(segs), "new_segs": 0,
                    "segs_at_best": len(segs), "showall_steps": 0,
                    "verbs": np.zeros(8, dtype=int),
                }

            def _summary(self) -> dict[str, Any]:
                t = self._t
                return {
                    "dz": t["z_best"] - t["z0"],
                    "new_segs": t["new_segs"],
                    "hop_climb": int(t["segs_at_best"] > t["n0"]),
                    "steps_at_best": t["steps_at_best"],
                    "showall_steps": t["showall_steps"],
                    "len": t["steps"],
                    "verbs": t["verbs"][:6].tolist(),
                }

            def _emit(self) -> dict[str, Any]:
                _Impl._ep_seq += 1
                d = self._summary()
                print(f"[zmaxleft-ep] n={_Impl._ep_seq} dz={d['dz']:.1f} "
                      f"new_segs={d['new_segs']} hop_climb={d['hop_climb']} "
                      f"best@{d['steps_at_best']} showall={d['showall_steps']} "
                      f"len={d['len']} verbs={d['verbs']}", flush=True)
                return d

            def reset(self, **kwargs):
                # TimeLimit truncates ABOVE this wrapper, so the episode-end
                # flag never reaches step() here in TimeLimit-only tasks; a
                # reset always follows an episode end at every level, so the
                # finished episode is emitted HERE (and mirrored into the reset
                # info as zmaxleft_prev for eval harnesses).
                prev = self._emit() if self._t.get("steps", 0) > 0 else None
                obs, info = self.env.reset(**kwargs)
                self._reset_track(obs)
                if prev is not None:
                    info = dict(info)
                    info["zmaxleft_prev"] = prev
                return obs, info

            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                t = self._t
                t["steps"] += 1
                verb = int(np.asarray(action).ravel()[0])
                if 0 <= verb < len(t["verbs"]):
                    t["verbs"][verb] += 1
                segs = set(obs.get("segments", ()))
                new = segs - t["seen"]
                if new:
                    t["seen"] |= new
                    t["new_segs"] += len(new)
                if not segs:
                    t["showall_steps"] += 1
                z = float(np.asarray(obs["position"])[2])
                if z > t["z_best"]:
                    t["z_best"] = z
                    t["steps_at_best"] = t["steps"]
                    t["segs_at_best"] = len(t["seen"])
                if terminated or truncated:
                    # only fires when an episode end reaches THIS level (task
                    # terminals / resilient truncations below TimeLimit)
                    info = dict(info)
                    info["zmaxleft"] = self._emit()
                    self._reset_track({})  # guard against double-emit on reset
                return obs, reward, terminated, truncated, info

        return _Impl(env)
