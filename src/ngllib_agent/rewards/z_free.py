"""Free-climb reward for zmax-left: maximize z across neurons, no terminal.

Task (zmax-left, 2026-09-24): from any start state, get as high as possible
within the TimeLimit, hopping neurons via left-pane selection when the current
one tops out. There is no computable per-start optimum, so there is NO success
terminal — episodes are TimeLimit-only and the return is "how high you got":

  reward_t = shaping_coef * max(0, z_t - z_best)          # best-so-far potential
           + novelty_bonus(t) * |newly selected segments|  # annealed scaffold
           + step_penalty                                   # no-op at fixed length

Best-so-far (not plain delta): descents and lateral setup moves cost nothing,
any new high pays once; the return telescopes to shaping_coef * (z_best_T - z_0).
z_best is per-EPISODE (a global/historical max would starve the signal and break
stationarity).

Selection novelty (approved 2026-09-24): a small one-time bonus for each NEW
distinct segment made visible this episode — direction-free (no ceiling gate:
the optimal route may pass through a LOWER-ceiling stepping-stone neuron), and
deduped for the episode's lifetime (re-select after deselect pays nothing).
It is exploration scaffolding, not objective: it anneals linearly to zero by
`novelty_end_steps` global env steps (same progress-file mechanism as the spawn
curriculum), after which the trained policy optimizes pure best-so-far z.

Needs `obs["segments"]` (the visible-set tuple ngllib adds on this branch).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:  # avoid a runtime ngllib import for pure-logic tests
    from ngllib import RewardFactory, TerminationFactory


@dataclass(frozen=True)
class ZFreeRewardConfig:
    shaping_coef: float = 0.001
    select_novelty_bonus: float = 0.1
    # Global env steps by which the novelty bonus reaches zero. None = never
    # anneal (ablation arms only — the approved production setting anneals).
    select_novelty_end_steps: float | None = 1_500_000
    # Only the FIRST cap new segments per episode pay (v1b lesson, job 982378:
    # uncapped 0.1/segment made select-count the objective — 175-200 new
    # segments/episode, whose mesh volume OOM-killed runners. The cap bounds
    # the scaffold's mass below the climbing signal without constraining HOW
    # or WHERE hops happen). None = uncapped (ablation only).
    select_novelty_cap: int | None = 10
    step_penalty: float = 0.0


def _z(obs: dict[str, Any]) -> float:
    return float(np.asarray(obs["position"])[2])


def _visible(obs: dict[str, Any]) -> frozenset[str]:
    return frozenset(str(s) for s in obs.get("segments", ()))


class _GlobalSteps:
    """total env steps from $CURRICULUM_PROGRESS_FILE (20 s TTL), else None —
    the same driver-published progress file the spawn curriculum reads."""

    def __init__(self) -> None:
        self._path = os.environ.get("CURRICULUM_PROGRESS_FILE")
        self._cache: tuple[float, float | None] = (0.0, None)

    def __call__(self) -> float | None:
        if not self._path:
            return None
        now = time.monotonic()
        ts, cached = self._cache
        if now - ts < 20.0:
            return cached
        steps = None
        try:
            with open(self._path) as f:
                steps = json.load(f).get("total_steps")
        except Exception:
            steps = None
        self._cache = (now, steps)
        return steps


def make_zfree_reward_factory(
    cfg: ZFreeRewardConfig = ZFreeRewardConfig(),
) -> "RewardFactory":
    """Factory: `task_info -> (obs, action, prev_obs, terminated) -> float`."""
    global_steps = _GlobalSteps()

    def factory(task_info: dict[str, Any]) -> Callable[..., float]:
        # Per-episode closure state, initialized lazily from the FIRST call's
        # prev_obs (= the reset observation).
        state = {"z_best": None, "seen": None}

        def novelty_scale() -> float:
            end = cfg.select_novelty_end_steps
            if end is None:
                return 1.0
            steps = global_steps()
            if steps is None:
                # no progress file (unit tests, ad-hoc envs): full bonus
                return 1.0
            return max(0.0, 1.0 - float(steps) / float(end))

        def reward_fn(obs, action, prev_obs, terminated) -> float:
            if state["z_best"] is None:
                state["z_best"] = _z(prev_obs)
                state["seen"] = set(_visible(prev_obs))
                state["n_init"] = len(state["seen"])
            r = cfg.step_penalty
            # best-so-far potential
            z = _z(obs)
            if z > state["z_best"]:
                r += cfg.shaping_coef * (z - state["z_best"])
                state["z_best"] = z
            # annealed selection-novelty scaffold (deduped for the episode,
            # payout capped at the first `select_novelty_cap` new segments)
            new = _visible(obs) - state["seen"]
            if new:
                n0 = len(state["seen"])
                state["seen"] |= new
                paid = len(new)
                if cfg.select_novelty_cap is not None:
                    already = max(0, n0 - state["n_init"])
                    paid = max(0, min(paid, cfg.select_novelty_cap - already))
                if paid:
                    r += cfg.select_novelty_bonus * novelty_scale() * paid
            return float(r)

        return reward_fn

    return factory


def make_no_termination_factory() -> "TerminationFactory":
    """TimeLimit-only episodes: the task terminal never fires."""

    def factory(task_info: dict[str, Any]) -> Callable[..., bool]:
        def terminated_fn(obs, action, prev_obs) -> bool:
            return False

        return terminated_fn

    return factory
