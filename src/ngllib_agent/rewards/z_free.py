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
    # v4 refinement terms (both 0 = v2/v3 behavior):
    # Symmetric plain-delta shaping alongside the best-so-far potential. It
    # telescopes to delta_coef*(z_final - z_start), so it is potential-based
    # (cannot change the optimal endpoint); under gamma<1 + budget termination
    # a dip pays now and recovers discounted-later, making descent costly in
    # proportion to depth WITHOUT the path bias of an asymmetric fine (the
    # lower stepping-stone route stays viable).
    delta_coef: float = 0.0
    # Charged once per DISTINCT new segment per episode (same dedup as the
    # novelty bonus): prices the real cost of a first mesh load. Reselect
    # after deselect reuses the cached mesh and is free. Sized well below a
    # good hop's value (~0.3-1.0) so ~40 shotgun selections cost ~0.1 while a
    # deliberate policy's 5-10 stay negligible.
    select_cost: float = 0.0
    # v5 exploration-conditioned load pricing: each new load additionally
    # costs explore_penalty_coef * max_u, where max_u = max over the episode's
    # SEEN segments (with landed meshes) of max(0, mesh_zmax - z_best) — the
    # best UNEXPLOITED candidate already in hand. Loading while holding an
    # unclimbed tall candidate is expensive; once everything in hand is
    # topped-out or below the episode best, only select_cost (the flat
    # externality floor that keeps dud-spam negative-EV) remains. Continuous,
    # threshold-free, derived from the task variable itself. Seen-set (not
    # visible-set) so deselecting an unexplored candidate cannot dodge it.
    explore_penalty_coef: float = 0.0


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
        state = {"z_best": None, "seen": None, "zmax": {}}

        def note_zmax(obs) -> None:
            for sid, zm in zip(obs.get("segments", ()),
                               obs.get("segment_zmax", ())):
                if float(zm) > 0.0:
                    state["zmax"][str(sid)] = float(zm)

        def max_unexplored() -> float:
            zb = state["z_best"]
            return max(0.0, max((state["zmax"][s] - zb for s in state["seen"]
                                 if s in state["zmax"]), default=0.0))

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
            # v4: symmetric plain-delta shaping (mild descent aversion)
            if cfg.delta_coef:
                r += cfg.delta_coef * (z - _z(prev_obs))
            # annealed selection-novelty scaffold (deduped for the episode,
            # payout capped at the first `select_novelty_cap` new segments)
            new = _visible(obs) - state["seen"]
            if new:
                # v5: price each new load against the best unexploited
                # candidate ALREADY in hand (before this step's additions;
                # their own meshes cannot have landed yet anyway).
                load_cost = cfg.select_cost
                if cfg.explore_penalty_coef:
                    load_cost += cfg.explore_penalty_coef * max_unexplored()
                n0 = len(state["seen"])
                state["seen"] |= new
                paid = len(new)
                if cfg.select_novelty_cap is not None:
                    already = max(0, n0 - state["n_init"])
                    paid = max(0, min(paid, cfg.select_novelty_cap - already))
                if paid:
                    r += cfg.select_novelty_bonus * novelty_scale() * paid
                r -= load_cost * len(new)
            note_zmax(obs)
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


def make_v3_termination_factory(
    budget_min: int = 100,
    budget_max: int = 500,
    stall_limit: int | None = 125,
    seed: int | None = None,
) -> "TerminationFactory":
    """v3 earliness pressure: TERMINATE (not truncate) each episode at a
    per-episode budget ~ U[budget_min, budget_max], or after `stall_limit`
    steps without a new episode z-best.

    Termination is load-bearing: RLlib bootstraps V(s) on truncation, which
    makes a random horizon pressure-free — only a true terminal (future value
    zeroed) prices dithering. The budget is unobserved by design (minimal
    bias): under an uncertain horizon the return-maximizing policy front-loads
    z gains. The uniform hazard is mildly non-Markovian (value depends on
    elapsed time the obs doesn't carry); PPO averages over it.

    EVAL PROTOCOL: evaluate with the v2 config (no v3 keys) so eval episodes
    stay TimeLimit-only and dz@budget remains paired-comparable across arms.
    """
    rng = np.random.default_rng(seed)

    def factory(task_info: dict[str, Any]) -> Callable[..., bool]:
        budget = int(rng.integers(budget_min, budget_max + 1))
        state = {"steps": 0, "z_best": None, "stall": 0}

        def terminated_fn(obs, action, prev_obs) -> bool:
            if state["z_best"] is None:
                state["z_best"] = _z(prev_obs)
            state["steps"] += 1
            z = _z(obs)
            if z > state["z_best"]:
                state["z_best"] = z
                state["stall"] = 0
            else:
                state["stall"] += 1
            if state["steps"] >= budget:
                return True
            return stall_limit is not None and state["stall"] >= stall_limit

        return terminated_fn

    return factory
