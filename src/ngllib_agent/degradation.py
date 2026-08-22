"""Degraded-throughput detector (REFINEMENT R10).

coord-test-v7's one blemish: an EnvRunner restart churned for ~100 min
(iters 257-273 at ~360s vs the ~40s cruise) and nothing noticed — RLlib only
distinguishes dead from alive, not "alive at 15% throughput". The proven cure
is the blunt one: exit nonzero so the coordinator respawns the workload from
checkpoint (~5 min, validated many times in v1-v7). This class decides WHEN.

Trip rule: median of the last `window` iteration times exceeds
max(factor * baseline, floor_s), baseline = median of all observations so far.
Median-of-window makes single hang-recovery iterations invisible (~400s
surrounded by ~40s — 27 of them in v7, all benign) while sustained degradation
trips after ceil(window/2) degraded iterations. A uniformly slow run never
trips (baseline tracks it): that's a config problem, not a runtime pathology.
"""

from __future__ import annotations

from collections import deque
from statistics import median


class DegradationDetector:
    def __init__(
        self,
        window: int = 5,
        factor: float = 3.0,
        floor_s: float = 180.0,
        warmup: int = 10,
    ):
        self.window = int(window)
        self.factor = float(factor)
        self.floor_s = float(floor_s)
        self.warmup = max(int(warmup), int(window))
        self._times: list[float] = []
        self._recent: deque[float] = deque(maxlen=self.window)

    @property
    def baseline_s(self) -> float | None:
        return median(self._times) if self._times else None

    def observe(self, iter_time_s: float) -> bool:
        """Record one iteration time; True = sustained degradation, act now."""
        self._times.append(float(iter_time_s))
        self._recent.append(float(iter_time_s))
        if len(self._times) < self.warmup:
            return False
        threshold = max(self.factor * median(self._times), self.floor_s)
        return median(self._recent) > threshold
