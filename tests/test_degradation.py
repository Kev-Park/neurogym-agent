from __future__ import annotations

from ngllib_agent.degradation import DegradationDetector


def _feed(det, times):
    return [det.observe(t) for t in times]


def test_cruise_never_trips():
    det = DegradationDetector()
    assert not any(_feed(det, [40.0] * 200))


def test_isolated_hang_recovery_iters_invisible():
    # v7's benign pattern: 27 single ~400s iters surrounded by 40s cruise.
    det = DegradationDetector()
    times = []
    for i in range(200):
        times.append(430.0 if i % 12 == 0 else 40.0)
    assert not any(_feed(det, times))


def test_two_adjacent_spikes_still_no_trip():
    # Median-of-5 needs >=3 slow entries; back-to-back recoveries stay benign.
    det = DegradationDetector()
    assert not any(_feed(det, [40.0] * 50 + [400.0, 400.0] + [40.0] * 50))


def test_sustained_degradation_trips_within_window():
    # The v7 blemish: ~40s cruise then ~360s churn. Must trip at the 3rd
    # degraded iteration (median-of-5 flips), not 100 minutes later.
    det = DegradationDetector()
    assert not any(_feed(det, [40.0] * 250))
    trips = _feed(det, [360.0] * 6)
    assert trips[2] and not any(trips[:2])


def test_uniformly_slow_run_never_trips():
    # A slow config is not a runtime pathology — baseline tracks it.
    det = DegradationDetector()
    assert not any(_feed(det, [360.0] * 200))


def test_floor_suppresses_mild_degradation():
    # 40 -> 100s is 2.5x but under the 180s floor: cheaper to ride out than
    # to pay a restart.
    det = DegradationDetector()
    assert not any(_feed(det, [40.0] * 100 + [100.0] * 50))


def test_no_trip_during_warmup():
    # Degraded from the very start: warmup absorbs it into the baseline
    # rather than tripping on garbage statistics.
    det = DegradationDetector(warmup=10)
    trips = _feed(det, [400.0] * 9)
    assert not any(trips)


def test_baseline_property():
    det = DegradationDetector()
    assert det.baseline_s is None
    _feed(det, [40.0, 60.0, 50.0])
    assert det.baseline_s == 50.0
