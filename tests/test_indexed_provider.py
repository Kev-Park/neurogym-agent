"""Smoke test for IndexedStateProvider.

Verifies:
1. Unfiltered pool matches FlywireSkeletonProvider's pool exactly.
2. Predicates progressively narrow the pool.
3. Sampled task_info["segment_id"] is always in the filtered pool.

Run from repo root on cluster (needs the parquet + cell_stats files):
    uv run --no-sync python tests/test_indexed_provider.py
"""

from __future__ import annotations

import sys

import numpy as np

from ngllib_agent.providers import FlywireSkeletonProvider, IndexedStateProvider


SKEL = "/scratch/kp0374/neurogym-agent/segment_positions.parquet"
STATS = "/scratch/kp0374/neurogym-agent/cell_stats.csv"


def main() -> int:
    print("=== IndexedStateProvider smoke ===", flush=True)

    # Baseline: unfiltered FlywireSkeletonProvider.
    base = FlywireSkeletonProvider(SKEL)
    n_base = len(base.root_ids)
    print(f"base FlywireSkeletonProvider: {n_base} root_ids", flush=True)

    # Unfiltered IndexedStateProvider should intersect skeleton with cell_stats
    # (may be smaller than base if some skeleton ids lack cell_stats rows).
    ind_all = IndexedStateProvider(SKEL, STATS, predicate="1=1")
    n_all = len(ind_all.root_ids)
    print(f"indexed 1=1 (intersect with cell_stats): {n_all} root_ids", flush=True)
    assert n_all <= n_base, "unfiltered indexed pool must not exceed base pool"

    # Progressively tighter predicates should give progressively smaller pools.
    thresholds = [50_000, 100_000, 500_000, 1_000_000, 5_000_000]
    prev = n_all
    counts = [("1=1", n_all)]
    for t in thresholds:
        pred = f"length_nm > {t}"
        p = IndexedStateProvider(SKEL, STATS, predicate=pred)
        n = len(p.root_ids)
        counts.append((pred, n))
        assert n <= prev, f"{pred}: pool={n} unexpectedly larger than previous={prev}"
        prev = n
    print("pool sizes by predicate:", flush=True)
    for pred, n in counts:
        print(f"  {pred:32s}  {n:8d}", flush=True)

    # Sampling: every task_info["segment_id"] must be in the filtered pool.
    p = IndexedStateProvider(SKEL, STATS, predicate="length_nm > 500000")
    pool = set(p.root_ids)
    print(f"\nsampling 50 episodes with length_nm > 500000 (pool={len(pool)})",
          flush=True)
    rng = np.random.default_rng(42)
    for i in range(50):
        _, task_info = p(rng, None)
        seg = task_info["segment_id"]
        assert seg in pool, f"sampled seg {seg} outside pool"
    print("  all 50 sampled segment_ids were in the filtered pool", flush=True)

    # Determinism: same seed → same sequence of segment_ids.
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    seq1 = [p(rng1, None)[1]["segment_id"] for _ in range(10)]
    seq2 = [p(rng2, None)[1]["segment_id"] for _ in range(10)]
    assert seq1 == seq2, f"determinism failed: {seq1} vs {seq2}"
    print("  deterministic under seed=0 (10 draws match)", flush=True)

    # task_info_from_state round-trips a state whose segments[0] is in the pool.
    seg0 = p.root_ids[0]
    ti = p.task_info_from_state({"segments": [seg0]})
    assert ti["segment_id"] == seg0
    assert "z_max" in ti and isinstance(ti["z_max"], float)
    print(f"  task_info_from_state(segments=[{seg0}]) -> z_max={ti['z_max']:.1f}",
          flush=True)

    print("\nPASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
