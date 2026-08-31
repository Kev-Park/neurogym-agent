"""MeshStore's decoded-mesh cache must stay byte-bounded.

It was an unbounded dict until 2026-08-31: every root_id ever visited stayed
resident, and a 32-env single-node run hit 206 GiB RSS and was cgroup-killed
at iteration 410 of 740 with no traceback (SIGKILL leaves none). The bound is
per store and local mode builds one per env, so this also guards the
NGL_NATIVE_MESH_LRU_MB knob the slurm scripts set.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


class _FakeMesh:
    def __init__(self, n):
        self.vertices = np.zeros((n, 3), "f4")
        self.faces = np.zeros((n, 3), "i4")


class _FakeVol:
    """CloudVolume stand-in; counts fetches so hits are observable."""

    def __init__(self, n=10_000):
        self.fetches = 0
        outer = self

        class _Mesh:
            @staticmethod
            def get(rid):
                outer.fetches += 1
                return {rid: _FakeMesh(n)}

        self.mesh = _Mesh()


def _store(budget_bytes, vol):
    from ngllib.native.em import MeshStore

    s = object.__new__(MeshStore)   # skip CloudVolume construction
    s._vol = vol
    s._meshes = OrderedDict()
    s._bytes = 0
    s._budget = budget_bytes
    return s


def test_evicts_least_recently_used_and_stays_under_budget():
    vol = _FakeVol()
    per_mesh = 10_000 * 3 * 4 * 2          # verts f4 + faces i4
    s = _store(4 * per_mesh, vol)

    for rid in range(6):
        s.get(rid)

    assert s.cached_bytes <= 4 * per_mesh
    assert len(s._meshes) == 4
    assert list(s._meshes) == [2, 3, 4, 5]   # 0 and 1 evicted
    assert vol.fetches == 6


def test_hit_does_not_refetch_and_renews_recency():
    vol = _FakeVol()
    per_mesh = 10_000 * 3 * 4 * 2
    s = _store(3 * per_mesh, vol)

    s.get("a"); s.get("b"); s.get("c")
    s.get("a")                       # hit: no fetch, "a" becomes newest
    assert vol.fetches == 3
    s.get("d")                       # evicts "b", not "a"
    assert set(s._meshes) == {"a", "c", "d"}


def test_never_evicts_the_entry_just_inserted():
    """A single mesh larger than the whole budget must still be returned."""
    vol = _FakeVol()
    s = _store(1024, vol)            # budget far below one mesh
    v, f = s.get("big")
    assert v.shape == (10_000, 3)
    assert list(s._meshes) == ["big"]


def test_drop_reclaims_bytes():
    vol = _FakeVol()
    s = _store(10 << 20, vol)
    s.get("x")
    assert s.cached_bytes > 0
    s.drop("x")
    assert s.cached_bytes == 0
    s.drop("x")                      # idempotent
    assert s.cached_bytes == 0
