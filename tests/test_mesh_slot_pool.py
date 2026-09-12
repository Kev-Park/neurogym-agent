"""MeshRenderer's mesh storage must not allocate GL objects per mesh.

The driver does not hand released VRAM back to the process: with one
alloc/release pair per mesh, a simulator env leaked ~half of every evicted
mesh (2026-09-12: one env, 12 min, torch flat, VAO LRU bounded at 200 MiB,
process GL memory +670 MiB). Storage is therefore a bounded pool of slots
whose capacities only grow, so GL allocation events are O(slots + growths),
never O(episodes). These tests count exactly those events through a fake
context -- no EGL needed.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


class _Buf:
    def __init__(self, log, size):
        self.log, self.size, self.writes = log, size, 0
        log.append("alloc")

    def orphan(self, size):
        self.log.append("orphan"); self.size = size

    def write(self, data):
        assert len(data) <= self.size, "write past capacity"
        self.writes += 1


class _Vao:
    def __init__(self, log):
        self.log, self.rendered = log, []
        log.append("vao")

    def render(self, mode, vertices):
        self.rendered.append(vertices)


class _Ctx:
    def __init__(self):
        self.log = []

    def buffer(self, data=None, reserve=0):
        return _Buf(self.log, reserve if data is None else len(data))

    def vertex_array(self, prog, content, index_buffer=None):
        return _Vao(self.log)


def _renderer(budget_bytes):
    from ngllib.native.render3d import MeshRenderer

    r = object.__new__(MeshRenderer)    # skip EGL context creation
    r.ctx = _Ctx()
    r.prog = {"color": type("U", (), {"value": None})()}
    r._budget = budget_bytes
    r._n_slots = max(4, min(64, budget_bytes // MeshRenderer.SLOT_NOMINAL_BYTES))
    r._slots = []
    r._vaos = OrderedDict()
    return r


def _mesh(n_verts, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.random((n_verts, 3), dtype=np.float32)
    f = rng.integers(0, n_verts, size=(n_verts, 3), dtype=np.int32)
    return v, f, np.zeros_like(v)


def _allocs(r):
    return r.ctx.log.count("alloc") + r.ctx.log.count("vao")


def test_slot_count_follows_budget():
    from ngllib.native.render3d import MeshRenderer

    assert _renderer(200 << 20)._n_slots == 6
    assert _renderer(2 << 30)._n_slots == 64          # capped
    assert _renderer(1 << 20)._n_slots == 4           # floor
    assert MeshRenderer.SLOT_NOMINAL_BYTES == 32 << 20


def test_many_episodes_allocate_only_n_slots():
    """The invariant the leak fix rests on: 100 distinct meshes through a
    4-slot pool must create exactly 4 slots' worth of GL objects."""
    r = _renderer(1 << 20)                            # 4 slots
    v, f, vn = _mesh(1000)
    for rid in range(100):
        r.load_mesh(str(rid), v, f, normals=vn)
    assert len(r._slots) == 4
    assert _allocs(r) == 4 * 3                        # vbo + ibo + vao each
    # every slot grows once on first fill (vbo + ibo), then never again
    assert r.ctx.log.count("orphan") == 4 * 2
    assert len(r._vaos) == 4                          # only the last 4 resident
    assert set(r._vaos) == {"96", "97", "98", "99"}


def test_eviction_is_least_recently_drawn():
    r = _renderer(1 << 20)
    v, f, vn = _mesh(500)
    for rid in "abcd":
        r.load_mesh(rid, v, f, normals=vn)
    r._draw_meshes(["a"], [(1, 1, 1)])                # a becomes newest
    r.load_mesh("e", v, f, normals=vn)                # evicts b, not a
    assert "a" in r._vaos and "b" not in r._vaos


def test_capacity_grows_monotonically_and_only_when_exceeded():
    r = _renderer(1 << 20)
    small_v, small_f, small_n = _mesh(100)
    big_v, big_f, big_n = _mesh(5000)
    r.load_mesh("s", small_v, small_f, normals=small_n)
    sid = r._vaos["s"][0]
    cap0 = r._slots[sid]["vcap"]
    n_orphan = r.ctx.log.count("orphan")
    r.load_mesh("s", big_v, big_f, normals=big_n, replace=True)
    assert r._vaos["s"][0] == sid                     # same slot, refined in place
    assert r._slots[sid]["vcap"] > cap0
    assert r.ctx.log.count("orphan") == n_orphan + 2  # vbo and ibo grew once
    n_orphan = r.ctx.log.count("orphan")
    r.load_mesh("s", small_v, small_f, normals=small_n, replace=True)
    assert r.ctx.log.count("orphan") == n_orphan      # shrinking never reallocates


def test_replace_keeps_slot_and_updates_index_count():
    r = _renderer(1 << 20)
    v1, f1, n1 = _mesh(300, seed=1)
    v2, f2, n2 = _mesh(700, seed=2)
    r.load_mesh("x", v1, f1, normals=n1)
    r.load_mesh("x", v2, f2, normals=n2, replace=True)
    sid, n_idx = r._vaos["x"]
    assert n_idx == f2.size
    r._draw_meshes(["x"], [(1, 0, 0)])
    assert r._slots[sid]["vao"].rendered == [f2.size]


def test_repeat_load_without_replace_is_a_noop_touch():
    r = _renderer(1 << 20)
    v, f, vn = _mesh(200)
    r.load_mesh("x", v, f, normals=vn)
    n = _allocs(r); writes = r._slots[0]["vbo"].writes
    r.load_mesh("x", v, f, normals=vn)
    assert _allocs(r) == n and r._slots[0]["vbo"].writes == writes


def test_vao_bytes_reports_slot_capacity():
    r = _renderer(1 << 20)
    assert r._vao_bytes == 0
    v, f, vn = _mesh(1000)
    r.load_mesh("x", v, f, normals=vn)
    assert r._vao_bytes == v.nbytes * 2 + f.nbytes   # pos+nrm floats, indices
