"""Per-process cache budgets must be sizable per node.

Every simulator cache is bounded PER PROCESS, and a node runs 64-192 of
them. The 2026-09-12 topology sweep found 64 runners x 2 envs (FW=2) reaching
~108 GB host RAM per GPU step (CloudVolume chunk LRUs: 256 MB x ~10 handles x
3 processes per runner) and ~1 GB VRAM per process climbing at ~1.2 GB/h
(mesh VAO LRU: 2 GB default, no knob) -- both cgroup/CUDA-killed with no
traceback. Production 32x1 FW=1 survived on a third of the processes.

These guard the two knobs added for that, alongside NGL_NATIVE_MESH_LRU_MB
(see test_mesh_store_lru.py): the slurm scripts size all three per node.
"""

from __future__ import annotations


def test_chunk_lru_default_is_256mb_per_handle(monkeypatch):
    from ngllib.native.em import EMTiles

    monkeypatch.delenv("NGL_NATIVE_CHUNK_LRU_MB", raising=False)
    assert EMTiles.lru_bytes() == 256 << 20


def test_chunk_lru_reads_node_sizing_knob(monkeypatch):
    from ngllib.native.em import EMTiles

    monkeypatch.setenv("NGL_NATIVE_CHUNK_LRU_MB", "48")
    assert EMTiles.lru_bytes() == 48 << 20


def test_vao_budget_default_is_2gb_per_process(monkeypatch):
    from ngllib.native.render3d import MeshRenderer

    monkeypatch.delenv("NGL_NATIVE_VAO_LRU_MB", raising=False)
    assert MeshRenderer.vao_budget_bytes() == 2 << 30


def test_vao_budget_reads_node_sizing_knob(monkeypatch):
    from ngllib.native.render3d import MeshRenderer

    monkeypatch.setenv("NGL_NATIVE_VAO_LRU_MB", "200")
    assert MeshRenderer.vao_budget_bytes() == 200 << 20


def test_vao_budget_explicit_argument_wins_over_env(monkeypatch):
    """A caller that sizes the budget itself (the service did, at 4 GB) must
    not be silently overridden by a node-wide env var."""
    from ngllib.native.render3d import MeshRenderer

    monkeypatch.setenv("NGL_NATIVE_VAO_LRU_MB", "200")
    assert MeshRenderer.vao_budget_bytes(4 << 30) == 4 << 30


def test_environment_no_longer_pins_the_renderer_budget():
    """NativeEnvironment used to default mesh_budget_bytes to 2 GB and pass it
    down, which would have overridden the env knob before it reached the
    renderer. None means 'let the renderer resolve it'."""
    import inspect

    from ngllib.native.environment import NativeEnvironment

    p = inspect.signature(NativeEnvironment.__init__).parameters["mesh_budget_bytes"]
    assert p.default is None
