from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ngllib_agent.providers import FlywireSkeletonProvider


@pytest.fixture
def parquet(tmp_path):
    # two segments; known z_max per segment
    root_id = ["A"] * 3 + ["B"] * 2
    x = [1.0, 2.0, 3.0, 10.0, 11.0]
    y = [1.0, 2.0, 3.0, 10.0, 11.0]
    z = [5.0, 50.0, 25.0, 7.0, 3.0]  # A max=50, B max=7
    tbl = pa.table(
        {
            "root_id": pa.array(root_id, pa.string()),
            "x": pa.array(x, pa.float32()),
            "y": pa.array(y, pa.float32()),
            "z": pa.array(z, pa.float32()),
        }
    )
    p = tmp_path / "seg.parquet"
    pq.write_table(tbl, p)
    return str(p)


def test_root_ids_discovered(parquet):
    prov = FlywireSkeletonProvider(parquet)
    assert set(prov.root_ids) == {"A", "B"}


def test_call_returns_valid_state_and_task_info(parquet):
    prov = FlywireSkeletonProvider(parquet)
    rng = np.random.default_rng(0)
    state, task_info = prov(rng, None)

    assert set(task_info) == {"segment_id", "z_max"}
    rid = task_info["segment_id"]
    assert rid in {"A", "B"}
    assert task_info["z_max"] == (50.0 if rid == "A" else 7.0)

    assert state["segments"] == [rid]
    assert len(state["position"]) == 3
    assert len(state["projectionOrientation"]) == 4  # quaternion
    # start position is one of the segment's nodes
    assert isinstance(state["position"][2], float)


def test_call_respects_segment_id_option(parquet):
    prov = FlywireSkeletonProvider(parquet)
    rng = np.random.default_rng(1)
    state, task_info = prov(rng, {"segment_id": "B"})
    assert task_info["segment_id"] == "B"
    assert task_info["z_max"] == 7.0


def test_seeded_call_is_reproducible(parquet):
    prov = FlywireSkeletonProvider(parquet)
    a = prov(np.random.default_rng(42), None)
    b = prov(np.random.default_rng(42), None)
    assert a[1] == b[1]
    assert a[0]["position"] == b[0]["position"]
    assert a[0]["projectionOrientation"] == b[0]["projectionOrientation"]


def test_task_info_from_state(parquet):
    prov = FlywireSkeletonProvider(parquet)
    ti = prov.task_info_from_state({"segments": ["A"], "position": [1.0, 1.0, 5.0]})
    assert ti == {"segment_id": "A", "z_max": 50.0}


def test_task_info_from_state_requires_segments(parquet):
    prov = FlywireSkeletonProvider(parquet)
    with pytest.raises(NotImplementedError):
        prov.task_info_from_state({"position": [0, 0, 0]})
    with pytest.raises(NotImplementedError):
        prov.task_info_from_state("https://example/#!{}")

def test_exclude_root_ids_holdout(parquet):
    prov = FlywireSkeletonProvider(parquet, exclude_root_ids=["B"])
    assert prov.root_ids == ["A"]
    rng = np.random.default_rng(0)
    for _ in range(5):
        _, task_info = prov(rng, None)
        assert task_info["segment_id"] == "A"
    # explicit segment_id (the eval path) bypasses the holdout on purpose
    _, task_info = prov(rng, {"segment_id": "B"})
    assert task_info["segment_id"] == "B"


def test_exclude_all_raises(parquet):
    with pytest.raises(ValueError):
        FlywireSkeletonProvider(parquet, exclude_root_ids=["A", "B"])
