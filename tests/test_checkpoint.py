from __future__ import annotations

import pickle
import pytest

from ngllib_agent.distributed.checkpoint import (
    AsyncCheckpointer,
    atomic_pickle,
    latest_checkpoint,
    load_checkpoint,
    resume_metadata,
    public_config,
)


class _FakeAlgo:
    def __init__(self):
        self.n = 0

    def get_state(self):
        self.n += 1
        return {"weights": [1, 2, 3], "snapshot": self.n}


def test_atomic_pickle_roundtrip(tmp_path):
    p = tmp_path / "ckpt_000001.pkl"
    atomic_pickle({"a": 1}, p)
    assert load_checkpoint(p) == {"a": 1}
    assert not p.with_suffix(".pkl.tmp").exists()  # tmp renamed away


def test_latest_checkpoint_ordering(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    for it in (10, 2, 30):
        atomic_pickle({"it": it}, tmp_path / f"ckpt_{it:06d}.pkl")
    (tmp_path / "ckpt_garbage.pkl.tmp").write_bytes(b"x")  # ignored
    best = latest_checkpoint(tmp_path)
    assert best is not None and best.name == "ckpt_000030.pkl"


def test_checkpointer_cadence_and_meta(tmp_path):
    ck = AsyncCheckpointer(tmp_path, every=5, use_actor=False)
    algo = _FakeAlgo()
    assert ck.maybe_save(algo, 3) is None            # off-cadence
    path = ck.maybe_save(algo, 5, meta={"wandb_id": "abc"})
    assert path is not None and load_checkpoint(path)["snapshot"] == algo.n
    meta = (tmp_path / "meta.json").read_text()
    assert '"wandb_id": "abc"' in meta and '"iteration": 5' in meta
    ck.finalize()  # no-op on sync path


def test_interrupted_write_preserves_previous(tmp_path):
    ck = AsyncCheckpointer(tmp_path, every=1, use_actor=False)
    algo = _FakeAlgo()
    ck.maybe_save(algo, 1)
    # simulate a crashed later write: orphan tmp file only
    (tmp_path / "ckpt_000002.pkl.tmp").write_bytes(b"partial")
    best = latest_checkpoint(tmp_path)
    assert best.name == "ckpt_000001.pkl"
    assert pickle.loads(best.read_bytes())["weights"] == [1, 2, 3]


def test_resume_uses_checkpoint_metadata_not_newer_heartbeat(tmp_path):
    cfg = {"env": {"parquet_path": "same.parquet"}}
    ck = AsyncCheckpointer(tmp_path, every=1, use_actor=False)
    ck.maybe_save(_FakeAlgo(), 1, {"config": cfg, "total_steps": 128})
    (tmp_path / "meta.json").write_text('{"total_steps": 9999}')
    path, metadata = resume_metadata(tmp_path, cfg)
    assert path.name == "ckpt_000001.pkl"
    assert metadata["total_steps"] == 128
    with pytest.raises(ValueError, match="different or unpaired"):
        resume_metadata(tmp_path, {"env": {"parquet_path": "other.parquet"}})
    assert resume_metadata(tmp_path, {}, explicit=True)[0] == path


def test_resume_requires_explicit_legacy_config_and_redacts_credentials(tmp_path):
    assert resume_metadata(tmp_path, {}) == (None, {})
    path = tmp_path / "ckpt_000001.pkl"
    atomic_pickle({}, path)
    with pytest.raises(ValueError, match="different or unpaired"):
        resume_metadata(tmp_path, {})
    assert resume_metadata(tmp_path, {}, explicit=True) == (path, {})
    assert public_config({"obs": {"patch_tokens": True}, "cave_token": "secret"}) == {
        "obs": {"patch_tokens": True}}


def test_make_env_creator_single_env(monkeypatch, tmp_path):
    import ngllib_agent.env_build as eb

    built = []
    monkeypatch.setattr(eb, "build_env", lambda cfg: built.append(cfg) or "ENV")
    creator = eb.make_env_creator({"k": 1})
    assert creator() == "ENV"                       # no num_envs -> single env
    assert creator({"num_envs": 1}) == "ENV"        # M=1 -> single env
    assert built == [{"k": 1}, {"k": 1}]
