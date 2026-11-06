import copy
import json

import pytest

from ngllib_agent.data import load_manifest, resolve_task_config, validate_scene, write_manifest
from ngllib_agent.tasks import TaskComponents, build_task
from ngllib_agent.distributed.checkpoint import AsyncCheckpointer, resume_metadata


def custom_task(cfg, *, manifest):
    return TaskComponents(None, lambda task: lambda obs, a, previous, done: 0.,
                          lambda task: lambda obs, a, previous: False)


def test_custom_factory_does_not_require_a_flywire_table():
    cfg, manifest = resolve_task_config({"env": {}, "task": {"factory": __name__ + ":custom_task"}})
    assert isinstance(build_task(cfg, manifest), TaskComponents)
    assert "parquet_path" not in cfg["env"]


def test_manifest_relocatable_verified_and_dataset_checked(tmp_path):
    source = tmp_path / "table.parquet"
    source.write_bytes(b"fixture")
    path = tmp_path / "manifest.json"
    write_manifest(path, dataset={"id": "test", "voxel_nm": [4., 4., 40.]},
                   coordinates={"frame": "viewer_voxel", "units": "voxel"},
                   precomputation={"seed": 12}, artifacts={"reset_data": source})
    cfg, manifest = resolve_task_config({"env": {}, "task": {"manifest": str(path)}})
    assert cfg["env"]["parquet_path"] == str(source)
    assert cfg["task"]["data_identity"]["artifacts"]["reset_data"]
    from ngllib import DatasetSpec
    with pytest.raises(ValueError, match="disagrees"):
        validate_scene(manifest, DatasetSpec("image", "seg", (8., 8., 80.)))
    source.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="changed"):
        load_manifest(path)


def test_resume_rejects_changed_table_even_with_explicit_resume(tmp_path):
    source = tmp_path / "table.parquet"
    source.write_bytes(b"first")
    cfg, _ = resolve_task_config({"env": {"parquet_path": str(source)}})
    class Algo:
        def get_state(self):
            return {}
    AsyncCheckpointer(tmp_path, every=1, use_actor=False).maybe_save(Algo(), 1, {"config": cfg})
    source.write_bytes(b"different")
    changed, _ = resolve_task_config({"env": {"parquet_path": str(source)}})
    with pytest.raises(ValueError, match="task data changed"):
        resume_metadata(tmp_path, changed, explicit=True)
    assert resume_metadata(tmp_path, changed, explicit=True, allow_data_change=True)[0]


def test_provider_override_is_independent_of_reward_selection():
    cfg = {"env": {}, "reward": {"mode": "z_navigate", "z_tolerance": 1.,
            "success": 1., "z_shaping_coef": .001, "step_penalty": 0.},
           "task": {"provider_factory": __name__ + ":custom_provider"}}
    cfg, manifest = resolve_task_config(cfg)
    task = build_task(cfg, manifest)
    state, info = task.reset_state_provider(None, {})
    assert info["z_max"] == 10


def custom_provider(cfg, *, manifest):
    return lambda rng, options: ({}, {"z_max": 10})
