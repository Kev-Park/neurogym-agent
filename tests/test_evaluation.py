import copy
import json

import pytest

from ngllib_agent.evaluation import checkpoint_config, compare_results


def test_checkpoint_config_requires_legacy_config_and_checks_architecture(tmp_path):
    path = tmp_path / "ckpt_000001.pkl"
    with pytest.raises(ValueError, match="Unpaired"):
        checkpoint_config(path)
    config = {"obs": {"mode": "dino"}, "action": {}, "model": {}, "reward": {"mode": "z_navigate"}}
    path.with_suffix(".json").write_text(json.dumps({"config": config}))
    loaded = checkpoint_config(path)
    loaded["obs"]["mode"] = "pos"
    assert checkpoint_config(path) == config
    with pytest.raises(ValueError, match="architecture"):
        checkpoint_config(path, loaded)


def test_comparison_pairs_ids_and_rejects_incomplete_or_different_protocol():
    left = {"complete": True, "protocol": {"env": {"backend": "simulator"}},
            "episodes": [{"id": "x", "episode_return": 1, "z_gain_voxels": 2}]}
    right = copy.deepcopy(left)
    right["episodes"][0]["z_gain_voxels"] = 5
    assert compare_results(left, right)["x"]["z_gain_voxels"] == 3
    right["protocol"]["env"]["backend"] = "chrome"
    with pytest.raises(ValueError, match="protocols"):
        compare_results(left, right)
    assert compare_results(left, right, True)["x"]["z_gain_voxels"] == 3
    right["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        compare_results(left, right, True)
