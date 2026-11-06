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


def test_frozen_termination_seed_is_independent_of_factory_rng():
    from ngllib_agent.rewards.z_free import make_v3_termination_factory

    horizons = []
    for seed in (1, 2):
        terminal = make_v3_termination_factory(10, 30, stall_limit=None, seed=seed)(
            {"termination_seed": 1234})
        for step in range(1, 31):
            if terminal({"position": [0, 0, step]}, None, {"position": [0, 0, step - 1]}):
                horizons.append(step)
                break
    assert len(horizons) == 2 and horizons[0] == horizons[1]
