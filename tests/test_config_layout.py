import pytest

from ngllib_agent.env_build import dino_layout, load_config
from ngllib_agent.train import build_argparser


def test_dino_layout_rejects_ignored_explicit_settings():
    assert dino_layout({}, {"use_left_pane": False})["left_pane"] is False
    for config in ({"left_pane": True}, {"right_pane": False}, {"image_size": [84, 84]}):
        with pytest.raises(ValueError):
            dino_layout(config, {"use_left_pane": False})


def test_packaged_default_has_explicit_independent_entropy_settings():
    config = load_config(build_argparser().parse_args([]).config)
    for key in ("cell_entropy_scale", "dbl_entropy_scale", "verb_entropy_scale"):
        assert config["model"][key] == 1.0
    assert config["model"]["cell_entropy_stopgrad"] is False
    assert dino_layout(config["env"], config["obs"])["left_pane"] is False
