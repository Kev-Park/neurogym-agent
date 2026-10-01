from __future__ import annotations

import os

import numpy as np
import pytest

# Under `uv run`, ray.init auto-ships the CWD (1.2GB repo) as a runtime_env and
# fails on the 512MB cap — same fix as ppo_smoke.py.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

torch = pytest.importorskip("torch")
ray = pytest.importorskip("ray")

import gymnasium as gym
from gymnasium import spaces

from ngllib_agent.policies import HierarchicalPPOModule

NVEC = [3, 1024, 9, 9, 9, 9]
IMG_DIM, POS_DIM = 768, 8

OBS_SPACE = spaces.Dict(
    {
        "image_features": spaces.Box(-np.inf, np.inf, (IMG_DIM,), np.float32),
        "pos_state": spaces.Box(-np.inf, np.inf, (POS_DIM,), np.float32),
    }
)
ACT_SPACE = spaces.MultiDiscrete(NVEC)


def _module(**kw):
    return HierarchicalPPOModule(
        observation_space=OBS_SPACE,
        action_space=ACT_SPACE,
        model_config={"pos_hidden_dim": 32, "trunk_hiddens": [64]},
        **kw,
    )


def _batch(b=5):
    from ray.rllib.core.columns import Columns

    return {
        Columns.OBS: {
            "image_features": torch.randn(b, IMG_DIM),
            "pos_state": torch.randn(b, POS_DIM),
        }
    }


def test_forward_shapes():
    from ray.rllib.core.columns import Columns

    m = _module()
    out = m.forward_inference(_batch())
    assert out[Columns.ACTION_DIST_INPUTS].shape == (5, sum(NVEC))
    out = m.forward_exploration(_batch())
    assert out[Columns.ACTION_DIST_INPUTS].shape == (5, sum(NVEC))
    out = m.forward_train(_batch())
    assert Columns.EMBEDDINGS in out
    assert out[Columns.ACTION_DIST_INPUTS].shape == (5, sum(NVEC))


def test_compute_values():
    m = _module()
    v = m.compute_values(_batch())
    assert v.shape == (5,)
    # and via precomputed embeddings
    from ray.rllib.core.columns import Columns

    out = m.forward_train(_batch())
    v2 = m.compute_values(_batch(), embeddings=out[Columns.EMBEDDINGS])
    assert v2.shape == (5,)


def test_inference_only_has_no_vf():
    m = _module(inference_only=True)
    assert not hasattr(m, "_vf_head")
    m.forward_inference(_batch())  # still computes actions


def test_dist_cls_bound_to_nvec():
    m = _module()
    dist_cls = m.get_inference_action_dist_cls()
    logits = torch.randn(2, sum(NVEC))
    dist = dist_cls.from_logits(logits)
    assert dist.sample().shape == (2, 6)


PANES, GRID, PDIM = 2, 4, 6
NVEC_2PANE = [5, GRID * (PANES * GRID), 9, 9, 9, 9]   # one cell per token, panes side by side
SPATIAL_OBS_SPACE = spaces.Dict(
    {
        "pos_state": spaces.Box(-np.inf, np.inf, (POS_DIM,), np.float32),
        "patch_features": spaces.Box(-np.inf, np.inf, (PANES, GRID, GRID, PDIM), np.float16),
    }
)


def _spatial_module(nvec=NVEC_2PANE, **cfg):
    return HierarchicalPPOModule(
        observation_space=SPATIAL_OBS_SPACE,
        action_space=spaces.MultiDiscrete(nvec),
        model_config={"pos_hidden_dim": 16, "spatial_channels": 8, "spatial_blocks": 2,
                      "flat_channels": 4, "hidden": 32, **cfg},
    )


def _spatial_batch(b=3):
    from ray.rllib.core.columns import Columns

    return {
        Columns.OBS: {
            "pos_state": torch.randn(b, POS_DIM),
            "patch_features": torch.randn(b, PANES, GRID, GRID, PDIM).half(),
        }
    }


def test_spatial_forward_shapes_and_values():
    from ray.rllib.core.columns import Columns

    m = _spatial_module()
    assert m._grid == (PANES, GRID, GRID)
    out = m.forward_train(_spatial_batch())
    assert out[Columns.ACTION_DIST_INPUTS].shape == (3, sum(NVEC_2PANE))
    assert out[Columns.ACTION_DIST_INPUTS].dtype == torch.float32
    assert m.compute_values(_spatial_batch()).shape == (3,)
    dist = m.get_inference_action_dist_cls().from_logits(out[Columns.ACTION_DIST_INPUTS])
    assert dist.sample().shape == (3, 6)


def test_spatial_cell_logits_are_local():
    """A perturbation in one pane's token grid must move that pane's cell
    logits more than the other pane's (the click head is a map, not pooled)."""
    from ray.rllib.core.columns import Columns

    torch.manual_seed(0)
    m = _spatial_module()
    base = _spatial_batch(1)
    pert = {Columns.OBS: {k: v.clone() for k, v in base[Columns.OBS].items()}}
    pert[Columns.OBS]["patch_features"][0, 1] += 5.0          # 3D pane only
    n_verb = NVEC_2PANE[0]
    d = (m.forward_inference(pert)[Columns.ACTION_DIST_INPUTS]
         - m.forward_inference(base)[Columns.ACTION_DIST_INPUTS])[0, n_verb:n_verb + NVEC_2PANE[1]]
    d = d.abs().reshape(GRID, PANES * GRID)
    assert d[:, GRID:].mean() > d[:, :GRID].mean()


def test_spatial_cell_layout_is_row_major_over_side_by_side_panes():
    """cell index = row * (P*G) + pane * G + col, matching ActionSpec.cell_to_pixel
    over click_bounds that span EM then 3D."""
    from ray.rllib.core.columns import Columns

    m = _spatial_module()
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        # score = conv1x1([fmap; ctx]) with all-zero weights, so set the bias
        # to 0 and make the score read a single fmap channel: weight[0,0]=1.
        m._cell_score.weight[0, 0, 0, 0] = 1.0
        # fmap = in_proj(x) + pane_embed + blocks(=0 with zero weights, GN affine 0)
        m._in_proj.weight[0, 0, 0, 0] = 1.0            # fmap[0] = x[..., 0]
    b = _spatial_batch(1)
    x = torch.zeros(1, PANES, GRID, GRID, PDIM)
    x[0, 1, 2, 3, 0] = 7.0                              # 3D pane, row 2, col 3
    b[Columns.OBS]["patch_features"] = x.half()
    logits = m.forward_inference(b)[Columns.ACTION_DIST_INPUTS][0]
    cell = logits[NVEC_2PANE[0]:NVEC_2PANE[0] + NVEC_2PANE[1]]
    assert int(cell.argmax()) == 2 * (PANES * GRID) + 1 * GRID + 3


def test_spatial_click_grid_must_match_tokens():
    with pytest.raises(ValueError):
        _spatial_module(nvec=[5, 100, 9, 9, 9, 9])


def test_cell_entropy_scale():
    from ngllib_agent.policies.hierarchical import HierarchicalMultiCategorical

    logits = torch.randn(4, sum(NVEC))
    h1 = HierarchicalMultiCategorical.for_nvec(NVEC, normalize_entropy=True).from_logits(logits).entropy()
    h0 = HierarchicalMultiCategorical.for_nvec(
        NVEC, normalize_entropy=True, cell_entropy_scale=0.0).from_logits(logits).entropy()
    assert torch.all(h0 < h1)


class _FakeZNavEnv(gym.Env):
    """Same spaces as the DINO-wrapped env; reward loosely favors action_type=2."""

    def __init__(self, config=None):
        self.observation_space = OBS_SPACE
        self.action_space = ACT_SPACE
        self._t = 0

    def _obs(self):
        return {
            "image_features": np.random.randn(IMG_DIM).astype(np.float32),
            "pos_state": np.random.randn(POS_DIM).astype(np.float32),
        }

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        reward = 1.0 if int(action[0]) == 2 else 0.0
        return self._obs(), reward, False, self._t >= 20, {}


class _FakeSpatialEnv(_FakeZNavEnv):
    def __init__(self, config=None):
        super().__init__(config)
        self.observation_space = SPATIAL_OBS_SPACE
        self.action_space = spaces.MultiDiscrete(NVEC_2PANE)

    def _obs(self):
        return {
            "pos_state": np.random.randn(POS_DIM).astype(np.float32),
            "patch_features": np.random.randn(PANES, GRID, GRID, PDIM).astype(np.float16),
        }


@pytest.mark.parametrize("env_cls", [_FakeZNavEnv, _FakeSpatialEnv])
def test_ppo_one_iter_end_to_end(env_cls):
    """Full PPO train iteration through the custom module + gated distribution."""
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    ray.init(include_dashboard=False, num_cpus=2, ignore_reinit_error=True)
    try:
        config = (
            PPOConfig()
            .environment(env_cls)
            .framework("torch")
            .env_runners(num_env_runners=0, rollout_fragment_length="auto")
            .learners(num_learners=0)
            .training(train_batch_size=64, minibatch_size=32, num_epochs=1)
            .rl_module(
                rl_module_spec=RLModuleSpec(
                    module_class=HierarchicalPPOModule,
                    model_config={"pos_hidden_dim": 16, "trunk_hiddens": [32],
                                  "spatial_channels": 8, "spatial_blocks": 1,
                                  "flat_channels": 4, "hidden": 32},
                )
            )
        )
        algo = config.build_algo() if hasattr(config, "build_algo") else config.build()
        result = algo.train()
        pol = (result.get("learners", {}) or {}).get("default_policy", {}) or {}
        assert pol.get("total_loss") is not None
        assert np.isfinite(float(pol["total_loss"]))
        algo.stop()
    finally:
        ray.shutdown()
