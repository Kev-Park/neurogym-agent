"""Check terminal versus truncated GAE and independence across episode chunks."""
import numpy as np

from ngllib_agent.boundary_credit import chunk_advantages
from ray.rllib.utils.postprocessing.value_predictions import compute_value_targets


def test_bootstrap_is_used_only_for_nonterminal_transition():
    rewards = [0., 1.]
    values = [.2, .3, .4]
    assert np.allclose(chunk_advantages(rewards, values, [False, True], .9, 1), [.7, .7])
    assert np.allclose(chunk_advantages(rewards, values, [False, False], .9, 1), [1.024, 1.06])


def test_installed_gae_does_not_leak_across_chunks():
    values = np.asarray([.2, .3, .4, 10., 20., 30.], np.float32)
    rewards = np.asarray([0., 1., 0., 100., 200., 0.], np.float32)
    terminated = np.asarray([0., 0., 0., 0., 1., 0.], np.float32)
    truncated = np.asarray([0., 1., 0., 0., 0., 0.], np.float32)
    targets = compute_value_targets(values, rewards, terminated, truncated, .9, .8)
    expected = chunk_advantages(rewards[:2], values[:3], terminated[:2], .9, .8)
    assert np.allclose(targets[:2] - values[:2], expected)
    changed = rewards.copy()
    changed[3:5] *= 1000
    other = compute_value_targets(values, changed, terminated, truncated, .9, .8)
    assert np.allclose(other[:2], targets[:2])
