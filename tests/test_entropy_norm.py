from __future__ import annotations

import torch

from ngllib_agent.policies.hierarchical import HierarchicalMultiCategorical

NVEC = [3, 1024, 9, 9, 9, 9]


def _logits(verb):
    parts = [torch.tensor([verb], dtype=torch.float32),
             torch.zeros(1, 1024), torch.zeros(1, 9),
             torch.zeros(1, 9), torch.zeros(1, 9), torch.zeros(1, 9)]
    return torch.cat(parts, dim=-1)


def test_normalized_entropy_is_verb_neutral():
    # Uniform param heads, verb one-hot: every branch offers the same
    # normalized bonus, so verb allocation must not change the entropy.
    cls = HierarchicalMultiCategorical.for_nvec(NVEC, normalize_entropy=True)
    e_click = cls.from_logits(_logits([50.0, 0.0, 0.0])).entropy()
    e_zoom = cls.from_logits(_logits([0.0, 0.0, 50.0])).entropy()
    assert torch.allclose(e_click, e_zoom, atol=1e-4)
    assert abs(e_click.item() - 1.0) < 1e-3  # ~0 verb H + 1.0 branch


def test_unnormalized_entropy_favors_click():
    # The v7 pathology: the click branch offers ln1024 vs zoom's ln9.
    cls = HierarchicalMultiCategorical.for_nvec(NVEC)
    e_click = cls.from_logits(_logits([50.0, 0.0, 0.0])).entropy().item()
    e_zoom = cls.from_logits(_logits([0.0, 0.0, 50.0])).entropy().item()
    assert e_click > e_zoom + 3.0


def test_logp_and_kl_unaffected_by_normalization():
    a = torch.tensor([[2, 5, 1, 2, 3, 4]])
    dn = HierarchicalMultiCategorical.for_nvec(NVEC, normalize_entropy=True)
    du = HierarchicalMultiCategorical.for_nvec(NVEC)
    n1, n2 = dn.from_logits(_logits([0.5, 0.0, 0.0])), dn.from_logits(_logits([0.0, 0.5, 0.0]))
    u1, u2 = du.from_logits(_logits([0.5, 0.0, 0.0])), du.from_logits(_logits([0.0, 0.5, 0.0]))
    assert torch.allclose(n1.logp(a), u1.logp(a))
    assert torch.allclose(n1.kl(n2), u1.kl(u2))
