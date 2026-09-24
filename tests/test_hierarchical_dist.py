from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ngllib_agent.policies import HierarchicalMultiCategorical

NVEC = [3, 1024, 9, 9, 9, 9]
DIM = sum(NVEC)
DistCls = HierarchicalMultiCategorical.for_nvec(NVEC)


def _dist(logits=None):
    if logits is None:
        logits = torch.randn(4, DIM)
    return DistCls.from_logits(logits), logits


def test_for_nvec_validates():
    with pytest.raises(ValueError):
        HierarchicalMultiCategorical.for_nvec([2, 1024, 9, 9, 9])  # legacy 5-dim
    with pytest.raises(ValueError):
        HierarchicalMultiCategorical.for_nvec([6, 1024, 9, 9, 9, 9])  # verbs must be 3, 4 or 5
    HierarchicalMultiCategorical.for_nvec([4, 1024, 9, 9, 9, 9])      # double_click verb: valid
    HierarchicalMultiCategorical.for_nvec([5, 1024, 9, 9, 9, 9])      # xs_zoom verb: valid


def test_zoom_head_is_credited_by_both_zoom_verbs():
    """Verbs 2 (3D) and 4 (2D) share the zoom bin head.

    If only verb 2 credited it, an xs_zoom step would train an ungated head --
    the same bug the cell head had before double_click was added to it.
    """
    torch = pytest.importorskip("torch")
    cls = HierarchicalMultiCategorical.for_nvec([5, 8, 3, 3, 3, 3])
    logits = torch.zeros(2, 5 + 8 + 3 * 3 + 3)
    d = cls.from_logits(logits)
    a3d = torch.zeros(2, 6, dtype=torch.long)
    a3d[:, 0] = 2                                  # 3D zoom
    a2d = torch.zeros(2, 6, dtype=torch.long)
    a2d[:, 0] = 4                                  # 2D zoom
    a2d[:, 5] = a3d[:, 5] = 1                      # same zoom bin
    # Same bin, same head: the only logp difference is the verb's own term,
    # which is equal here because the verb logits are uniform.
    assert torch.allclose(d.logp(a3d), d.logp(a2d))
    # And a rotate step must NOT pick up the zoom head.
    arot = torch.zeros(2, 6, dtype=torch.long)
    arot[:, 0] = 1
    arot[:, 5] = 1
    assert not torch.allclose(d.logp(arot), d.logp(a2d))


def test_sample_shape_and_ranges():
    dist, _ = _dist()
    s = dist.sample()
    assert s.shape == (4, 6)
    for i, n in enumerate(NVEC):
        assert int(s[:, i].min()) >= 0 and int(s[:, i].max()) < n


def test_logp_matches_manual_gating():
    torch.manual_seed(0)
    dist, logits = _dist()
    a = torch.tensor([[0, 5, 0, 0, 0, 0],   # click -> cell head only
                      [1, 0, 2, 3, 4, 0],   # rotate -> rot heads only
                      [2, 0, 0, 0, 0, 7],   # zoom -> zoom head only
                      [0, 1023, 8, 8, 8, 8]])  # click; rot/zoom bins ignored
    lp = dist.logp(a)

    split = torch.split(logits, NVEC, dim=-1)
    logsm = [torch.log_softmax(s, -1) for s in split]

    def head(i, row, idx):
        return logsm[i][row, idx].item()

    exp0 = head(0, 0, 0) + head(1, 0, 5)
    exp1 = head(0, 1, 1) + head(2, 1, 2) + head(3, 1, 3) + head(4, 1, 4)
    exp2 = head(0, 2, 2) + head(5, 2, 7)
    exp3 = head(0, 3, 0) + head(1, 3, 1023)
    assert np.allclose(lp.numpy(), [exp0, exp1, exp2, exp3], atol=1e-5)


def test_logp_click_independent_of_rotate_logits():
    torch.manual_seed(1)
    logits = torch.randn(1, DIM)
    a = torch.tensor([[0, 42, 0, 0, 0, 0]])
    lp1 = DistCls.from_logits(logits).logp(a)
    perturbed = logits.clone()
    perturbed[:, NVEC[0] + NVEC[1]:] += 5.0  # rotate + zoom logits
    lp2 = DistCls.from_logits(perturbed).logp(a)
    assert torch.allclose(lp1, lp2, atol=1e-6)


def test_entropy_gated_by_type_probs():
    # Point-mass on click: entropy ~= H(cell) (H(type) ~ 0, other heads gated out)
    logits = torch.zeros(1, DIM)
    logits[0, 0] = 100.0  # type=click certain
    dist = DistCls.from_logits(logits)
    h = dist.entropy()
    h_cell = np.log(NVEC[1])  # uniform cell head
    assert abs(float(h) - h_cell) < 1e-3


def test_kl_self_zero_and_gating():
    torch.manual_seed(2)
    dist, logits = _dist()
    other = DistCls.from_logits(logits.clone())
    assert torch.allclose(dist.kl(other), torch.zeros(4), atol=1e-5)

    # Perturb ONE rotate-x bin (a constant shift over a whole head is a no-op —
    # softmax is shift-invariant): gated KL must be positive but smaller than
    # the ungated component KL (weighted by p(rotate) < 1).
    perturbed = logits.clone()
    rot_x0 = NVEC[0] + NVEC[1]
    perturbed[:, rot_x0] += 2.0
    kl = dist.kl(DistCls.from_logits(perturbed))
    assert (kl >= -1e-6).all()
    assert (kl > 1e-5).all()

    p_rotate = torch.softmax(logits[:, : NVEC[0]], -1)[:, 1]
    lp_a = torch.log_softmax(logits[:, rot_x0: rot_x0 + 9], -1)
    lp_b = torch.log_softmax(perturbed[:, rot_x0: rot_x0 + 9], -1)
    kl_x = (lp_a.exp() * (lp_a - lp_b)).sum(-1)
    assert torch.allclose(kl, p_rotate * kl_x, atol=1e-5)


def test_to_deterministic_is_per_head_argmax():
    torch.manual_seed(3)
    _, logits = _dist()
    dist = DistCls.from_logits(logits)
    det = dist.to_deterministic()
    a = det.sample()
    split = torch.split(logits, NVEC, dim=-1)
    expected = torch.stack([s.argmax(-1) for s in split], dim=-1)
    assert torch.equal(a.long(), expected)
