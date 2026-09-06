"""Statistics in scripts/eval_compare.py.

These guard the numbers we make decisions on: a silently wrong p-value is
worse than no test, because it looks like evidence. The sign-flip null is
checked against cases with a known closed form (McNemar's binomial tail) and
against a policy compared with itself.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "eval_compare", Path(__file__).resolve().parents[1] / "scripts" / "eval_compare.py")
ec = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ec)


def write(tmp_path, name, per_pair):
    p = tmp_path / name
    p.write_text(json.dumps({"summary": {}, "per_pair": per_pair}))
    return str(p)


def rows(successes, reps=1):
    """successes[pair] -> count of successful repeats out of `reps`."""
    out = []
    for i, s in enumerate(successes):
        for r in range(reps):
            out.append({"pair_idx": i, "rep": r, "terminated": r < s})
    return out


def test_load_groups_repeats_by_pair(tmp_path):
    f = write(tmp_path, "a.json", rows([0, 1, 2], reps=2))
    got = ec.load(f)
    assert got == {0: [False, False], 1: [True, False], 2: [True, True]}


def test_load_defaults_missing_rep_to_zero(tmp_path):
    """Tier-1 files predate --repeats and carry no 'rep' key."""
    f = write(tmp_path, "a.json", [{"pair_idx": 3, "terminated": True}])
    assert ec.load(f) == {3: [True]}


def test_mcnemar_matches_binomial_tail():
    a = [True] * 10 + [False] * 5 + [True] * 20
    b = [False] * 10 + [True] * 5 + [True] * 20
    b10, b01, p = ec.mcnemar(a, b)
    assert (b10, b01) == (10, 5)
    n = 15
    expect = 2.0 * sum(math.comb(n, i) for i in range(6)) / 2.0 ** n
    assert p == pytest.approx(expect)


def test_mcnemar_no_discordance_is_p_one():
    assert ec.mcnemar([True, False], [True, False]) == (0, 0, 1.0)


def test_signflip_exact_matches_sign_test_for_equal_magnitudes():
    """With every |d| equal, the sign-flip null on the mean IS the sign test,
    so the exact branch must reproduce the binomial two-sided tail."""
    d = np.array([0.2] * 7 + [-0.2] * 1)
    p = ec.signflip_p(d, np.random.default_rng(0), 1000)
    expect = 2.0 * sum(math.comb(8, i) for i in range(2)) / 2.0 ** 8
    assert p == pytest.approx(expect, rel=0.02)


def test_signflip_symmetric_differences_are_not_significant():
    d = np.array([0.2, -0.2, 0.4, -0.4, 0.6, -0.6])
    assert ec.signflip_p(d, np.random.default_rng(0), 1000) > 0.5


def test_signflip_monte_carlo_branch_agrees_with_exact_scale():
    """n>18 takes the sampling branch; a lopsided effect must still be small-p."""
    d = np.concatenate([np.full(25, 0.2), np.full(3, -0.2)])
    assert ec.signflip_p(d, np.random.default_rng(0), 20000) < 0.01


def test_signed_ranks_average_ties():
    r = ec.signed_ranks(np.array([0.5, -0.5, 1.0]))
    assert r[0] == pytest.approx(1.5)
    assert r[1] == pytest.approx(-1.5)
    assert r[2] == pytest.approx(3.0)


def test_noise_floor_zero_for_deterministic_policy():
    var, disc = ec.noise_floor([[True] * 4, [False] * 4], k=4)
    assert var == pytest.approx(0.0)
    assert disc == pytest.approx(0.0)


def test_noise_floor_is_maximal_for_coin_flip_policy():
    """x=k/2 maximises the unbiased estimate of p(1-p), which is 0.25 at p=0.5
    -- so a pure coin flip would flip ~50% of pairs against a rerun of itself."""
    var, disc = ec.noise_floor([[True, True, False, False]] * 8, k=4)
    assert var == pytest.approx(1.0 / 3.0)
    assert disc == pytest.approx(2.0 / 3.0)


def test_compare_self_reports_no_difference(tmp_path, capsys):
    rng = np.random.default_rng(3)
    succ = rng.integers(0, 6, size=40).tolist()
    f = write(tmp_path, "a.json", rows(succ, reps=5))
    g = write(tmp_path, "b.json", rows(succ, reps=5))
    import sys
    argv = sys.argv
    sys.argv = ["eval_compare", f, g, "--labels", "x", "y"]
    try:
        assert ec.main() == 0
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "difference          : +0.00 pp" in out
    assert "comparing at k=5" in out


def test_compare_tier1_files_stop_before_tier2(tmp_path, capsys):
    f = write(tmp_path, "a.json", rows([1, 0, 1, 1], reps=1))
    g = write(tmp_path, "b.json", rows([0, 0, 1, 1], reps=1))
    import sys
    argv = sys.argv
    sys.argv = ["eval_compare", f, g]
    try:
        assert ec.main() == 0
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "Tier 1: McNemar exact" in out
    assert "Tier 2 needs repeats>1" in out


def test_compare_uses_only_shared_pairs(tmp_path, capsys):
    f = write(tmp_path, "a.json", rows([1, 1, 1], reps=1))
    g = write(tmp_path, "b.json", rows([1, 1], reps=1))
    import sys
    argv = sys.argv
    sys.argv = ["eval_compare", f, g]
    try:
        assert ec.main() == 0
    finally:
        sys.argv = argv
    assert "pairs compared      : 2" in capsys.readouterr().out
