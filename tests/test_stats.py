"""Tests for src/stats.py (run with: python -m pytest tests -q)."""
import numpy as np
import pytest

from src.evaluate import compute_metrics
from src.stats import (bootstrap_f1, compare, fast_macro_f1, friedman_nemenyi, holm, mcnemar_exact, percentile_ci,
                       per_batch_f1, two_level_means, verdict)


def test_fast_macro_f1_equals_the_reference_implementation():
    rng = np.random.RandomState(0)
    for _ in range(200):
        n = rng.randint(5, 300); y = rng.randint(1, 7, n); p = np.where(rng.rand(n) < 0.6, y, rng.randint(1, 7, n))
        assert fast_macro_f1(y, p) == pytest.approx(compute_metrics(y, p)["macro_f1"], abs=1e-12)


def test_fast_macro_f1_ignores_classes_absent_from_the_truth():
    y = np.array([1, 1, 2, 2]); p = np.array([1, 3, 2, 2])
    assert fast_macro_f1(y, p) == pytest.approx((2 / 3 + 1.0) / 2)


def _toy(seed=0, n_per=300, batches=(4, 5, 6)):
    rng = np.random.RandomState(seed)
    batch = np.repeat(batches, n_per); y = rng.randint(1, 7, len(batch))
    good = np.where(rng.rand(len(y)) < 0.9, y, rng.randint(1, 7, len(y)))
    bad = np.where(rng.rand(len(y)) < 0.4, y, rng.randint(1, 7, len(y)))
    return y, batch, good, bad


def test_bootstrap_is_paired_and_reproducible():
    y, batch, good, bad = _toy()
    F1, _ = bootstrap_f1(y, {"a": good, "b": good}, batch, [4, 5, 6], n_boot=50, seed=3)
    np.testing.assert_array_equal(F1[:, :, 0], F1[:, :, 1])                       # identical predictions -> identical scores
    F2, _ = bootstrap_f1(y, {"a": good, "b": good}, batch, [4, 5, 6], n_boot=50, seed=3)
    np.testing.assert_array_equal(F1, F2)


def test_confidence_interval_covers_the_observed_value_and_ranks_models_correctly():
    y, batch, good, bad = _toy(seed=1)
    F, names = bootstrap_f1(y, {"good": good, "bad": bad}, batch, [4, 5, 6], n_boot=300, seed=1)
    pt = {n: per_batch_f1(y, p, batch, [4, 5, 6]) for n, p in {"good": good, "bad": bad}.items()}
    lo, hi = percentile_ci(F.mean(axis=1)[:, 0]); assert lo <= pt["good"].mean() <= hi
    c = compare(F, names, "good", "bad", pt)
    assert c["diff"] > 0.3 and c["ci_two_level"][0] > 0 and c["verdict"] == "clear" and c["batches_A_better"] == 3


def test_two_level_interval_is_wider_than_the_conditional_one_when_batches_differ():
    rng = np.random.RandomState(2); batches = [4, 5, 6, 7, 8]
    batch = np.repeat(batches, 400); y = rng.randint(1, 7, len(batch))
    acc = np.repeat([0.95, 0.8, 0.6, 0.4, 0.2], 400)                                # accuracy differs a lot between batches
    p = np.where(rng.rand(len(y)) < acc, y, rng.randint(1, 7, len(y)))
    F, _ = bootstrap_f1(y, {"m": p}, batch, batches, n_boot=400, seed=2)
    c = percentile_ci(F.mean(axis=1)[:, 0]); t = percentile_ci(two_level_means(F)[:, 0])
    assert (t[1] - t[0]) > 3 * (c[1] - c[0])


def test_verdict_rule():
    assert verdict((0.01, 0.05), (0.02, 0.04)) == "clear"
    assert verdict((-0.02, 0.05), (0.01, 0.04)) == "suggestive"
    assert verdict((-0.02, 0.05), (-0.01, 0.04)) == "inconclusive"
    assert verdict((-0.06, -0.01), (-0.05, -0.02)) == "clear"


def test_holm_adjustment():
    adj = holm([0.01, 0.04, 0.03]); np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])
    assert (holm([0.5, 0.5]) <= 1).all()


def test_mcnemar_exact_matches_the_binomial_formula():
    y = np.ones(20, int); a = np.ones(20, int); b = np.where(np.arange(20) < 10, 2, 1)       # A right 10 times where B is wrong, never reverse
    r = mcnemar_exact(y, a, b)
    assert r["A_correct_B_wrong"] == 10 and r["A_wrong_B_correct"] == 0 and r["p"] == pytest.approx(2 * 0.5 ** 10)
    assert mcnemar_exact(y, a, a)["p"] == 1.0


def test_friedman_nemenyi_critical_difference_matches_demsar_2006():
    rng = np.random.RandomState(0); M = rng.rand(10, 7)
    r = friedman_nemenyi(M, list("abcdefg"))
    assert r["cd"] == pytest.approx(2.949 * np.sqrt(7 * 8 / (6 * 10)), abs=0.01)           # q_0.05(k=7) = 2.949
    assert sorted(np.round(r["avg_rank"].sum(), 6).ravel()) == [28.0]                      # ranks of 7 models sum to 28 per block / blocks
    clear = np.column_stack([np.full(10, v) + rng.rand(10) * 0.01 for v in [0.9, 0.7, 0.5, 0.3]])
    s = friedman_nemenyi(clear, list("abcd")); assert s["p"] < 1e-3 and list(np.argsort(s["avg_rank"])) == [0, 1, 2, 3]
