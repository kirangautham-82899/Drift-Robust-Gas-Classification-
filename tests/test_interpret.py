"""Tests for src/interpret.py (run with: python -m pytest tests -q)."""
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src.data import RAW_DIR, load_all
from src.interpret import (class_conditional_drift, discriminability, feature_groups, feature_scores,
                           permutation_importance_groups, select_features)

needs_data = pytest.mark.skipif(not (RAW_DIR / "batch1.dat").exists(), reason="raw data not downloaded")


def test_every_grouping_partitions_all_128_features():
    for kind, groups in feature_groups().items():
        cols = sorted(c for g in groups.values() for c in g)
        assert cols == list(range(128)), kind
    g = feature_groups()
    assert len(g["sensor"]) == 16 and len(g["type"]) == 8 and len(g["family"]) == 3
    assert all(len(v) == 8 for v in g["sensor"].values()) and all(len(v) == 16 for v in g["type"].values())


class _Threshold(BaseEstimator, ClassifierMixin):
    """Predicts class 1/2 from column 0 only."""
    def predict(self, X):
        return np.where(X[:, 0] > 0, 2, 1)


def test_permutation_importance_finds_the_feature_the_model_uses():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(400, 6)); y = np.where(X[:, 0] > 0, 2, 1)
    imp = permutation_importance_groups(_Threshold(), X, y, {"used": [0], "ignored": [1, 2], "also ignored": [3, 4, 5]}, n_repeats=5)
    d = imp.set_index("group").importance_mean
    assert d["used"] > 0.3 and abs(d["ignored"]) < 1e-12 and abs(d["also ignored"]) < 1e-12


def test_permutation_shuffles_a_group_with_one_shared_permutation():
    seen = []

    class _Spy(_Threshold):
        def predict(self, X):
            seen.append(X.copy()); return super().predict(X)

    X = np.arange(40.0).reshape(20, 2); y = np.ones(20, int)
    permutation_importance_groups(_Spy(), X, y, {"both": [0, 1]}, n_repeats=1)
    shuffled = seen[1]                                         # seen[0] is the unshuffled baseline call
    order = shuffled[:, 0] / 2                                 # row index each sample came from
    np.testing.assert_allclose(shuffled[:, 1], order * 2 + 1)  # column 1 moved together with column 0


def test_drift_metric_is_larger_for_a_shifted_feature():
    rng = np.random.RandomState(1)
    n = 120; y = np.tile(np.repeat([1, 2], n // 2), 3); batch = np.repeat([1, 2, 3], n)
    X = rng.normal(size=(len(y), 2)) + y[:, None] * 3
    X[batch == 2, 0] += 4; X[batch == 3, 0] += 8          # feature 0 drifts, feature 1 is stable
    d = class_conditional_drift(X, y, batch)
    assert d[0] > 3 * d[1] and d[1] < 0.5


def test_drift_needs_enough_samples():
    X = np.zeros((6, 2)); y = np.array([1, 1, 1, 2, 2, 2]); b = np.array([1, 1, 1, 2, 2, 2])
    with pytest.raises(ValueError):
        class_conditional_drift(X, y, b, later=(2,), min_n=10)


def test_discriminability_ranks_informative_features_first():
    rng = np.random.RandomState(2); y = rng.randint(1, 4, 300)
    X = np.column_stack([y + rng.normal(0, 0.3, 300), rng.normal(size=300)])
    F = discriminability(X, y)
    assert F[0] > 50 * F[1]


@needs_data
def test_selection_rules_are_deterministic_sized_and_ordered():
    X, y, b = load_all(); s = feature_scores(X, y, b)
    assert len(select_features(s, "all")) == 128
    for rule in ["stable", "discriminative", "balanced"]:
        a, c = select_features(s, rule, 0.5), select_features(s, rule, 0.5)
        assert len(a) == 64 and np.array_equal(a, c) and list(a) == sorted(a)
    stable = select_features(s, "stable", 0.5)
    assert s.drift.to_numpy()[stable].max() <= np.sort(s.drift.to_numpy())[63] + 1e-12     # exactly the 64 lowest-drift features
    r1, r2 = select_features(s, "random", 0.5, seed=1), select_features(s, "random", 0.5, seed=2)
    assert len(r1) == 64 and not np.array_equal(r1, r2)


@needs_data
def test_feature_scores_use_only_the_training_batches():
    X, y, b = load_all(); s1 = feature_scores(X, y, b)
    X2 = X.copy(); X2[b > 3] += 1000.0; y2 = y.copy(); y2[b > 3] = 1       # wreck every later batch
    s2 = feature_scores(X2, y2, b)
    np.testing.assert_allclose(s1[["F", "drift"]].to_numpy(), s2[["F", "drift"]].to_numpy())
