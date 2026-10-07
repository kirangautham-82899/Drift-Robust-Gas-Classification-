"""Tests for src/ensemble.py (run with: python -m pytest tests -q)."""
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier

from src.data import RAW_DIR, load_all
from src.ensemble import ensemble_predict, weighted_vote

needs_data = pytest.mark.skipif(not (RAW_DIR / "batch1.dat").exists(), reason="raw data not downloaded")


def test_weighted_vote_hand_examples():
    P = np.array([[1, 2, 3], [1, 3, 3], [2, 2, 3]])
    np.testing.assert_array_equal(weighted_vote(P, [1, 1, 1]), [1, 2, 3])                  # plain majority
    np.testing.assert_array_equal(weighted_vote(P, [1, 1, 5]), [2, 2, 3])                  # a heavy expert overrules the others
    np.testing.assert_array_equal(weighted_vote(np.array([[1], [2]]), [1, 1]), [1])         # ties go to the smaller label


class _Log(BaseEstimator, ClassifierMixin):
    """Records the training-set size of every fit and always predicts the majority class it saw."""
    sizes = []

    def fit(self, X, y):
        _Log.sizes.append(len(X)); self.c_ = np.bincount(y).argmax(); return self

    def predict(self, X):
        return np.full(len(X), self.c_)


@needs_data
def test_experts_use_batches_up_to_k_minus_2_and_weights_use_batch_k_minus_1_only():
    X, y, b = load_all()
    _Log.sizes = []
    ensemble_predict(_Log(), X, y, b, 6)
    assert _Log.sizes == [int((b == k).sum()) for k in (1, 2, 3, 4)]        # one expert per batch 1..4, none on batches 5 or 6
    _Log.sizes = []
    ensemble_predict(_Log(), X, y, b, 2)
    assert _Log.sizes == [int((b == 1).sum())]                              # k = 2: a single expert on batch 1


@needs_data
@pytest.mark.parametrize("weighting", ["accuracy", "equal"])
def test_test_labels_are_never_used(weighting):
    X, y, b = load_all()
    y2 = y.copy(); te = b == 6; y2[te] = np.random.RandomState(0).randint(1, 7, te.sum())
    p1 = ensemble_predict(KNeighborsClassifier(), X, y, b, 6, weighting); p2 = ensemble_predict(KNeighborsClassifier(), X, y2, b, 6, weighting)
    np.testing.assert_array_equal(p1, p2)


@needs_data
def test_equal_weights_ignore_labels_of_the_most_recent_batch():
    X, y, b = load_all()
    y2 = y.copy(); y2[b == 5] = np.random.RandomState(1).randint(1, 7, int((b == 5).sum()))   # scramble batch 5 = k-1 labels for k = 6
    e1 = ensemble_predict(KNeighborsClassifier(), X, y, b, 6, "equal"); e2 = ensemble_predict(KNeighborsClassifier(), X, y2, b, 6, "equal")
    np.testing.assert_array_equal(e1, e2)                                    # equal weights ignore every label outside the experts' own batches


def test_unknown_weighting_raises():
    with pytest.raises(ValueError):
        ensemble_predict(KNeighborsClassifier(), np.zeros((3, 2)), np.ones(3, int), np.ones(3, int), 3, "magic")
