"""Tests for src/stacking.py (run with: python -m pytest tests -q)."""
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.data import RAW_DIR, load_all
from src.evaluate import evaluate_protocol
from src.stacking import BatchStackingClassifier

needs_data = pytest.mark.skipif(not (RAW_DIR / "batch1.dat").exists(), reason="raw data not downloaded")


class _Spy(BaseEstimator, ClassifierMixin):
    """Base model that records the size of every training set it is fitted on."""
    sizes = []

    def fit(self, X, y):
        _Spy.sizes.append(len(X))
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        p = np.full((len(X), len(self.classes_)), 1.0 / len(self.classes_))
        return p


def _toy(sizes=(30, 40, 50), seed=0):
    rng = np.random.RandomState(seed)
    g = np.concatenate([np.full(n, i + 1) for i, n in enumerate(sizes)])
    y = rng.randint(1, 4, len(g))
    X = rng.normal(size=(len(g), 5)) + y[:, None]
    return X, y, g


def test_batch_folds_leave_one_batch_out():
    X, y, g = _toy()
    _Spy.sizes = []
    BatchStackingClassifier([("spy", _Spy())], cv_strategy="batch").fit(X, y, groups=g)
    # three folds train on the other two batches (90, 80, 70 samples), then one refit on all 120
    assert sorted(_Spy.sizes) == [70, 80, 90, 120]


def test_single_batch_falls_back_to_stratified_folds():
    X, y, _ = _toy(sizes=(100,))
    _Spy.sizes = []
    st = BatchStackingClassifier([("spy", _Spy())], cv_strategy="batch", n_splits=5).fit(X, y, groups=np.ones(100))
    assert st.cv_strategy_used_ == "random" and len(_Spy.sizes) == 6   # 5 folds + final refit


def test_random_strategy_ignores_groups():
    X, y, g = _toy()
    st = BatchStackingClassifier([("spy", _Spy())], cv_strategy="random").fit(X, y, groups=g)
    assert st.cv_strategy_used_ == "random" and st.n_folds_ == 5


def test_stacking_learns_an_easy_problem():
    X, y, g = _toy(sizes=(80, 80, 80), seed=1)
    st = BatchStackingClassifier([("knn", KNeighborsClassifier()), ("lr", LogisticRegression(max_iter=500))]).fit(X, y, groups=g)
    assert (st.predict(X) == y).mean() > 0.6
    assert st.predict_proba(X).shape == (len(y), 3)
    assert sum(st.base_weights().values()) == pytest.approx(1.0)


def test_class_missing_from_a_training_fold_does_not_crash():
    rng = np.random.RandomState(0)
    g = np.repeat([1, 2, 3], 40)
    y = np.where(g == 3, 3, rng.randint(1, 3, 120))        # class 3 exists only in batch 3
    X = rng.normal(size=(120, 4)) + y[:, None]
    st = BatchStackingClassifier([("knn", KNeighborsClassifier())]).fit(X, y, groups=g)
    assert st.predict(X).shape == (120,)


@needs_data
def test_batch_ids_reach_the_stacker_through_the_pipeline():
    X, y, b = load_all()
    _Spy.sizes = []
    stack = BatchStackingClassifier([("spy", _Spy())], cv_strategy="batch")
    evaluate_protocol(stack, X, y, b, "P2", "raw")          # P2 has a single training batch -> stratified fallback
    assert len(_Spy.sizes) == 9 * 6                         # 9 test batches x (5 folds + 1 refit)
    n_b1 = int((b == 1).sum())
    assert _Spy.sizes.count(n_b1) == 9                      # final refit on B1, once per test batch
    _Spy.sizes = []
    evaluate_protocol(BatchStackingClassifier([("spy", _Spy())], cv_strategy="batch"), X, y, b, "P1", "raw")
    n1, n2, n3 = (int((b == k).sum()) for k in (1, 2, 3))
    per_split = sorted([n1 + n2 + n3, n2 + n3, n1 + n3, n1 + n2])   # final refit + 3 leave-one-batch-out folds
    assert sorted(_Spy.sizes[:4]) == per_split
    assert len(_Spy.sizes) == 7 * 4
