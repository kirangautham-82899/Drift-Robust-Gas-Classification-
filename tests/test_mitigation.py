"""Tests for src/mitigation.py (run with: python -m pytest tests -q)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.data import RAW_DIR, load_all
from src.evaluate import compute_metrics, evaluate_protocol
from src.mitigation import (METHODS, P1_METHODS, batchwise_standardize, coral_align, predict_mitigated, run_grid,
                            supports)

needs_data = pytest.mark.skipif(not (RAW_DIR / "batch1.dat").exists(), reason="raw data not downloaded")


@pytest.fixture(scope="module")
def data():
    return load_all()


class _Spy(BaseEstimator, ClassifierMixin):
    """Records the size (and sample weights) of every training set it is fitted on."""
    log = []

    def fit(self, X, y, sample_weight=None):
        _Spy.log.append({"n": len(X), "w": None if sample_weight is None else np.asarray(sample_weight).copy()})
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


# ---------------- the maths ----------------
def test_batchwise_standardize_uses_each_batch_own_statistics():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(5, 2, (50, 3)), rng.normal(-3, 0.5, (40, 3))]); b = np.repeat([1, 2], [50, 40])
    Z = batchwise_standardize(X, b)
    for k in (1, 2):
        np.testing.assert_allclose(Z[b == k].mean(axis=0), 0, atol=1e-9); np.testing.assert_allclose(Z[b == k].std(axis=0), 1, atol=1e-9)
    X2 = X.copy(); X2[b == 2] += 100                                   # changing batch 2 must not change batch 1's result
    np.testing.assert_allclose(batchwise_standardize(X2, b)[b == 1], Z[b == 1])


def test_batchwise_standardize_handles_constant_features():
    X = np.column_stack([np.ones(10), np.arange(10.0)]); Z = batchwise_standardize(X, np.ones(10))
    assert np.isfinite(Z).all() and np.allclose(Z[:, 0], 0)


def test_coral_to_itself_is_the_identity():
    Z = np.random.RandomState(1).normal(size=(200, 6))
    np.testing.assert_allclose(coral_align(Z, Z, reg=1.0), Z, atol=1e-8)


def test_coral_makes_the_source_covariance_match_the_target():
    rng = np.random.RandomState(2); d = 5
    A, B = rng.normal(size=(d, d)), rng.normal(size=(d, d))
    Zs, Zt = rng.normal(size=(20000, d)) @ A, rng.normal(size=(20000, d)) @ B
    out = coral_align(Zs, Zt, reg=1e-9)
    np.testing.assert_allclose(np.cov(out, rowvar=False), np.cov(Zt, rowvar=False), rtol=0.05, atol=0.05)


# ---------------- the method logic ----------------
def test_recency_needs_sample_weight_support():
    assert not supports(KNeighborsClassifier(), METHODS["recency-h2"])      # kNN.fit has no sample_weight
    assert supports(SVC(), METHODS["recency-h2"]) and supports(KNeighborsClassifier(), METHODS["window-2"])


@needs_data
def test_window_trains_only_on_the_last_batches(data):
    X, y, b = data
    _Spy.log = []
    predict_mitigated(_Spy(), "raw", X, y, b, [1, 2, 3, 4, 5], 6, METHODS["window-2"])
    assert _Spy.log[0]["n"] == int(((b == 4) | (b == 5)).sum())


@needs_data
def test_recency_weights_halve_with_every_half_life(data):
    X, y, b = data
    _Spy.log = []
    predict_mitigated(_Spy(), "raw", X, y, b, [1, 2, 3, 4, 5], 6, {"half_life": 1.0})
    w = _Spy.log[0]["w"]; train = np.isin(b, [1, 2, 3, 4, 5])
    for k, expected in [(5, 1.0), (4, 0.5), (3, 0.25), (2, 0.125), (1, 0.0625)]:
        np.testing.assert_allclose(w[b[train] == k], expected)


@needs_data
@pytest.mark.parametrize("method", ["baseline", "window-2", "batch-std", "coral"])
def test_test_labels_are_never_used(data, method):
    X, y, b = data
    y2 = y.copy(); te = b == 6; y2[te] = np.random.RandomState(0).randint(1, 7, te.sum())     # scramble ONLY the test labels
    p1, _ = predict_mitigated(SVC(), "raw", X, y, b, [1, 2, 3, 4, 5], 6, METHODS[method])
    p2, _ = predict_mitigated(SVC(), "raw", X, y2, b, [1, 2, 3, 4, 5], 6, METHODS[method])
    np.testing.assert_array_equal(p1, p2)


@needs_data
def test_never_trains_on_the_future(data):
    X, y, b = data
    with pytest.raises(AssertionError):
        predict_mitigated(SVC(), "raw", X, y, b, [1, 2, 6], 6, METHODS["baseline"])


@needs_data
def test_baseline_method_reproduces_the_standard_evaluation(data):
    X, y, b = data
    ref = evaluate_protocol(KNeighborsClassifier(), X, y, b, "P3", "raw", name="kNN").set_index("test_batch").macro_f1
    for k in (2, 3, 4, 5):
        pred, _ = predict_mitigated(KNeighborsClassifier(), "raw", X, y, b, list(range(1, k)), k, METHODS["baseline"])
        assert compute_metrics(y[b == k], pred)["macro_f1"] == pytest.approx(ref[k])


@needs_data
def test_run_grid_shape_and_columns(data):
    X, y, b = data
    out = run_grid({"kNN": KNeighborsClassifier()}, X, y, b, variants=("raw",), protocols=("P1",), n_jobs=2)
    assert len(out) == len(P1_METHODS) * 7 and set(out.method) == set(P1_METHODS) and set(out.test_batch) == set(range(4, 11))
    assert {"macro_f1", "macro_f1_gases1to5", "toluene_recall", "toluene_in_train"} <= set(out.columns)
