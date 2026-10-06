"""Tests for src/evaluate.py (run with: python -m pytest tests -q)."""
import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.data import RAW_DIR, load_all
from src.evaluate import (compute_metrics, evaluate_protocol, make_pipeline, protocol_splits,
                          random_split_baseline, summarize, _fit_predict)

needs_data = pytest.mark.skipif(not (RAW_DIR / "batch1.dat").exists(), reason="raw data not downloaded")


@pytest.fixture(scope="module")
def data():
    return load_all()


# ---------------- protocol definitions ----------------
@pytest.mark.parametrize("protocol,n_splits", [("P1", 7), ("P2", 9), ("P3", 9)])
def test_protocols_are_chronological(protocol, n_splits):
    splits = protocol_splits(protocol)
    assert len(splits) == n_splits
    for train, test in splits:
        assert max(train) < test, "training on the future would leak"
        assert test not in train


def test_p1_matches_the_ppt():
    assert protocol_splits("P1") == [((1, 2, 3), k) for k in range(4, 11)]


def test_unknown_protocol_raises():
    with pytest.raises(ValueError):
        protocol_splits("P9")


# ---------------- metrics (hand-computed) ----------------
def test_metrics_hand_example():
    m = compute_metrics(np.array([1, 1, 2, 2]), np.array([1, 2, 2, 2]))
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["macro_precision"] == pytest.approx((1 + 2 / 3) / 2)
    assert m["macro_recall"] == pytest.approx((0.5 + 1) / 2)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_macro_ignores_classes_absent_from_truth():
    m = compute_metrics(np.array([1, 1, 2, 2]), np.array([1, 3, 2, 2]))  # class 3 predicted but absent
    assert m["n_classes_test"] == 2
    assert m["macro_f1"] == pytest.approx((2 / 3 + 1.0) / 2)


# ---------------- leakage safety ----------------
def test_scaler_is_fitted_on_training_data_only():
    rng = np.random.RandomState(0)
    X_tr, X_te = rng.normal(0, 1, (50, 4)), rng.normal(100, 1, (50, 4))
    y_tr = rng.randint(1, 3, 50)
    pipe = make_pipeline(KNeighborsClassifier(), "raw").fit(X_tr, y_tr)
    np.testing.assert_allclose(pipe.named_steps["scaler"].mean_, X_tr.mean(axis=0))
    assert abs(pipe.named_steps["scaler"].mean_.mean() - np.vstack([X_tr, X_te]).mean()) > 10


class _Spy(BaseEstimator, ClassifierMixin):
    seen = []

    def fit(self, X, y):
        _Spy.seen.append(len(X))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


@needs_data
def test_models_are_fitted_only_on_training_batches(data):
    X, y, b = data
    _Spy.seen = []
    df = evaluate_protocol(_Spy(), X, y, b, "P1")
    assert _Spy.seen == df["n_train"].tolist() == [int((b <= 3).sum())] * 7
    assert df["n_test"].tolist() == [int((b == k).sum()) for k in range(4, 11)]


def test_lda_adapts_to_fewer_classes():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.normal(c * 3, 1, (30, 6)) for c in range(3)])
    y = np.repeat([1, 2, 3], 30)
    pred = _fit_predict(KNeighborsClassifier(), "LDA", X, y, X)  # 3 classes -> at most 2 axes, must not crash
    assert (pred == y).mean() > 0.9


@needs_data
def test_pca_keeps_95_percent_variance(data):
    X, y, b = data
    pipe = make_pipeline(SVC(), "PCA").fit(X[b <= 3], y[b <= 3])
    dr = pipe.named_steps["dr"]
    assert dr.explained_variance_ratio_.sum() >= 0.95 and dr.n_components_ < 128


def test_pca_variance_option_is_forwarded():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 20)) * np.linspace(5, 0.1, 20)   # decaying variances
    y = rng.randint(1, 4, 200)
    n = {v: make_pipeline(KNeighborsClassifier(), "PCA", pca_variance=v).fit(X, y).named_steps["dr"].n_components_
         for v in (0.90, 0.99)}
    assert n[0.90] < n[0.99]
    _fit_predict(KNeighborsClassifier(), "PCA", X, y, X, pca_variance=0.90)  # must be accepted


# ---------------- regression against the earlier quick check ----------------
@needs_data
def test_default_svm_and_knn_reproduce_quick_check(data):
    X, y, b = data
    expected = {  # accuracy rounded to 2 decimals, measured before the framework existed
        "SVM": (SVC(), [0.88, 0.95, 0.70, 0.68, 0.55, 0.74, 0.43]),
        "kNN": (KNeighborsClassifier(), [0.73, 0.76, 0.63, 0.62, 0.45, 0.62, 0.50]),
    }
    for name, (est, acc) in expected.items():
        df = evaluate_protocol(est, X, y, b, "P1", name=name)
        np.testing.assert_allclose(df["accuracy"].to_numpy(), acc, atol=0.0051)


@needs_data
def test_random_split_overestimates_time_aware_performance(data):
    X, y, b = data
    rnd = random_split_baseline(SVC(), X, y, name="SVM")["macro_f1"].iloc[0]
    time_aware = summarize(evaluate_protocol(SVC(), X, y, b, "P1", name="SVM"))["mean_macro_f1"].iloc[0]
    assert rnd - time_aware > 0.15


@needs_data
def test_summarize_weights(data):
    X, y, b = data
    df = evaluate_protocol(KNeighborsClassifier(), X, y, b, "P1", name="kNN")
    s = summarize(df).iloc[0]
    assert s["n_test_batches"] == 7
    assert s["mean_accuracy"] == pytest.approx(df["accuracy"].mean())
    assert s["weighted_accuracy"] == pytest.approx((df["accuracy"] * df["n_test"]).sum() / df["n_test"].sum())
