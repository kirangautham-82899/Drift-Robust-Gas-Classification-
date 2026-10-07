"""Worker functions that collect PER-SAMPLE predictions (needed for paired bootstrap and McNemar tests).

All functions are top-level so that joblib can run them in parallel. Predictions are returned as int8 vectors in
dataset order: ``P1`` vectors cover the samples of batches 4-10, ``P3`` vectors cover batches 2-10.
"""
import numpy as np

from src.evaluate import _fit_pipeline
from src.mitigation import METHODS, predict_mitigated


def fixed_train_predictions(key, estimator, variant, X, y, batch, cols=None):
    """Fit once on batches 1-3 (optionally on a feature subset) and predict batches 4-10."""
    tr, te = batch <= 3, batch >= 4
    Xs = X if cols is None else X[:, [int(c) for c in cols]]
    pipe = _fit_pipeline(estimator, variant, Xs[tr], y[tr], groups_tr=batch[tr])
    return key, pipe.predict(Xs[te]).astype(np.int8)


def mitigated_p1_predictions(key, estimator, variant, method, X, y, batch):
    """Per-batch mitigated predictions for test batches 4-10 (training = batches 1-3), concatenated in batch order."""
    parts = [predict_mitigated(estimator, variant, X, y, batch, (1, 2, 3), k, METHODS[method])[0] for k in range(4, 11)]
    return key, np.concatenate(parts).astype(np.int8)


def rolling_prediction(key, estimator, variant, method, X, y, batch, k):
    """One rolling-retraining split: train on batches 1..k-1 (then the method), predict batch k."""
    pred, _ = predict_mitigated(estimator, variant, X, y, batch, tuple(range(1, k)), k, METHODS[method])
    return key, k, pred.astype(np.int8)


def rolling_stack_prediction(key, estimator, variant, X, y, batch, k):
    """Rolling split for estimators that need the batch ids of the training samples (stacking): train on batches 1..k-1."""
    tr, te = batch < k, batch == k
    pipe = _fit_pipeline(estimator, variant, X[tr], y[tr], groups_tr=batch[tr])
    return key, k, pipe.predict(X[te]).astype(np.int8)
