"""Re-implementation (in the spirit of) the classifier ensemble of Vergara et al. (2012), the dataset paper.

Method as described in secondary literature (the original paper itself was not accessed): one SVM is trained on each
batch; the weight of every classifier is its performance on the MOST RECENT labelled batch; the ensemble predicts by a
weighted combination of the classifiers' votes.

Our time-aware version, for a test batch ``k`` (declared before running):
* experts: one default-setting SVM per batch ``1 .. k-2``; features standardised with a scaler fitted on those batches;
* weights: the accuracy of each expert on batch ``k-1`` (the most recent labelled batch; ``equal`` weights as an ablation);
* prediction: weighted vote, ties broken towards the smaller gas label;
* ``k = 2`` has only batch 1, so a single expert trained on it is used (no weighting possible).
Test labels are never used; the test batch is only predicted.
"""
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

GASES = np.arange(1, 7)


def weighted_vote(predictions, weights):
    """Weighted majority vote. ``predictions``: (n_experts, n_samples) labels 1-6; ties go to the smaller label."""
    P = np.asarray(predictions); w = np.asarray(weights, dtype=float)
    scores = np.stack([((P == g) * w[:, None]).sum(axis=0) for g in GASES], axis=1)
    return GASES[np.argmax(scores, axis=1)]


def ensemble_predict(estimator, X, y, batch, test_batch, weighting="accuracy"):
    """Predict ``test_batch`` with the per-batch expert ensemble; ``weighting`` is ``"accuracy"`` or ``"equal"``."""
    if weighting not in ("accuracy", "equal"):
        raise ValueError("weighting must be 'accuracy' or 'equal'")
    te = batch == test_batch
    if test_batch <= 2:                                          # only batch 1 exists: a single expert, nothing to weight
        sc = StandardScaler().fit(X[batch == 1]); m = clone(estimator).fit(sc.transform(X[batch == 1]), y[batch == 1])
        return m.predict(sc.transform(X[te]))
    train_batches = list(range(1, test_batch - 1))                # 1 .. k-2 ; batch k-1 is kept for the weights
    recent = batch == test_batch - 1
    sc = StandardScaler().fit(X[np.isin(batch, train_batches)])
    experts = [clone(estimator).fit(sc.transform(X[batch == b]), y[batch == b]) for b in train_batches]
    if weighting == "accuracy":
        w = np.array([(m.predict(sc.transform(X[recent])) == y[recent]).mean() for m in experts])
        w = w if w.sum() > 0 else np.ones(len(experts))
    else:
        w = np.ones(len(experts))
    return weighted_vote([m.predict(sc.transform(X[te])) for m in experts], w)
