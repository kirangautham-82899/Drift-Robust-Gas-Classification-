"""Leakage-safe stacking ensemble for drifting data.

How stacking works
------------------
1. Several *base models* (e.g. SVM, kNN, ...) each predict gas probabilities / scores.
2. A *meta-learner* (logistic regression) learns how to combine those predictions.

The meta-learner must be trained on predictions the base models made for samples they
had **not** been trained on ("out-of-fold" predictions); otherwise it would learn to trust
models that merely memorised the training data.

Why batch folds
---------------
The usual way to get out-of-fold predictions is random K-fold, but samples from the same
batch are near-identical "twins" (see notebook 01), so a random fold leaks: the base model
has seen a twin of every validation sample, looks far better than it will on future data,
and the meta-learner over-trusts it. Here the folds are **leave-one-batch-out** inside the
training data (train on the other training batches, predict the held-out batch). If only
one training batch exists (protocol P2) that is impossible and we fall back to stratified
5-fold, which is reported in ``cv_strategy_used_``.
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold


def _scores(est, X):
    """Probabilities if available, otherwise decision-function scores (e.g. SVC)."""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X), True
    out = est.decision_function(X)
    if out.ndim == 1:                       # binary case
        out = np.column_stack([-out, out])
    return out, False


def _aligned_scores(est, X, classes):
    """Scores of a fitted model as an (n, len(classes)) matrix, even if a class was unseen in training."""
    out, is_proba = _scores(est, X)
    if out.shape[1] == len(classes):
        return out
    fill = np.zeros((len(X), len(classes))) if is_proba else (out.min(axis=1, keepdims=True) - 1.0) * np.ones((len(X), len(classes)))
    fill[:, np.searchsorted(classes, est.classes_)] = out
    return fill


def _fold_scores(est, X, y, tr, va, classes):
    model = clone(est).fit(X[tr], y[tr])
    return va, _aligned_scores(model, X[va], classes)


class BatchStackingClassifier(ClassifierMixin, BaseEstimator):
    """Stacking with batch-aware (or, for the ablation, random) out-of-fold predictions.

    Parameters
    ----------
    estimators : list of ``(name, estimator)`` base models
    final_estimator : meta-learner (default ``LogisticRegression(max_iter=2000)``)
    cv_strategy : ``"batch"`` (leave-one-batch-out, needs ``groups``), ``"forward"`` (forward chaining,
        needs ``groups``) or ``"random"`` (stratified K-fold - the leaky ablation)
    """

    def __init__(self, estimators, final_estimator=None, cv_strategy="batch", n_splits=5, random_state=42, n_jobs=1):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _splits(self, X, y, groups):
        if self.cv_strategy not in ("batch", "random", "forward"):
            raise ValueError("cv_strategy must be 'batch', 'random' or 'forward'")
        if self.cv_strategy == "batch" and groups is not None and len(np.unique(groups)) >= 2:
            self.cv_strategy_used_ = "batch"
            return list(LeaveOneGroupOut().split(X, y, groups))
        if self.cv_strategy == "forward" and groups is not None and len(np.unique(groups)) >= 2:
            # forward chaining: predict batch j with experts trained only on the batches BEFORE j (never the future);
            # the oldest batch has no out-of-fold predictions and is not used to train the meta-learner
            self.cv_strategy_used_ = "forward"
            order = np.sort(np.unique(groups)); groups = np.asarray(groups)
            return [(np.where(np.isin(groups, order[:j]))[0], np.where(groups == order[j])[0]) for j in range(1, len(order))]
        self.cv_strategy_used_ = "random"
        return list(StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state).split(X, y))

    def fit(self, X, y, groups=None):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        splits = self._splits(X, y, groups)
        jobs = [(i, tr, va) for i, (_, est) in enumerate(self.estimators) for tr, va in splits]
        out = Parallel(n_jobs=self.n_jobs)(
            delayed(_fold_scores)(self.estimators[i][1], X, y, tr, va, self.classes_) for i, tr, va in jobs
        )
        meta = np.zeros((len(y), len(self.estimators) * len(self.classes_)))
        k = len(self.classes_)
        covered = np.zeros(len(y), dtype=bool)
        for (i, _, _), (va, s) in zip(jobs, out):
            meta[va, i * k:(i + 1) * k] = s
            covered[va] = True                       # every row for the partition strategies; not the oldest batch for 'forward'
        self.base_models_ = [(n, clone(e).fit(X, y)) for n, e in self.estimators]    # refit on all training data
        final = self.final_estimator if self.final_estimator is not None else LogisticRegression(max_iter=2000)
        self.final_estimator_ = clone(final).fit(meta[covered], y[covered])
        self.n_folds_ = len(splits)
        self.meta_rows_ = int(covered.sum())
        return self

    def _meta(self, X):
        X = np.asarray(X)
        return np.hstack([_aligned_scores(m, X, self.classes_) for _, m in self.base_models_])

    def predict(self, X):
        return self.final_estimator_.predict(self._meta(X))

    def predict_proba(self, X):
        return self.final_estimator_.predict_proba(self._meta(X))

    def base_weights(self):
        """Share of the meta-learner's total |coefficient| given to each base model (sums to 1)."""
        k = len(self.classes_)
        w = np.abs(self.final_estimator_.coef_)
        per_model = np.array([w[:, i * k:(i + 1) * k].sum() for i in range(len(self.base_models_))])
        return dict(zip([n for n, _ in self.base_models_], per_model / per_model.sum()))


BASE_NAMES = ["SVM", "kNN", "Random Forest", "Gradient Boosting", "Naive Bayes"]


def make_stackers():
    """The two preliminary stackers: leakage-safe batch folds and the leaky random-fold ablation."""
    from src.models import default_models   # local import keeps this module free of a models dependency
    def bases():
        dm = default_models()
        return [(n, dm[n]) for n in BASE_NAMES]
    return {
        "Stacking (batch CV)": BatchStackingClassifier(bases(), cv_strategy="batch"),
        "Stacking (random CV)": BatchStackingClassifier(bases(), cv_strategy="random"),
    }


BASE_NAMES_FAST = ["SVM", "kNN", "Random Forest", "Naive Bayes"]     # declared for the redesigned stacker: diverse and fast


def make_fast_stackers():
    """Four-expert stackers that differ ONLY in how the out-of-fold predictions are produced."""
    from src.models import default_models
    def bases():
        dm = default_models()
        return [(n, dm[n]) for n in BASE_NAMES_FAST]
    return {
        "Stack4 (forward)": BatchStackingClassifier(bases(), cv_strategy="forward"),
        "Stack4 (batch CV)": BatchStackingClassifier(bases(), cv_strategy="batch"),
        "Stack4 (random CV)": BatchStackingClassifier(bases(), cv_strategy="random"),
    }
