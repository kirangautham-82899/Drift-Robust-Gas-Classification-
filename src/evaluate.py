"""Time-aware, leakage-safe evaluation for the drift project.

Why this module exists
----------------------
The 10 batches are chronological. A random train/test split puts near-identical
"twin" samples in both sets and hides drift (see notebook 01). Instead we train on
*earlier* batches and test on *later* ones, one test batch at a time, so the score
of each test batch shows how performance decays as the data gets older/newer.

Protocols
---------
``P1``  train on batches 1-3          -> test on each of 4..10   (the PPT protocol)
``P2``  train on batch 1              -> test on each of 2..10   (Vergara et al., setting 1)
``P3``  train on batches 1..k-1       -> test on batch k, k=2..10 (rolling / Vergara setting 2)

Leakage safety
--------------
Every model is wrapped in a scikit-learn ``Pipeline`` (scaler -> optional PCA/LDA ->
classifier) that is cloned and fitted on the training batches only. The scaler, PCA
and LDA therefore never see test data.

Metric note
-----------
Macro precision/recall/F1 are averaged over the gases that are actually **present in the
test batch** (some batches have no Toluene). Averaging over absent classes would only add
meaningless zeros. Wrong predictions of an absent class still lower the scores of the
classes that are present.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

GAS_LABELS = [1, 2, 3, 4, 5, 6]
VARIANTS = ("raw", "PCA", "LDA")
PROTOCOL_DESCRIPTIONS = {
    "P1": "train B1-3 -> test each of B4..B10",
    "P2": "train B1 -> test each of B2..B10",
    "P3": "train B1..k-1 -> test B_k (k = 2..10, rolling)",
}
METRIC_COLUMNS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]


def protocol_splits(protocol):
    """Return the list of ``(train_batches, test_batch)`` pairs of a protocol."""
    if protocol == "P1":
        return [((1, 2, 3), k) for k in range(4, 11)]
    if protocol == "P2":
        return [((1,), k) for k in range(2, 11)]
    if protocol == "P3":
        return [(tuple(range(1, k)), k) for k in range(2, 11)]
    raise ValueError(f"unknown protocol {protocol!r}; choose from {sorted(PROTOCOL_DESCRIPTIONS)}")


def make_pipeline(estimator, variant="raw", pca_variance=0.95, lda_components=5, random_state=42):
    """Build ``StandardScaler -> [PCA | LDA] -> estimator``.

    ``variant``: ``"raw"`` (scaling only), ``"PCA"`` (keep ``pca_variance`` of the variance,
    a rule of thumb, not tuned) or ``"LDA"`` (up to ``lda_components`` discriminant axes;
    5 is the maximum for 6 classes).
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    steps = [("scaler", StandardScaler())]
    if variant == "PCA":
        steps.append(("dr", PCA(n_components=pca_variance, svd_solver="full", random_state=random_state)))
    elif variant == "LDA":
        steps.append(("dr", LinearDiscriminantAnalysis(n_components=lda_components)))
    steps.append(("clf", clone(estimator)))
    return Pipeline(steps)


def compute_metrics(y_true, y_pred):
    """Accuracy plus macro/weighted scores over the classes present in ``y_true``."""
    present = np.unique(y_true)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=present, average="macro", zero_division=0
    )
    _, _, wf1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=present, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f1,
        "weighted_f1": wf1,
        "n_classes_test": len(present),
    }


def _batch_range(batches):
    return f"{min(batches)}-{max(batches)}" if len(batches) > 1 else f"{batches[0]}"


def _fit_predict(estimator, variant, X_tr, y_tr, X_te, lda_components=5, random_state=42, pca_variance=0.95):
    pipe = make_pipeline(estimator, variant, pca_variance=pca_variance,
                         lda_components=lda_components, random_state=random_state)
    if variant == "LDA":  # LDA can give at most (n_classes_in_train - 1) axes
        pipe.set_params(dr__n_components=min(lda_components, len(np.unique(y_tr)) - 1))
    pipe.fit(X_tr, y_tr)
    return pipe.predict(X_te)


def evaluate_protocol(estimator, X, y, batch, protocol="P1", variant="raw", name=None,
                      return_confusion=False, **pipeline_kwargs):
    """Evaluate one model under a time-aware protocol.

    Returns a DataFrame with one row per test batch (``model, variant, protocol,
    train_batches, test_batch, n_train, n_test`` + metrics). With
    ``return_confusion=True`` also returns ``{test_batch: 6x6 confusion matrix}``
    (rows = true gas 1-6, columns = predicted gas 1-6).
    """
    name = name or type(estimator).__name__
    rows, cms = [], {}
    for train_batches, test_batch in protocol_splits(protocol):
        tr = np.isin(batch, train_batches)
        te = batch == test_batch
        # a time-aware split must never train on the future
        assert max(train_batches) < test_batch and not (tr & te).any()
        y_pred = _fit_predict(estimator, variant, X[tr], y[tr], X[te], **pipeline_kwargs)
        rows.append({
            "model": name, "variant": variant, "protocol": protocol,
            "train_batches": _batch_range(train_batches), "test_batch": test_batch,
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            **compute_metrics(y[te], y_pred),
        })
        if return_confusion:
            cms[test_batch] = confusion_matrix(y[te], y_pred, labels=GAS_LABELS)
    df = pd.DataFrame(rows)
    return (df, cms) if return_confusion else df


def random_split_baseline(estimator, X, y, variant="raw", name=None, test_size=0.3, seed=42, **pipeline_kwargs):
    """The *misleading* reference: a stratified random split over all batches.

    Used only to show how much a random split overestimates performance.
    """
    name = name or type(estimator).__name__
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    y_pred = _fit_predict(estimator, variant, X[tr], y[tr], X[te], **pipeline_kwargs)
    return pd.DataFrame([{
        "model": name, "variant": variant, "protocol": "random",
        "train_batches": "all", "test_batch": 0, "n_train": len(tr), "n_test": len(te),
        **compute_metrics(y[te], y_pred),
    }])


def summarize(results):
    """Collapse per-batch rows into one row per ``(model, variant, protocol)``.

    * ``mean_*``      - simple average over test batches (each batch counts equally)
    * ``weighted_*``  - average weighted by test-batch size (each sample counts equally)
    * ``min_macro_f1`` - the worst test batch
    """
    def agg(g):
        w = g["n_test"].to_numpy()
        return pd.Series({
            "n_test_batches": len(g),
            "mean_accuracy": g["accuracy"].mean(),
            "mean_macro_f1": g["macro_f1"].mean(),
            "weighted_accuracy": np.average(g["accuracy"], weights=w),
            "weighted_macro_f1": np.average(g["macro_f1"], weights=w),
            "min_macro_f1": g["macro_f1"].min(),
        })
    return (results.groupby(["model", "variant", "protocol"], sort=False)
            .apply(agg, include_groups=False).reset_index())
