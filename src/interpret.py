"""Interpretability: which sensors / feature types carry the gas signal, and which of them drift?

Feature layout (see ``src/data.py``): feature ``8 * sensor + type`` with 16 sensors and 8 feature types
(DR, |DR|, three rising EMAs, three decaying EMAs). Three groupings are used:

* ``sensor``  - 16 groups of 8 features (one physical sensor)
* ``type``    - 8 groups of 16 features (one kind of feature, across all sensors)
* ``family``  - steady-state {DR, |DR|}, rising EMAs, decaying EMAs

Rules that keep the analysis honest
-----------------------------------
* **Permutation importance** is an *analysis of a fitted model*: it scores the model on test batches (so it uses
  their labels) and must NOT be used to choose features that are then tested on the same batches.
* **Feature selection** (``feature_scores`` + ``select_features``) uses ONLY the training batches (labels allowed
  there); the fraction kept is declared in advance. The selected features are then tested on later batches.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from src.data import FEATURE_TYPES, FEATURES_PER_SENSOR, N_FEATURES, N_SENSORS, feature_names
from src.evaluate import _fit_pipeline, compute_metrics, evaluate_fixed_train, evaluate_protocol

FAMILIES = {"steady-state (DR, |DR|)": [0, 1], "rising EMA": [2, 3, 4], "decaying EMA": [5, 6, 7]}


def feature_groups():
    """``{"sensor": {...}, "type": {...}, "family": {...}}`` with lists of column indices."""
    sensors = {f"S{s + 1:02d}": [s * FEATURES_PER_SENSOR + t for t in range(FEATURES_PER_SENSOR)] for s in range(N_SENSORS)}
    types = {name: [s * FEATURES_PER_SENSOR + t for s in range(N_SENSORS)] for t, name in enumerate(FEATURE_TYPES)}
    families = {name: [s * FEATURES_PER_SENSOR + t for s in range(N_SENSORS) for t in ts] for name, ts in FAMILIES.items()}
    return {"sensor": sensors, "type": types, "family": families}


def permutation_importance_groups(model, X_te, y_te, groups, n_repeats=10, seed=42):
    """Mean drop in macro-F1 when a whole group of features is shuffled together across the samples of a batch.

    All columns of a group share ONE row permutation, so within-group correlations are preserved (single-feature
    permutation is uninformative here because features are highly redundant).
    Returns a DataFrame ``group, importance_mean, importance_std`` (macro-F1 drop; larger = more important).
    """
    rng = np.random.RandomState(seed)
    base = compute_metrics(y_te, model.predict(X_te))["macro_f1"]
    rows = []
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X_te.copy()
            Xp[:, cols] = X_te[rng.permutation(len(X_te))][:, cols]
            drops.append(base - compute_metrics(y_te, model.predict(Xp))["macro_f1"])
        rows.append({"group": name, "importance_mean": float(np.mean(drops)), "importance_std": float(np.std(drops))})
    return pd.DataFrame(rows)


def discriminability(X, y):
    """ANOVA F-statistic of the gas label for every feature (higher = the gases differ more along that feature)."""
    F, _ = f_classif(X, y)
    return np.nan_to_num(F)


def class_conditional_drift(X, y, batch, ref=1, later=(2, 3), min_n=10, eps=1e-9):
    """How far each feature's per-gas median moves from the reference batch to later batches.

    For every later batch and every gas with at least ``min_n`` samples in both batches the shift is
    ``|median_later - median_ref| / robust spread`` (IQR / 1.349 of the two batches pooled); the result is the mean
    over (batch, gas) pairs. Uses labels, so it must only be called with TRAINING batches.
    """
    shifts = []
    for later_b in later:
        for g in np.unique(y):
            a = X[(batch == ref) & (y == g)]
            b = X[(batch == later_b) & (y == g)]
            if len(a) < min_n or len(b) < min_n:
                continue
            both = np.vstack([a, b])
            spread = (np.percentile(both, 75, axis=0) - np.percentile(both, 25, axis=0)) / 1.349
            shifts.append(np.abs(np.median(b, axis=0) - np.median(a, axis=0)) / np.maximum(spread, eps))
    if not shifts:
        raise ValueError("no gas has enough samples in both the reference and a later batch")
    return np.mean(shifts, axis=0)


def feature_scores(X, y, batch, train_batches=(1, 2, 3)):
    """Per-feature discriminability and drift, estimated from the training batches ONLY."""
    tb = sorted(train_batches)
    m = np.isin(batch, tb)
    F = discriminability(X[m], y[m])
    drift = class_conditional_drift(X[m], y[m], batch[m], ref=tb[0], later=tuple(tb[1:]))
    names = feature_names()
    return pd.DataFrame({"feature": range(N_FEATURES), "name": names, "sensor": [n.split("_")[0] for n in names],
                         "type": [FEATURE_TYPES[i % FEATURES_PER_SENSOR] for i in range(N_FEATURES)], "F": F, "drift": drift})


def select_features(scores, rule, frac=0.5, seed=None):
    """Pre-declared selection rules; returns sorted column indices.

    ``all``           every feature
    ``stable``        the ``frac`` of features with the LOWEST drift
    ``discriminative`` the ``frac`` with the highest F
    ``balanced``      the ``frac`` with the highest ``F / (1 + drift)``
    ``random``        a random ``frac`` of the features (null baseline)
    """
    n = len(scores); k = max(1, int(round(frac * n)))
    if rule == "all":
        return np.arange(n)
    if rule == "stable":
        return np.sort(scores.sort_values(["drift", "feature"]).feature.to_numpy()[:k])
    if rule == "discriminative":
        return np.sort(scores.sort_values(["F", "feature"], ascending=[False, True]).feature.to_numpy()[:k])
    if rule == "balanced":
        s = scores.assign(score=scores.F / (1.0 + scores.drift))
        return np.sort(s.sort_values(["score", "feature"], ascending=[False, True]).feature.to_numpy()[:k])
    if rule == "random":
        return np.sort(np.random.RandomState(seed).choice(n, k, replace=False))
    raise ValueError(f"unknown rule {rule!r}")


# ---------------------------------------------------------------- workers used by scripts/run_interpretability.py
def evaluate_subset(estimator, X, y, batch, cols, protocol, name, meta):
    """Retrain on a feature subset and evaluate under P1 (fit once) or another protocol; ``meta`` adds labelling columns."""
    cols = [int(c) for c in cols]
    Xs = X[:, cols]
    if protocol == "P1":
        df = evaluate_fixed_train(estimator, Xs, y, batch, (1, 2, 3), list(range(4, 11)), "raw", name=name, protocol="P1")
    else:
        df = evaluate_protocol(estimator, Xs, y, batch, protocol, "raw", name=name)
    for k, v in meta.items():
        df[k] = v
    df["n_features"] = len(cols)
    return df


def permutation_task(name, estimator, X, y, batch, test_batch, n_repeats=10, seed=42):
    """Fit on batches 1-3 (scaling only), then permute every sensor / type / family group on ONE test batch."""
    tr, te = batch <= 3, batch == test_batch
    model = _fit_pipeline(estimator, "raw", X[tr], y[tr])
    parts = []
    for kind, groups in feature_groups().items():
        df = permutation_importance_groups(model, X[te], y[te], groups, n_repeats=n_repeats, seed=seed)
        df.insert(0, "group_kind", kind); parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "test_batch", test_batch); out.insert(0, "model", name)
    return out
