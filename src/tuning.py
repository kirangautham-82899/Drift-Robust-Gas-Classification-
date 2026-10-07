"""Hyperparameter tuning with forward-chaining validation INSIDE the training batches.

The golden rule: **test batches are never used to choose hyperparameters.**

Why forward chaining
--------------------
The data are chronological, so validation must also look forward in time. With training
batches (1, 2, 3) the validation folds are

    fold 1: fit on batch 1        -> validate on batch 2
    fold 2: fit on batches 1, 2   -> validate on batch 3

and a setting is scored by its mean validation macro-F1. (Random K-fold would validate on
near-identical "twins" of the training samples and always favour memorising settings.)

Every grid contains scikit-learn's default setting, so the effect of tuning can be read
off directly (tuned validation score >= default validation score by construction).
"""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import _batch_range, _evaluate_split, _fit_pipeline, compute_metrics, evaluate_fixed_train

SEED = 42


def forward_chaining_folds(train_batches):
    """``[(fit_batches, validation_batch), ...]`` that never train on the future."""
    tb = sorted(train_batches)
    return [(tuple(tb[:j]), tb[j]) for j in range(1, len(tb))]


def tuning_space():
    """``{model: (estimator, grid, default_params)}``; each grid contains the default setting."""
    return {
        "kNN": (KNeighborsClassifier(),
                {"n_neighbors": [1, 3, 5, 9, 15, 25], "weights": ["uniform", "distance"]},
                {"n_neighbors": 5, "weights": "uniform"}),
        "SVM": (SVC(),
                {"C": [0.1, 1.0, 10.0, 100.0], "gamma": ["scale", 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]},
                {"C": 1.0, "gamma": "scale"}),
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                          {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 5, 20]},
                          {"max_depth": None, "min_samples_leaf": 1}),
        "Random Forest": (RandomForestClassifier(random_state=SEED),
                          {"max_depth": [8, 16, None], "max_features": ["sqrt", 0.3], "min_samples_leaf": [1, 5]},
                          {"max_depth": None, "max_features": "sqrt", "min_samples_leaf": 1}),
        "Naive Bayes": (GaussianNB(),
                        {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]},
                        {"var_smoothing": 1e-9}),
        "AdaBoost": (AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED), random_state=SEED),
                     {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1.0], "estimator__max_depth": [1, 2, 3]},
                     {"n_estimators": 50, "learning_rate": 1.0, "estimator__max_depth": 1}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED),
                              {"learning_rate": [0.05, 0.1, 0.2], "max_depth": [2, 3, 4]},
                              {"learning_rate": 0.1, "max_depth": 3}),
    }


def expand_grid(grid):
    """All combinations of a ``{param: [values]}`` grid, in a deterministic order."""
    combos = [{}]
    for key, values in grid.items():
        combos = [{**c, key: v} for c in combos for v in values]
    return combos


def params_key(params):
    return json.dumps(params, sort_keys=True, default=str)


def _score_one(name, est, params, variant, X, y, batch, fit_batches, val_batch, idx):
    model = clone(est).set_params(**params)
    tr, va = np.isin(batch, fit_batches), batch == val_batch
    assert max(fit_batches) < val_batch                       # validation is always in the future of the fit data
    pipe = _fit_pipeline(model, variant, X[tr], y[tr])
    m = compute_metrics(y[va], pipe.predict(X[va]))
    return {"idx": idx, "model": name, "variant": variant, "params": params_key(params),
            "fit_batches": _batch_range(fit_batches), "val_batch": val_batch,
            "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}


_HEAVY_FIRST = ["Gradient Boosting", "AdaBoost", "Random Forest", "SVM", "Decision Tree", "kNN", "Naive Bayes"]


def tune_scores(X, y, batch, train_batches=(1, 2, 3), variants=("raw", "PCA", "LDA"), models=None, n_jobs=-1, verbose=0):
    """Score every grid combination on every forward-chaining fold (parallel, heavy models first)."""
    space = tuning_space()
    names = [m for m in (models or list(space)) if m in space]
    folds = forward_chaining_folds(train_batches)
    tasks = []
    for name in sorted(names, key=_HEAVY_FIRST.index):
        est, grid, _ = space[name]
        for variant in variants:
            for params in expand_grid(grid):
                for fit_b, val_b in folds:
                    tasks.append((name, est, params, variant, fit_b, val_b))
    out = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_score_one)(n, e, p, v, X, y, batch, fb, vb, i) for i, (n, e, p, v, fb, vb) in enumerate(tasks))
    return pd.DataFrame(out).sort_values("idx").drop(columns="idx").reset_index(drop=True)


def select_best(scores):
    """Best setting per (model, variant) by mean validation macro-F1.

    Ties are broken in favour of the scikit-learn default, then by grid order, so the result is deterministic
    and tuning never changes a setting without a validation gain.
    """
    space = tuning_space()
    rows = []
    for (model, variant), g in scores.groupby(["model", "variant"], sort=False):
        agg = g.groupby("params", sort=False).agg(val_macro_f1=("macro_f1", "mean"), val_accuracy=("accuracy", "mean"),
                                                  n_folds=("macro_f1", "size")).reset_index()
        default_key = params_key(space[model][2])
        agg["is_default"] = agg["params"] == default_key
        agg["order"] = range(len(agg))
        best = agg.sort_values(["val_macro_f1", "is_default", "order"], ascending=[False, False, True]).iloc[0]
        d = agg[agg.is_default].iloc[0]
        rows.append({"model": model, "variant": variant, "best_params": best["params"],
                     "val_macro_f1_tuned": best["val_macro_f1"], "val_macro_f1_default": d["val_macro_f1"],
                     "val_gain": best["val_macro_f1"] - d["val_macro_f1"],
                     "changed_from_default": not best["is_default"], "n_settings": len(agg), "n_folds": int(best["n_folds"])})
    return pd.DataFrame(rows)


def _tuned_estimator(name, params_json):
    est = clone(tuning_space()[name][0])
    return est.set_params(**json.loads(params_json))


def _eval_p1(name, variant, params_json, X, y, batch):
    df = evaluate_fixed_train(_tuned_estimator(name, params_json), X, y, batch, (1, 2, 3), list(range(4, 11)),
                              variant, name=name, protocol="P1")
    return df


def _eval_p3(name, variant, params_json, X, y, batch, k):
    row, _ = _evaluate_split(_tuned_estimator(name, params_json), X, y, batch, tuple(range(1, k)), k, "P3", variant, name)
    return pd.DataFrame([row])


def evaluate_tuned(best, X, y, batch, n_jobs=-1, verbose=0, include_p3=True):
    """Test the tuned settings. P1 = train B1-3 / test B4-10 (one fit per model+pipeline).

    P3 is limited to test batches k >= 4: the settings were chosen on batches 1-3, so test batches 2 and 3 would leak.
    """
    tasks = []                                         # (priority, delayed call); heavy models start first
    for _, r in best.iterrows():
        prio = _HEAVY_FIRST.index(r.model)
        tasks.append((prio, delayed(_eval_p1)(r.model, r.variant, r.best_params, X, y, batch)))
        if include_p3:
            tasks.extend((prio, delayed(_eval_p3)(r.model, r.variant, r.best_params, X, y, batch, k)) for k in range(4, 11))
    tasks.sort(key=lambda t: t[0])
    out = Parallel(n_jobs=n_jobs, verbose=verbose)(call for _, call in tasks)
    res = pd.concat(out, ignore_index=True)
    res["tuned"] = True
    return res.sort_values(["model", "variant", "protocol", "test_batch"]).reset_index(drop=True)
