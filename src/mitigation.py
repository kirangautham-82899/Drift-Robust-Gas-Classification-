"""Drift-mitigation methods for the rolling (P3) and fixed (P1) protocols.

Settings were **declared before any result was seen** (so they cannot be cherry-picked):
main = window 2, half-life 2 batches, CORAL regularisation 1.0. The other window / half-life
values are reported as a sensitivity check only.

Assumptions (always state them when reporting a method)
-------------------------------------------------------
* ``window`` / ``half_life``  need *labelled* data from recent batches (periodic re-calibration).
* ``batch`` / ``coral``       need **no labels**, but are *transductive*: they use the **unlabelled
  feature readings of the whole test batch** (never its labels) to estimate its statistics.
Test labels are never used by any method (tested).

Methods
-------
window     train only on the last ``w`` batches before the test batch.
recency    weight training samples by ``0.5 ** (age / half_life)``; age = 0 for the newest training batch.
batch      standardise every batch with its OWN per-feature mean and standard deviation
           (training batches individually, the test batch with its own statistics).
coral      CORrelation ALignment (Sun et al., 2016): standardise with the training scaler, then re-colour the
           training features so their covariance matches the unlabelled test batch: Zs @ Cs^-1/2 @ Ct^1/2,
           with C = cov + reg * I.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.evaluate import compute_metrics, make_pipeline, protocol_splits

# name -> specification (None/absent keys mean "not used")
METHODS = {
    "baseline": {},
    "window-1": {"window": 1},
    "window-2": {"window": 2},
    "window-3": {"window": 3},
    "recency-h1": {"half_life": 1.0},
    "recency-h2": {"half_life": 2.0},
    "batch-std": {"standardize": "batch"},
    "coral": {"standardize": "coral", "coral_reg": 1.0},
    "window-2 + batch-std": {"window": 2, "standardize": "batch"},
}
MAIN_METHODS = ["window-2", "recency-h2", "batch-std", "coral"]          # pre-declared
P1_METHODS = ["baseline", "batch-std", "coral"]                           # fixed training set: only unlabelled-target methods apply


def batchwise_standardize(X, batch, eps=1e-8):
    """z-score each batch with its own per-feature mean and standard deviation."""
    X = np.asarray(X, dtype=float)
    out = np.empty_like(X)
    for b in np.unique(batch):
        m = batch == b
        mu, sd = X[m].mean(axis=0), X[m].std(axis=0)
        out[m] = (X[m] - mu) / np.where(sd < eps, 1.0, sd)
    return out


def _sym_power(C, power):
    w, V = np.linalg.eigh(C)
    return (V * np.maximum(w, 1e-12) ** power) @ V.T


def coral_align(Zs, Zt, reg=1.0):
    """Re-colour source features ``Zs`` so that their covariance matches the (unlabelled) target ``Zt``."""
    d = Zs.shape[1]
    Cs = np.cov(Zs, rowvar=False) + reg * np.eye(d)
    Ct = np.cov(Zt, rowvar=False) + reg * np.eye(d)
    return Zs @ _sym_power(Cs, -0.5) @ _sym_power(Ct, 0.5)


def supports(estimator, spec):
    """Recency weighting needs an estimator whose ``fit`` accepts ``sample_weight`` (kNN does not)."""
    import inspect
    return "half_life" not in spec or "sample_weight" in inspect.signature(estimator.fit).parameters


def predict_mitigated(estimator, variant, X, y, batch, train_batches, test_batch, spec, random_state=42):
    """Fit on ``train_batches`` (after applying the mitigation ``spec``) and predict ``test_batch``.

    ``y`` is only read for training rows; test labels are never touched.
    """
    tb = sorted(train_batches)
    assert max(tb) < test_batch, "a time-aware split must never train on the future"
    if spec.get("window"):
        tb = tb[-spec["window"]:]
    tr, te = np.isin(batch, tb), batch == test_batch
    Xtr, ytr, btr, Xte = X[tr], y[tr], batch[tr], X[te]

    mode = spec.get("standardize", "global")
    scale = True
    if mode == "batch":
        Xtr, Xte, scale = batchwise_standardize(Xtr, btr), batchwise_standardize(Xte, batch[te]), False
    elif mode == "coral":
        sc = StandardScaler().fit(Xtr)
        Zs, Zt = sc.transform(Xtr), sc.transform(Xte)
        Xtr, Xte, scale = coral_align(Zs, Zt, spec.get("coral_reg", 1.0)), Zt, False

    pipe = make_pipeline(estimator, variant, random_state=random_state, scale=scale)
    if variant == "LDA":
        pipe.set_params(dr__n_components=min(5, len(np.unique(ytr)) - 1))
    fit_params = {}
    if spec.get("half_life"):
        age = (test_batch - 1) - btr                      # 0 = the newest possible training batch
        fit_params["clf__sample_weight"] = 0.5 ** (age / spec["half_life"])
    pipe.fit(Xtr, ytr, **fit_params)
    return pipe.predict(Xte), ytr


def _one(name, est, variant, protocol, method, X, y, batch, train_batches, test_batch):
    spec = METHODS[method]
    pred, ytr = predict_mitigated(est, variant, X, y, batch, train_batches, test_batch, spec)
    te = batch == test_batch; yt = y[te]; tol = yt == 6
    m = compute_metrics(yt, pred)
    return {"model": name, "variant": variant, "protocol": protocol, "method": method, "test_batch": test_batch,
            "train_batches": f"{min(train_batches)}-{max(train_batches)}", "n_test": int(te.sum()),
            "n_train_used": int(len(ytr)), "toluene_in_train": int((ytr == 6).sum()),
            "accuracy": m["accuracy"], "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "macro_f1_gases1to5": f1_score(yt, pred, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0),
            "toluene_recall": float((pred[tol] == 6).mean()) if tol.any() else np.nan}


_HEAVY = {"Random Forest": 0, "SVM": 1, "Naive Bayes": 2, "kNN": 3}


def run_grid(models, X, y, batch, variants=("raw", "LDA"), protocols=("P3", "P1"), n_jobs=-1, verbose=0):
    """All (model, pipeline, protocol, method, test batch) combinations, in parallel (heavy models first)."""
    tasks = []
    for name, est in models.items():
        for variant in variants:
            for protocol in protocols:
                methods = list(METHODS) if protocol == "P3" else P1_METHODS
                for method in methods:
                    if not supports(est, METHODS[method]):
                        continue
                    for train_b, test_b in protocol_splits(protocol):
                        tasks.append((_HEAVY.get(name, 9), (name, est, variant, protocol, method, X, y, batch, train_b, test_b)))
    tasks.sort(key=lambda t: t[0])
    out = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(_one)(*a) for _, a in tasks)
    return pd.DataFrame(out).sort_values(["model", "variant", "protocol", "method", "test_batch"]).reset_index(drop=True)
