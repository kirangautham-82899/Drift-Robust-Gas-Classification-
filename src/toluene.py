"""Toluene-aware evaluation: per-gas metrics and a "how many recent Toluene samples are needed?" experiment.

Background (notebook 06): Toluene is only 79 of the 3,275 training samples of protocol P1 (74 from batch 1) and no model
recognises it on later batches, so the P1 macro-F1 mixes drift with "the model never saw the gas".

Declared before running
-----------------------
* P4: train on batches 1-6, test on batches 7-10 (batch 6 holds 467 Toluene samples; every test batch contains Toluene).
* Learning curve: training set = batches 1-3 + ``n`` samples drawn from batch 6, n in (0, 5, 10, 25, 50, 100, 200, 467),
  5 random draws each, tested on batches 7-10. Arm ``toluene`` draws Toluene samples only; the control arm ``random``
  draws ``n`` samples of any gas (about 20% of them Toluene), to show whether the benefit is specific to Toluene.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.evaluate import _fit_pipeline, compute_metrics

N_GRID = (0, 5, 10, 25, 50, 100, 200, 467)
SEEDS = tuple(range(5))
GASES = [1, 2, 3, 4, 5, 6]


def per_gas_table(y_true, y_pred, batch, batches):
    """Support, predictions, recall, precision and F1 of every gas on every batch (NaN where undefined)."""
    rows = []
    for b in batches:
        m = batch == b
        yt, yp = y_true[m], y_pred[m]
        for g in GASES:
            support, predicted, tp = int((yt == g).sum()), int((yp == g).sum()), int(((yt == g) & (yp == g)).sum())
            rows.append({"batch": b, "gas": g, "support": support, "predicted": predicted, "true_positives": tp,
                         "recall": tp / support if support else np.nan, "precision": tp / predicted if predicted else np.nan,
                         "f1": 2 * tp / (support + predicted) if support + predicted else np.nan})
    return pd.DataFrame(rows)


def augmented_training_indices(batch, y, arm, n, seed, base_batches=(1, 2, 3), pool_batch=6):
    """Row indices of the batches 1-3 training set plus ``n`` extra samples drawn from ``pool_batch``."""
    base = np.where(np.isin(batch, base_batches))[0]
    pool = np.where(batch == pool_batch)[0]
    if arm == "toluene":
        pool = pool[y[pool] == 6]
    elif arm != "random":
        raise ValueError("arm must be 'toluene' or 'random'")
    n = min(n, len(pool))
    extra = np.random.RandomState(seed).choice(pool, size=n, replace=False) if n > 0 else np.empty(0, dtype=int)
    return np.concatenate([base, extra])


def learning_curve_task(model_name, estimator, arm, n, seed, X, y, batch, test_min=7):
    """Fit on the augmented training set (scaling only) and score every test batch >= ``test_min`` separately."""
    idx = augmented_training_indices(batch, y, arm, n, seed)
    assert batch[idx].max() < test_min, "training data must come from before the test batches"
    pipe = _fit_pipeline(estimator, "raw", X[idx], y[idx])
    rows = []
    for tb in range(test_min, 11):
        te = batch == tb
        pred = pipe.predict(X[te]); yt = y[te]; tol = yt == 6
        m = compute_metrics(yt, pred)
        rows.append({"model": model_name, "arm": arm, "n": n, "seed": seed, "test_batch": tb, "n_train": len(idx), "toluene_in_train": int((y[idx] == 6).sum()),
                     "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                     "macro_f1_gases1to5": f1_score(yt, pred, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0),
                     "toluene_recall": float((pred[tol] == 6).mean()), "toluene_f1": f1_score(yt, pred, labels=[6], average="macro", zero_division=0)})
    return rows
