"""Collect per-sample predictions of every configuration used in the significance tests (heavy, cached).

Usage:  python scripts/run_predictions.py
Writes results/10_predictions_P1.csv (rows = samples of batches 4-10) and results/10_predictions_P3.csv
(rows = samples of batches 2-10); one column per configuration, named ``protocol|model|pipeline|kind``.
Settings are the ones declared in earlier steps: defaults, tuned (07_best_params.csv), stacking, the pre-declared
50% feature selections, and the pre-declared mitigation methods. Nothing is re-tuned or re-selected here.
"""
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                                           # noqa: E402
import pandas as pd                                                          # noqa: E402
from joblib import Parallel, delayed                                         # noqa: E402

from src.data import load_all                                                # noqa: E402
from src.interpret import feature_scores, select_features                    # noqa: E402
from src.models import default_models, reference_model                       # noqa: E402
from src.predictions import fixed_train_predictions, mitigated_p1_predictions, rolling_prediction   # noqa: E402
from src.stacking import make_stackers                                       # noqa: E402
from src.tuning import _tuned_estimator                                      # noqa: E402

R = ROOT / "results"
VARIANTS = ["raw", "PCA", "LDA"]

if __name__ == "__main__":
    X, y, batch = load_all(); dm = default_models(); t0 = time.time()
    p1_tasks, p3_tasks = [], []                                              # (priority, delayed call)

    # --- P1: defaults, tuned settings, majority floor, stacking ---------------------------------------------------
    for m, est in dm.items():
        for v in VARIANTS:
            heavy = 0 if m == "Gradient Boosting" else 1
            p1_tasks.append((heavy, delayed(fixed_train_predictions)(f"P1|{m}|{v}|default", est, v, X, y, batch)))
    best = pd.read_csv(R / "07_best_params.csv")
    for _, r in best.iterrows():
        heavy = 0 if r.model == "Gradient Boosting" else 1
        p1_tasks.append((heavy, delayed(fixed_train_predictions)(f"P1|{r.model}|{r.variant}|tuned", _tuned_estimator(r.model, r.best_params), r.variant, X, y, batch)))
    p1_tasks.append((1, delayed(fixed_train_predictions)("P1|Majority class|raw|default", reference_model()["Majority class"], "raw", X, y, batch)))
    for name, st in make_stackers().items():
        for v in VARIANTS:
            p1_tasks.append((0, delayed(fixed_train_predictions)(f"P1|{name}|{v}|default", st, v, X, y, batch)))

    # --- P1: pre-declared 50% feature selections (scores from batches 1-3 only) ----------------------------------------
    scores = feature_scores(X, y, batch, (1, 2, 3))
    for rule in ["stable", "discriminative", "balanced"]:
        cols = select_features(scores, rule, 0.5)
        for m in ["SVM", "Random Forest", "kNN"]:
            p1_tasks.append((1, delayed(fixed_train_predictions)(f"P1|{m}|raw|sel-{rule}", dm[m], "raw", X, y, batch, cols)))

    # --- P1: label-free mitigation (per-batch standardisation, CORAL) -------------------------------------------------
    for m in ["SVM", "kNN", "Random Forest", "Naive Bayes"]:
        for v in ["raw", "LDA"]:
            for meth in ["batch-std", "coral"]:
                p1_tasks.append((1, delayed(mitigated_p1_predictions)(f"P1|{m}|{v}|{meth}", dm[m], v, meth, X, y, batch)))

    # --- P3 (rolling): all methods for SVM and Random Forest; baseline for the other models ---------------------------
    from src.mitigation import METHODS
    for m in ["SVM", "Random Forest"]:
        for meth in METHODS:
            for k in range(2, 11):
                p3_tasks.append((1, delayed(rolling_prediction)(f"P3|{m}|raw|{meth}", dm[m], "raw", meth, X, y, batch, k)))
    for m in ["kNN", "Decision Tree", "Naive Bayes", "AdaBoost", "Gradient Boosting"]:
        for k in range(2, 11):
            p3_tasks.append((0 if m == "Gradient Boosting" else 1, delayed(rolling_prediction)(f"P3|{m}|raw|baseline", dm[m], "raw", "baseline", X, y, batch, k)))

    tasks = sorted(p1_tasks + p3_tasks, key=lambda t: t[0])
    print(f"{len(p1_tasks)} fixed-train tasks + {len(p3_tasks)} rolling tasks", flush=True)
    out = Parallel(n_jobs=-1, verbose=5)(c for _, c in tasks)

    p1 = {}; p3 = {}
    for o in out:
        if len(o) == 2:
            p1[o[0]] = o[1]
        else:
            p3.setdefault(o[0], {})[o[1]] = o[2]
    n1, n3 = int((batch >= 4).sum()), int((batch >= 2).sum())
    d1 = pd.DataFrame({k: v for k, v in sorted(p1.items())}); assert len(d1) == n1
    d3 = pd.DataFrame({k: np.concatenate([v[kk] for kk in range(2, 11)]) for k, v in sorted(p3.items())}); assert len(d3) == n3
    d1.to_csv(R / "10_predictions_P1.csv", index=False); d3.to_csv(R / "10_predictions_P3.csv", index=False)
    (R / "10_runtime_seconds.txt").write_text(str(int(time.time() - t0)))
    print(f"done in {time.time() - t0:.0f} s | P1 columns: {d1.shape[1]} | P3 columns: {d3.shape[1]}")
