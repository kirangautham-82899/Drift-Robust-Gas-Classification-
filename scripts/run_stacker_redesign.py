"""Redesigned (forward-chaining) stacker vs the earlier fold designs, with identical experts (heavy-ish; cached).

Usage:  python scripts/run_stacker_redesign.py
Experts (declared in advance): SVM, kNN, Random Forest, Naive Bayes, all defaults; meta-learner: default logistic regression.
Only the way out-of-fold predictions are produced differs: forward chaining / leave-one-batch-out / random folds.
Writes per-sample predictions to results/11_predictions_P1.csv (batches 4-10; pipelines raw, PCA, LDA) and
results/11_predictions_P3.csv (batches 2-10; raw features, rolling retraining).
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

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
from joblib import Parallel, delayed                                          # noqa: E402

from src.data import load_all                                                 # noqa: E402
from src.predictions import fixed_train_predictions, rolling_stack_prediction  # noqa: E402
from src.stacking import make_fast_stackers                                   # noqa: E402

R = ROOT / "results"

if __name__ == "__main__":
    X, y, batch = load_all(); t0 = time.time(); tasks = []
    for name, st in make_fast_stackers().items():
        for v in ["raw", "PCA", "LDA"]:
            tasks.append((0, delayed(fixed_train_predictions)(f"P1|{name}|{v}|default", st, v, X, y, batch)))
        for k in range(2, 11):
            tasks.append((10 - k + 1, delayed(rolling_stack_prediction)(f"P3|{name}|raw|default", st, "raw", X, y, batch, k)))
    tasks.sort(key=lambda t: t[0])
    out = Parallel(n_jobs=-1, verbose=5)(c for _, c in tasks)
    p1, p3 = {}, {}
    for o in out:
        if len(o) == 2:
            p1[o[0]] = o[1]
        else:
            p3.setdefault(o[0], {})[o[1]] = o[2]
    d1 = pd.DataFrame(dict(sorted(p1.items()))); d3 = pd.DataFrame({k: np.concatenate([v[kk] for kk in range(2, 11)]) for k, v in sorted(p3.items())})
    assert len(d1) == int((batch >= 4).sum()) and len(d3) == int((batch >= 2).sum())
    d1.to_csv(R / "11_predictions_P1.csv", index=False); d3.to_csv(R / "11_predictions_P3.csv", index=False)
    (R / "11_runtime_seconds.txt").write_text(str(int(time.time() - t0)))
    print(f"done in {time.time() - t0:.0f} s | P1 columns: {d1.shape[1]} | P3 columns: {d3.shape[1]}")
