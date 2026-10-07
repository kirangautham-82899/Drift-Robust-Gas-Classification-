"""Vergara-style per-batch SVM ensemble under rolling retraining (P3), accuracy-weighted and equal-weighted.

Usage:  python scripts/run_ensemble.py
Writes results/13_predictions_P3.csv (rows = samples of batches 2-10; one column per variant) and results/13_runtime_seconds.txt.
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

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from joblib import Parallel, delayed                  # noqa: E402
from sklearn.svm import SVC                           # noqa: E402

from src.data import load_all                         # noqa: E402
from src.ensemble import ensemble_predict             # noqa: E402

if __name__ == "__main__":
    X, y, batch = load_all(); t0 = time.time()
    jobs = [(w, k) for w in ["accuracy", "equal"] for k in range(2, 11)]
    out = Parallel(n_jobs=-1, verbose=5)(delayed(ensemble_predict)(SVC(), X, y, batch, k, w) for w, k in jobs)
    cols = {}
    for w in ["accuracy", "equal"]:
        cols[f"P3|SVM|raw|ensemble-{w}"] = np.concatenate([o.astype(np.int8) for (ww, k), o in zip(jobs, out) if ww == w])
    d = pd.DataFrame(cols); assert len(d) == int((batch >= 2).sum())
    d.to_csv(ROOT / "results" / "13_predictions_P3.csv", index=False)
    (ROOT / "results" / "13_runtime_seconds.txt").write_text(str(int(time.time() - t0)))
    print(f"done in {time.time() - t0:.0f} s | columns: {list(d.columns)}")
