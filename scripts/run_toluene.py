"""Toluene-aware experiments (default-setting models; cached).

Usage:  python scripts/run_toluene.py
Stage A  results/12_predictions_P4.csv   protocol P4 (train batches 1-6, test 7-10): per-sample predictions, 7 models x 3 pipelines
Stage B  results/12_learning_curve.csv   batches 1-3 + n samples from batch 6 (Toluene-only arm vs random control), tested on 7-10
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

import pandas as pd                                                          # noqa: E402
from joblib import Parallel, delayed                                         # noqa: E402

from src.data import load_all                                                # noqa: E402
from src.models import default_models                                        # noqa: E402
from src.predictions import fixed_split_predictions                          # noqa: E402
from src.toluene import N_GRID, SEEDS, learning_curve_task                   # noqa: E402

R = ROOT / "results"

if __name__ == "__main__":
    X, y, batch = load_all(); dm = default_models(); t0 = time.time(); tasks = []
    for m, est in dm.items():
        for v in ["raw", "PCA", "LDA"]:
            tasks.append((0 if m == "Gradient Boosting" else 1, "A", delayed(fixed_split_predictions)(f"P4|{m}|{v}|default", est, v, X, y, batch, 6, 7)))
    for m in ["Random Forest", "SVM", "kNN"]:
        for arm in ["toluene", "random"]:
            for n in N_GRID:
                for s in SEEDS:
                    tasks.append((2, "B", delayed(learning_curve_task)(m, dm[m], arm, n, s, X, y, batch)))
    tasks.sort(key=lambda t: t[0])
    print(sum(t[1] == "A" for t in tasks), "P4 tasks +", sum(t[1] == "B" for t in tasks), "learning-curve tasks", flush=True)
    out = Parallel(n_jobs=-1, verbose=5)(c for _, _, c in tasks)
    a = {o[0]: o[1] for (_, kind, _), o in zip(tasks, out) if kind == "A"}
    pd.DataFrame(dict(sorted(a.items()))).to_csv(R / "12_predictions_P4.csv", index=False)
    rows = [r for (_, kind, _), o in zip(tasks, out) if kind == "B" for r in o]
    pd.DataFrame(rows).to_csv(R / "12_learning_curve.csv", index=False)
    (R / "12_runtime_seconds.txt").write_text(str(int(time.time() - t0)))
    print(f"done in {time.time() - t0:.0f} s")
