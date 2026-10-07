"""Hyperparameter tuning (heavy) with forward-chaining validation inside training batches 1-3.

Usage:  python scripts/run_tuning.py            (each finished stage is cached; delete its CSV to recompute)

Stage 1  results/07_tuning_scores.csv      every setting x every validation fold (batches 1-3 only)
Stage 2  results/07_best_params.csv        best setting per model and pipeline (by mean validation macro-F1)
Stage 3  results/07_tuned_test_results.csv tuned models on the untouched test batches (P1: B4-10; P3: B4-10)
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

import pandas as pd                                                         # noqa: E402

from src.data import load_all                                               # noqa: E402
from src.tuning import evaluate_tuned, select_best, tune_scores             # noqa: E402

if __name__ == "__main__":
    R = ROOT / "results"
    X, y, batch = load_all()
    t_all = time.time()

    f1 = R / "07_tuning_scores.csv"
    if not f1.exists():
        t = time.time()
        tune_scores(X, y, batch, train_batches=(1, 2, 3), verbose=10).to_csv(f1, index=False)
        print(f"stage 1 (validation scores) done in {time.time() - t:.0f} s", flush=True)
    scores = pd.read_csv(f1)

    f2 = R / "07_best_params.csv"
    select_best(scores).to_csv(f2, index=False)
    best = pd.read_csv(f2)
    print(best[["model", "variant", "best_params", "val_macro_f1_default", "val_macro_f1_tuned", "val_gain"]].round(3).to_string(index=False), flush=True)

    f3 = R / "07_tuned_test_results.csv"
    if not f3.exists():
        t = time.time()
        evaluate_tuned(best, X, y, batch, verbose=10).to_csv(f3, index=False)
        print(f"stage 3 (test evaluation) done in {time.time() - t:.0f} s", flush=True)

    (R / "07_runtime_seconds.txt").write_text(str(int(time.time() - t_all)))
    print(f"all done in {time.time() - t_all:.0f} s")
