"""Drift-mitigation experiments (default-setting models; heavy, results are cached).

Usage:  python scripts/run_mitigation.py

Writes results/08_mitigation_results.csv and results/08_runtime_seconds.txt.
Models: SVM, kNN, Random Forest, Naive Bayes; pipelines: raw (scaling only) and LDA.
P3 (rolling retraining, test batches 2-10): every method in src/mitigation.METHODS.
P1 (train batches 1-3, test 4-10): baseline, per-batch standardisation, CORAL.
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

from src.data import load_all                 # noqa: E402
from src.mitigation import run_grid           # noqa: E402
from src.models import default_models         # noqa: E402

if __name__ == "__main__":
    X, y, batch = load_all()
    dm = default_models()
    models = {n: dm[n] for n in ["SVM", "kNN", "Random Forest", "Naive Bayes"]}
    t0 = time.time()
    res = run_grid(models, X, y, batch, variants=("raw", "LDA"), protocols=("P3", "P1"), verbose=10)
    (ROOT / "results" / "08_runtime_seconds.txt").write_text(str(int(time.time() - t0)))
    res.to_csv(ROOT / "results" / "08_mitigation_results.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s | rows: {len(res)}")
