"""Random-split reference (the misleading evaluation) for all seven baseline models.

Usage:  python scripts/run_random_baselines.py
Writes results/06_random_split_baselines.csv (stratified 70/30 split, seed 42, raw features).
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

import pandas as pd                                   # noqa: E402
from joblib import Parallel, delayed                  # noqa: E402

from src.data import load_all                         # noqa: E402
from src.evaluate import random_split_baseline        # noqa: E402
from src.models import default_models                 # noqa: E402

if __name__ == "__main__":
    X, y, _ = load_all()
    t0 = time.time()
    parts = Parallel(n_jobs=-1, verbose=5)(
        delayed(random_split_baseline)(est, X, y, "raw", name=name) for name, est in default_models().items()
    )
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(ROOT / "results" / "06_random_split_baselines.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s")
    print(out[["model", "accuracy", "macro_f1"]].round(3).to_string(index=False))
