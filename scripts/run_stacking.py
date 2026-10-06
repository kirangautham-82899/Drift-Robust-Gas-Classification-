"""Run the preliminary stacking grid (heavy: Gradient Boosting is slow) and cache the results.

Usage:  python scripts/run_stacking.py
Writes results/05_stacking_results.csv and results/05_runtime_seconds.txt.
Notebook 05 loads these files.
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

from src.data import load_all          # noqa: E402
from src.evaluate import evaluate_grid  # noqa: E402
from src.stacking import make_stackers  # noqa: E402

if __name__ == "__main__":
    X, y, batch = load_all()
    t0 = time.time()
    res = evaluate_grid(make_stackers(), X, y, batch, variants=("raw", "PCA", "LDA"),
                        protocols=("P1", "P2"), split_level=True, verbose=10)
    (ROOT / "results" / "05_runtime_seconds.txt").write_text(str(int(time.time() - t0)))
    res.to_csv(ROOT / "results" / "05_stacking_results.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s | rows: {len(res)}")
