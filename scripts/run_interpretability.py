"""Interpretability experiments (default-setting models, scaling only; results are cached).

Usage:  python scripts/run_interpretability.py

Stage A  09_perm_importance.csv   group permutation importance on test batches 4-10 (model fitted on batches 1-3)
Stage B  09_ablation_results.csv  retraining on feature subsets (only / leave-one-out for types, families, sensors)
Stage C  09_feature_scores.csv    per-feature discriminability and drift, from batches 1-3 ONLY
         09_selection_results.csv selection rules (all / stable / discriminative / balanced / random x50) at 25/50/75 %
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

import pandas as pd                                                    # noqa: E402
from joblib import Parallel, delayed                                   # noqa: E402

from src.data import load_all                                          # noqa: E402
from src.interpret import (evaluate_subset, feature_groups, feature_scores, permutation_task,   # noqa: E402
                           select_features)
from src.models import default_models                                  # noqa: E402

R = ROOT / "results"
N_RANDOM = 50
FRACS = (0.25, 0.5, 0.75)             # 0.5 is the pre-declared main setting; 0.25 / 0.75 are sensitivity checks


def stage_a(X, y, batch, dm):
    f = R / "09_perm_importance.csv"
    if f.exists():
        return
    t = time.time()
    models = {n: dm[n] for n in ["Random Forest", "SVM"]}                # heavy model first
    out = Parallel(n_jobs=-1, verbose=5)(delayed(permutation_task)(n, e, X, y, batch, tb) for n, e in models.items() for tb in range(4, 11))
    pd.concat(out, ignore_index=True).to_csv(f, index=False)
    print(f"stage A done in {time.time() - t:.0f} s", flush=True)


def stage_b(X, y, batch, dm):
    f = R / "09_ablation_results.csv"
    if f.exists():
        return
    t = time.time()
    g = feature_groups(); allc = list(range(128)); jobs = [("all", "all 128 features", allc)]
    for kind in ["type", "family", "sensor"]:
        for name, cols in g[kind].items():
            jobs.append((f"only {kind}", name, cols))
            if kind != "family":
                jobs.append((f"leave out {kind}", name, [c for c in allc if c not in set(cols)]))
    tasks = []
    for kind, name, cols in jobs:
        for model, proto in [("SVM", "P1"), ("SVM", "P3"), ("kNN", "P1")]:
            tasks.append((0 if proto == "P3" else 1, delayed(evaluate_subset)(dm[model], X, y, batch, cols, proto, model, {"subset_kind": kind, "subset": name})))
    tasks.sort(key=lambda a: a[0])
    out = Parallel(n_jobs=-1, verbose=5)(c for _, c in tasks)
    pd.concat(out, ignore_index=True).to_csv(f, index=False)
    print(f"stage B done in {time.time() - t:.0f} s", flush=True)


def stage_c(X, y, batch, dm):
    f = R / "09_selection_results.csv"
    scores = feature_scores(X, y, batch, (1, 2, 3)); scores.to_csv(R / "09_feature_scores.csv", index=False)
    if f.exists():
        return
    t = time.time()
    jobs = [("all", 1.0, 0)]
    for frac in FRACS:
        jobs += [(r, frac, 0) for r in ["stable", "discriminative", "balanced"]]
        jobs += [("random", frac, s) for s in range(N_RANDOM)]
    tasks = []
    for rule, frac, seed in jobs:
        cols = select_features(scores, rule, frac, seed=seed)
        for model in ["Random Forest", "SVM", "kNN"]:
            tasks.append((0 if model == "Random Forest" else 1, delayed(evaluate_subset)(dm[model], X, y, batch, cols, "P1", model, {"rule": rule, "frac": frac, "seed": seed})))
    tasks.sort(key=lambda a: a[0])
    out = Parallel(n_jobs=-1, verbose=5)(c for _, c in tasks)
    pd.concat(out, ignore_index=True).to_csv(f, index=False)
    print(f"stage C done in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    X, y, batch = load_all(); dm = default_models(); t0 = time.time()
    stage_a(X, y, batch, dm); stage_b(X, y, batch, dm); stage_c(X, y, batch, dm)
    (R / "09_runtime_seconds.txt").write_text(str(int(time.time() - t0)))
    print(f"all done in {time.time() - t0:.0f} s")
