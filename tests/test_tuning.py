"""Tests for src/tuning.py (run with: python -m pytest tests -q)."""
import json

import numpy as np
import pandas as pd
import pytest

from src.data import RAW_DIR, ROOT, load_all
from src.evaluate import evaluate_fixed_train, evaluate_protocol
from src.tuning import (evaluate_tuned, expand_grid, forward_chaining_folds, params_key, select_best,
                        tune_scores, tuning_space, _tuned_estimator)

needs_data = pytest.mark.skipif(not (RAW_DIR / "batch1.dat").exists(), reason="raw data not downloaded")


@pytest.fixture(scope="module")
def data():
    return load_all()


# ---------------- validation folds never look backwards ----------------
def test_forward_chaining_folds():
    assert forward_chaining_folds((1, 2, 3)) == [((1,), 2), ((1, 2), 3)]
    assert forward_chaining_folds((3, 1, 2)) == [((1,), 2), ((1, 2), 3)]      # order of input does not matter
    assert forward_chaining_folds((1,)) == []                                  # one batch: no validation possible
    for fit, val in forward_chaining_folds(range(1, 8)):
        assert max(fit) < val


def test_every_grid_contains_the_default_setting():
    for name, (_, grid, default) in tuning_space().items():
        assert params_key(default) in {params_key(c) for c in expand_grid(grid)}, name
        assert set(default) == set(grid), name


def test_expand_grid_is_complete_and_deterministic():
    g = {"a": [1, 2, 3], "b": ["x", "y"]}
    c = expand_grid(g)
    assert len(c) == 6 and c == expand_grid(g) and {"a": 3, "b": "y"} in c


# ---------------- selection rule ----------------
def _scores(rows):
    return pd.DataFrame([{"model": "kNN", "variant": "raw", "params": params_key(p), "fit_batches": "1", "val_batch": vb,
                          "macro_f1": f, "accuracy": f} for p, vb, f in rows])


def test_select_best_prefers_default_on_ties_and_never_loses_on_validation():
    default = tuning_space()["kNN"][2]
    other = {"n_neighbors": 9, "weights": "uniform"}
    tie = select_best(_scores([(default, 2, 0.6), (default, 3, 0.8), (other, 2, 0.7), (other, 3, 0.7)])).iloc[0]
    assert json.loads(tie.best_params) == default and not tie.changed_from_default and tie.val_gain == 0
    better = select_best(_scores([(default, 2, 0.6), (default, 3, 0.6), (other, 2, 0.7), (other, 3, 0.7)])).iloc[0]
    assert json.loads(better.best_params) == other and better.changed_from_default
    assert better.val_gain == pytest.approx(0.1)


# ---------------- on the real data ----------------
@needs_data
def test_tune_scores_use_only_training_batches(data):
    X, y, b = data
    sc = tune_scores(X, y, b, train_batches=(1, 2, 3), variants=("raw",), models=["Naive Bayes"], n_jobs=2)
    assert len(sc) == 8 * 2                                              # 8 settings x 2 forward-chaining folds
    assert set(sc.val_batch) == {2, 3} and set(sc.fit_batches) == {"1", "1-2"}


@needs_data
def test_fit_once_evaluation_equals_the_per_batch_loop(data):
    X, y, b = data
    from sklearn.neighbors import KNeighborsClassifier
    loop = evaluate_protocol(KNeighborsClassifier(), X, y, b, "P1", "raw", name="kNN")
    once = evaluate_fixed_train(KNeighborsClassifier(), X, y, b, (1, 2, 3), list(range(4, 11)), "raw", name="kNN")
    pd.testing.assert_frame_equal(loop.reset_index(drop=True), once.reset_index(drop=True))


@needs_data
def test_default_settings_in_the_tuning_space_reproduce_the_step4_baselines(data):
    X, y, b = data
    f = ROOT / "results" / "04_baselines_results.csv"
    if not f.exists():
        pytest.skip("baseline results not available")
    base = pd.read_csv(f)
    for name in ["kNN", "SVM", "Naive Bayes", "Decision Tree", "AdaBoost", "Random Forest"]:    # Gradient Boosting skipped (slow)
        est, _, default = tuning_space()[name]
        got = evaluate_fixed_train(est.set_params(**default), X, y, b, (1, 2, 3), list(range(4, 11)), "raw", name=name)
        ref = base[(base.model == name) & (base.protocol == "P1") & (base.variant == "raw")].sort_values("test_batch")
        np.testing.assert_allclose(got.macro_f1.to_numpy(), ref.macro_f1.to_numpy(), atol=1e-9, err_msg=name)


@needs_data
def test_tuned_evaluation_only_uses_untouched_test_batches(data):
    X, y, b = data
    best = pd.DataFrame([{"model": "Naive Bayes", "variant": "raw", "best_params": params_key({"var_smoothing": 1e-6})}])
    res = evaluate_tuned(best, X, y, b, n_jobs=2)
    p1, p3 = res[res.protocol == "P1"], res[res.protocol == "P3"]
    assert list(p1.test_batch) == list(range(4, 11)) and set(p1.train_batches) == {"1-3"}
    assert list(p3.test_batch) == list(range(4, 11))                     # batches 2-3 excluded: they were used for tuning
    assert res.tuned.all()
    assert _tuned_estimator("Naive Bayes", params_key({"var_smoothing": 1e-6})).var_smoothing == 1e-6
