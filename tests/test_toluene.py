"""Tests for src/toluene.py (run with: python -m pytest tests -q)."""
import numpy as np
import pytest
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

from src.data import RAW_DIR, load_all
from src.toluene import augmented_training_indices, learning_curve_task, per_gas_table

needs_data = pytest.mark.skipif(not (RAW_DIR / "batch1.dat").exists(), reason="raw data not downloaded")


def test_per_gas_table_hand_example():
    y = np.array([1, 1, 2, 2, 6, 6]); p = np.array([1, 2, 2, 2, 1, 6]); b = np.ones(6, int)
    t = per_gas_table(y, p, b, [1]).set_index("gas")
    assert t.loc[1, "recall"] == 0.5 and t.loc[1, "precision"] == 0.5 and t.loc[1, "f1"] == pytest.approx(0.5)
    assert t.loc[2, "recall"] == 1.0 and t.loc[2, "precision"] == pytest.approx(2 / 3) and t.loc[2, "f1"] == pytest.approx(0.8)
    assert t.loc[6, "recall"] == 0.5 and t.loc[6, "precision"] == 1.0 and t.loc[6, "support"] == 2
    assert np.isnan(t.loc[3, "recall"]) and np.isnan(t.loc[3, "precision"]) and np.isnan(t.loc[3, "f1"])   # gas absent everywhere


def test_per_gas_f1_equals_sklearn():
    rng = np.random.RandomState(0); y = rng.randint(1, 7, 400); p = np.where(rng.rand(400) < 0.6, y, rng.randint(1, 7, 400)); b = np.ones(400, int)
    t = per_gas_table(y, p, b, [1]).set_index("gas")
    for g in range(1, 7):
        assert t.loc[g, "f1"] == pytest.approx(f1_score(y, p, labels=[g], average="macro", zero_division=0))


@needs_data
def test_augmented_training_set_uses_only_batches_1_to_3_plus_samples_from_batch_6():
    X, y, b = load_all()
    base = augmented_training_indices(b, y, "toluene", 0, seed=0)
    assert set(b[base]) == {1, 2, 3} and len(base) == int((b <= 3).sum())
    for arm in ["toluene", "random"]:
        idx = augmented_training_indices(b, y, arm, 50, seed=1); extra = idx[len(base):]
        assert len(extra) == 50 and set(b[extra]) == {6} and len(set(extra)) == 50
        assert arm != "toluene" or set(y[extra]) == {6}                          # the Toluene arm adds Toluene only
    assert (y[augmented_training_indices(b, y, "random", 200, seed=2)[len(base):]] == 6).mean() < 0.6   # control arm is a natural mix
    a, c = augmented_training_indices(b, y, "toluene", 50, 1), augmented_training_indices(b, y, "toluene", 50, 1)
    assert np.array_equal(a, c) and not np.array_equal(a, augmented_training_indices(b, y, "toluene", 50, 2))   # reproducible, seed-dependent
    assert len(augmented_training_indices(b, y, "toluene", 10 ** 6, 0)) == len(base) + int(((b == 6) & (y == 6)).sum())   # capped at the pool
    with pytest.raises(ValueError):
        augmented_training_indices(b, y, "nonsense", 5, 0)


@needs_data
def test_learning_curve_with_zero_extra_samples_is_the_p1_model_and_reports_all_test_batches():
    X, y, b = load_all()
    rows = learning_curve_task("kNN", KNeighborsClassifier(), "toluene", 0, 0, X, y, b)
    assert [r["test_batch"] for r in rows] == [7, 8, 9, 10] and all(r["toluene_in_train"] == 79 and r["n_train"] == int((b <= 3).sum()) for r in rows)
    rows50 = learning_curve_task("kNN", KNeighborsClassifier(), "toluene", 50, 0, X, y, b)
    assert all(r["toluene_in_train"] == 129 for r in rows50)
