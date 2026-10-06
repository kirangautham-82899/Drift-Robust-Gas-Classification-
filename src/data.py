"""Data loading for the UCI Gas Sensor Array Drift dataset.

Each batchN.dat file is in LibSVM format: ``label 1:v1 2:v2 ... 128:v128``.
The 128 features are 16 sensors x 8 features per sensor; for sensor ``s``
(0-based) the features occupy columns ``8*s .. 8*s + 7`` in this order:

    0  DR        steady-state response (max resistance change)
    1  |DR|      normalised steady-state response
    2-4 EMA_inc  exponential moving average of the rising edge (alpha = 0.001, 0.01, 0.1)
    5-7 EMA_dec  exponential moving average of the decaying edge (alpha = 0.001, 0.01, 0.1)
"""
from pathlib import Path

import numpy as np
from sklearn.datasets import load_svmlight_file

N_BATCHES = 10
N_FEATURES = 128
N_SENSORS = 16
FEATURES_PER_SENSOR = 8
GAS_NAMES = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene",
}
FEATURE_TYPES = [
    "DR",
    "|DR|",
    "EMA_inc_0.001",
    "EMA_inc_0.01",
    "EMA_inc_0.1",
    "EMA_dec_0.001",
    "EMA_dec_0.01",
    "EMA_dec_0.1",
]

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def feature_names():
    """Return the 128 feature names as ``S01_DR``, ``S01_|DR|``, ..."""
    return [
        f"S{s + 1:02d}_{ft}"
        for s in range(N_SENSORS)
        for ft in FEATURE_TYPES
    ]


def load_batch(batch, path=RAW_DIR):
    """Load one batch (1-10) and return ``(X, y)`` as dense arrays."""
    if not 1 <= batch <= N_BATCHES:
        raise ValueError(f"batch must be in 1..{N_BATCHES}, got {batch}")
    X, y = load_svmlight_file(str(Path(path) / f"batch{batch}.dat"), n_features=N_FEATURES)
    return X.toarray(), y.astype(int)


def load_all(path=RAW_DIR):
    """Load all 10 batches.

    Returns
    -------
    X : (13910, 128) float array
    y : (13910,) int array, gas label 1-6
    batch : (13910,) int array, batch id 1-10 (chronological order)
    """
    Xs, ys, bs = [], [], []
    for b in range(1, N_BATCHES + 1):
        X, y = load_batch(b, path)
        Xs.append(X)
        ys.append(y)
        bs.append(np.full(len(y), b, dtype=int))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(bs)
