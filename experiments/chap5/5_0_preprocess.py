"""
Experiment 5.0 — Preprocessing: load raw datasets and save to data/processed/.

This script is the single entry point for all Chapter 5 scripts.
No model fitting, no visualisation — preprocessing only.

Label encoding (documented for confusion matrices in chap5):
  ECG200            : '-1' → 0  (abnormal),  '1' → 1  (normal)
  CharacterTrajector: '1'..'20' → 0..19  (sorted lexicographic → numeric order)
  RacketSports      : sorted class strings → 0..3
  NATOPS            : '1.0'..'6.0' → 0..5

Chapter 5: Datasets and Representations
Output: data/processed/
          ecg200_X_train.npy         shape (100, 96)
          ecg200_X_test.npy          shape (100, 96)
          ecg200_y_train.npy         shape (100,)
          ecg200_y_test.npy          shape (100,)
          chartraj_X_train.npy       list of arrays (variable length, d=3)
          chartraj_X_test.npy        list of arrays (variable length, d=3)
          chartraj_y_train.npy       shape (n_train,)
          chartraj_y_test.npy        shape (n_test,)
          racketsports_X_train.npy   shape (151, 30, 6)
          racketsports_X_test.npy    shape (152, 30, 6)
          racketsports_y_train.npy   shape (151,)
          racketsports_y_test.npy    shape (152,)
          natops_X_train.npy         shape (180, 51, 24)
          natops_X_test.npy          shape (180, 51, 24)
          natops_y_train.npy         shape (180,)
          natops_y_test.npy          shape (180,)
"""

import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sktime.datasets import load_UCR_UEA_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# ECG200
# ---------------------------------------------------------------------------

def load_ecg200(split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load one split of ECG200 via sktime and convert to numpy.

    Args:
        split: 'train' or 'test'.

    Returns:
        X: Float array of shape (n_samples, 96).
        y: Integer label array of shape (n_samples,). '-1' → 0, '1' → 1.
    """
    X_df, y_raw = load_UCR_UEA_dataset("ECG200", split=split, return_X_y=True)
    n = len(X_df)
    X = np.array([X_df.iloc[i, 0].to_numpy() for i in range(n)], dtype=np.float64)
    label_map = {"-1": 0, "1": 1}
    y = np.array([label_map[str(v)] for v in y_raw], dtype=np.int32)
    return X, y


def preprocess_ecg200() -> None:
    """Load, verify and save both ECG200 splits."""
    logger.info("Loading ECG200 train...")
    X_train, y_train = load_ecg200("train")
    logger.info("Loading ECG200 test...")
    X_test, y_test = load_ecg200("test")

    assert X_train.shape == (100, 96), f"Unexpected shape: {X_train.shape}"
    assert X_test.shape == (100, 96),  f"Unexpected shape: {X_test.shape}"
    assert not np.isnan(X_train).any(), "NaN in ECG200 train"
    assert not np.isnan(X_test).any(),  "NaN in ECG200 test"
    assert set(np.unique(y_train)) == {0, 1}
    assert set(np.unique(y_test))  == {0, 1}

    np.save(PROCESSED_DIR / "ecg200_X_train.npy", X_train)
    np.save(PROCESSED_DIR / "ecg200_X_test.npy",  X_test)
    np.save(PROCESSED_DIR / "ecg200_y_train.npy", y_train)
    np.save(PROCESSED_DIR / "ecg200_y_test.npy",  y_test)
    logger.info("ECG200 saved — X_train %s, X_test %s", X_train.shape, X_test.shape)


# ---------------------------------------------------------------------------
# CharacterTrajectories
# ---------------------------------------------------------------------------

def _df_to_series_list(X_df: pd.DataFrame) -> List[np.ndarray]:
    """Convert sktime nested DataFrame to a list of (length_i, d) arrays.

    Args:
        X_df: sktime nested DataFrame of shape (n_samples, d).

    Returns:
        List of n_samples arrays, each of shape (length_i, d).
    """
    n, d = X_df.shape
    series = []
    for i in range(n):
        cols = [X_df.iloc[i, dim].to_numpy() for dim in range(d)]
        series.append(np.stack(cols, axis=1).astype(np.float64))
    return series


def load_chartraj(split: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load one split of CharacterTrajectories via sktime.

    Args:
        split: 'train' or 'test'.

    Returns:
        X: List of n_samples arrays of shape (length_i, 3).
        y: Integer label array of shape (n_samples,). Classes '1'..'20' → 0..19.
    """
    X_df, y_raw = load_UCR_UEA_dataset(
        "CharacterTrajectories", split=split, return_X_y=True
    )
    X = _df_to_series_list(X_df)
    classes = sorted(set(str(v) for v in y_raw), key=lambda s: int(s))
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    y = np.array([label_map[str(v)] for v in y_raw], dtype=np.int32)
    return X, y


def preprocess_chartraj() -> None:
    """Load, verify and save both CharacterTrajectories splits."""
    logger.info("Loading CharacterTrajectories train...")
    X_train, y_train = load_chartraj("train")
    logger.info("Loading CharacterTrajectories test...")
    X_test, y_test = load_chartraj("test")

    assert all(x.shape[1] == 3 for x in X_train), "Expected d=3 for all train series"
    assert all(x.shape[1] == 3 for x in X_test),  "Expected d=3 for all test series"
    assert len(np.unique(y_train)) == 20, "Expected 20 classes in train"
    assert len(np.unique(y_test))  == 20, "Expected 20 classes in test"

    np.save(PROCESSED_DIR / "chartraj_X_train.npy", np.array(X_train, dtype=object),
            allow_pickle=True)
    np.save(PROCESSED_DIR / "chartraj_X_test.npy",  np.array(X_test, dtype=object),
            allow_pickle=True)
    np.save(PROCESSED_DIR / "chartraj_y_train.npy", y_train)
    np.save(PROCESSED_DIR / "chartraj_y_test.npy",  y_test)

    lengths = [x.shape[0] for x in X_train]
    logger.info(
        "CharTraj saved — train: %d series, lengths [%d, %d]; test: %d series",
        len(X_train), min(lengths), max(lengths), len(X_test),
    )


# ---------------------------------------------------------------------------
# UEA fixed-length loader (BasicMotions, NATOPS)
# ---------------------------------------------------------------------------

def load_uea_fixed(name: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load one split of a fixed-length UEA multivariate dataset via sktime.

    Args:
        name: UEA dataset name (e.g. 'BasicMotions', 'NATOPS').
        split: 'train' or 'test'.

    Returns:
        X: Float array of shape (n_samples, T, d).
        y: Integer label array of shape (n_samples,). Classes sorted → 0..K-1.
    """
    X_df, y_raw = load_UCR_UEA_dataset(name, split=split, return_X_y=True)
    n, d = X_df.shape
    T = len(X_df.iloc[0, 0])
    X = np.empty((n, T, d), dtype=np.float64)
    for i in range(n):
        for dim in range(d):
            X[i, :, dim] = X_df.iloc[i, dim].to_numpy()
    classes = sorted(str(v) for v in set(y_raw))
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    y = np.array([label_map[str(v)] for v in y_raw], dtype=np.int32)
    logger.info("  %s %s class map: %s", name, split, label_map)
    return X, y


def preprocess_racketsports() -> None:
    """Load, verify and save both RacketSports splits (T=30, d=6, 4 classes)."""
    logger.info("Loading RacketSports train...")
    X_train, y_train = load_uea_fixed("RacketSports", "train")
    logger.info("Loading RacketSports test...")
    X_test, y_test = load_uea_fixed("RacketSports", "test")

    assert X_train.ndim == 3 and X_train.shape[1] == 30, \
        f"Expected shape (n, 30, d), got {X_train.shape}"
    assert X_test.ndim == 3 and X_test.shape[1] == 30, \
        f"Expected shape (n, 30, d), got {X_test.shape}"
    assert X_train.shape[2] == X_test.shape[2], "d mismatch train/test"
    assert not np.isnan(X_train).any(), "NaN in RacketSports train"
    assert not np.isnan(X_test).any(),  "NaN in RacketSports test"
    assert len(np.unique(y_train)) == 4, "Expected 4 classes in train"
    assert len(np.unique(y_test))  == 4, "Expected 4 classes in test"

    np.save(PROCESSED_DIR / "racketsports_X_train.npy", X_train)
    np.save(PROCESSED_DIR / "racketsports_X_test.npy",  X_test)
    np.save(PROCESSED_DIR / "racketsports_y_train.npy", y_train)
    np.save(PROCESSED_DIR / "racketsports_y_test.npy",  y_test)
    logger.info(
        "RacketSports saved — X_train %s, X_test %s", X_train.shape, X_test.shape
    )


def preprocess_natops() -> None:
    """Load, verify and save both NATOPS splits (T=51, d=24, 6 classes)."""
    logger.info("Loading NATOPS train...")
    X_train, y_train = load_uea_fixed("NATOPS", "train")
    logger.info("Loading NATOPS test...")
    X_test, y_test = load_uea_fixed("NATOPS", "test")

    assert X_train.ndim == 3 and X_train.shape[1] == 51, \
        f"Expected shape (n, 51, d), got {X_train.shape}"
    assert X_test.ndim == 3 and X_test.shape[1] == 51, \
        f"Expected shape (n, 51, d), got {X_test.shape}"
    assert X_train.shape[2] == X_test.shape[2], "d mismatch train/test"
    assert not np.isnan(X_train).any(), "NaN in NATOPS train"
    assert not np.isnan(X_test).any(),  "NaN in NATOPS test"
    assert len(np.unique(y_train)) == 6, "Expected 6 classes in train"
    assert len(np.unique(y_test))  == 6, "Expected 6 classes in test"

    np.save(PROCESSED_DIR / "natops_X_train.npy", X_train)
    np.save(PROCESSED_DIR / "natops_X_test.npy",  X_test)
    np.save(PROCESSED_DIR / "natops_y_train.npy", y_train)
    np.save(PROCESSED_DIR / "natops_y_test.npy",  y_test)
    logger.info("NATOPS saved — X_train %s, X_test %s", X_train.shape, X_test.shape)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate() -> None:
    """Reload saved files and print a final summary."""
    logger.info("Validating saved files...")

    ecg_Xtr = np.load(PROCESSED_DIR / "ecg200_X_train.npy")
    ecg_Xte = np.load(PROCESSED_DIR / "ecg200_X_test.npy")
    ecg_ytr = np.load(PROCESSED_DIR / "ecg200_y_train.npy")

    ct_Xtr = np.load(PROCESSED_DIR / "chartraj_X_train.npy", allow_pickle=True)
    ct_Xte = np.load(PROCESSED_DIR / "chartraj_X_test.npy",  allow_pickle=True)
    ct_ytr = np.load(PROCESSED_DIR / "chartraj_y_train.npy")

    rs_Xtr = np.load(PROCESSED_DIR / "racketsports_X_train.npy")
    rs_Xte = np.load(PROCESSED_DIR / "racketsports_X_test.npy")
    rs_ytr = np.load(PROCESSED_DIR / "racketsports_y_train.npy")

    na_Xtr = np.load(PROCESSED_DIR / "natops_X_train.npy")
    na_Xte = np.load(PROCESSED_DIR / "natops_X_test.npy")
    na_ytr = np.load(PROCESSED_DIR / "natops_y_train.npy")

    ct_lengths_tr = [x.shape[0] for x in ct_Xtr]

    print()
    print("=== Preprocessing summary ===")
    print(
        f"Dataset ECG200: n_train={ecg_Xtr.shape[0]}, n_test={ecg_Xte.shape[0]}, "
        f"T={ecg_Xtr.shape[1]}, d=1, n_classes={len(np.unique(ecg_ytr))}"
    )
    print(
        f"Dataset CharacterTrajectories: n_train={len(ct_Xtr)}, n_test={len(ct_Xte)}, "
        f"T=[{min(ct_lengths_tr)}, {max(ct_lengths_tr)}] (variable), "
        f"d={ct_Xtr[0].shape[1]}, n_classes={len(np.unique(ct_ytr))}"
    )
    print(
        f"Dataset RacketSports: n_train={rs_Xtr.shape[0]}, n_test={rs_Xte.shape[0]}, "
        f"T={rs_Xtr.shape[1]}, d={rs_Xtr.shape[2]}, "
        f"n_classes={len(np.unique(rs_ytr))}"
    )
    print(
        f"Dataset NATOPS: n_train={na_Xtr.shape[0]}, n_test={na_Xte.shape[0]}, "
        f"T={na_Xtr.shape[1]}, d={na_Xtr.shape[2]}, "
        f"n_classes={len(np.unique(na_ytr))}"
    )
    print("==============================")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run full preprocessing pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("--- ECG200 ---")
    preprocess_ecg200()

    logger.info("--- CharacterTrajectories ---")
    preprocess_chartraj()

    logger.info("--- RacketSports ---")
    preprocess_racketsports()

    logger.info("--- NATOPS ---")
    preprocess_natops()

    logger.info("--- Validation ---")
    validate()

    logger.info("Done. All files saved to %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
