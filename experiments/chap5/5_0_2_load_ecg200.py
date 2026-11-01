"""
Experiment 5.0.2 — Load, explore and visualize ECG200.

Chapter 5, Section 5.1: Datasets
Dataset: ECG200 (UCR Time Series Archive). Univariate heartbeats (T=96),
         2 classes: 0 = abnormal (raw label -1), 1 = normal (raw label 1).
Input:  data/processed/ecg200_{X,y}_{train,test}.npy
Output: results/chap5/5_0_2_load_ecg200.csv
        results/chap5/5_0_2_load_ecg200.png
"""

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/chap5")
NAME = "5_0_2_load_ecg200"

# Labels 0..1 follow the encoding of 5_0_1_preprocess.py ('-1' → 0, '1' → 1).
CLASS_LABELS_PLOT = ["Anormal (−1)", "Normal (+1)"]
CLASS_COLORS = ["#d62728", "#1f77b4"]


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ECG200 train and test splits from data/processed/.

    Returns:
        X_train: Train time series, shape (100, 96).
        y_train: Train labels (0..1), shape (100,).
        X_test:  Test time series, shape (100, 96).
        y_test:  Test labels (0..1), shape (100,).
    """
    X_train = np.load(PROCESSED_DIR / "ecg200_X_train.npy")
    X_test = np.load(PROCESSED_DIR / "ecg200_X_test.npy")
    y_train = np.load(PROCESSED_DIR / "ecg200_y_train.npy")
    y_test = np.load(PROCESSED_DIR / "ecg200_y_test.npy")
    logger.info("Loaded ECG200: X_train %s, X_test %s", X_train.shape, X_test.shape)
    return X_train, y_train, X_test, y_test


def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Compute dataset statistics and class balance.

    Args:
        X_train: Train time series.
        y_train: Train labels.
        X_test:  Test time series.
        y_test:  Test labels.

    Returns:
        DataFrame with columns [metric, train, test]. Rows: n_series, T, d,
        n_classes, and one row per class holding sample counts.
    """
    classes = np.unique(np.concatenate([y_train, y_test]))
    n_classes = len(classes)

    rows = [
        {"metric": "n_series",  "train": len(y_train), "test": len(y_test)},
        {"metric": "T",         "train": X_train.shape[1], "test": X_test.shape[1]},
        {"metric": "d",         "train": 1,            "test": 1},
        {"metric": "n_classes", "train": n_classes,    "test": n_classes},
    ]
    for cls in classes:
        rows.append(
            {
                "metric": f"class_{cls}_count",
                "train":  int(np.sum(y_train == cls)),
                "test":   int(np.sum(y_test == cls)),
            }
        )

    results = pd.DataFrame(rows)
    logger.info("Dataset statistics:\n%s", results.to_string(index=False))
    return results


def plot_results(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> plt.Figure:
    """Plot sample time series per class.

    Args:
        X_train: Train time series, shape (n_samples, length).
        y_train: Train labels.

    Returns:
        Matplotlib figure.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    classes = np.unique(y_train)
    n_samples_per_class = 5
    n_cols = n_samples_per_class
    n_rows = len(classes)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3, n_rows * 2.5),
        sharex=True,
        sharey=True,
    )
    fig.suptitle("ECG200 — exemples de séries temporelles par classe", fontsize=14, y=1.01)

    for row_idx, cls in enumerate(classes):
        indices = np.where(y_train == cls)[0]
        chosen = rng.choice(indices, size=n_samples_per_class, replace=False)
        for col_idx, idx in enumerate(chosen):
            ax = axes[row_idx, col_idx]
            ax.plot(X_train[idx], color=CLASS_COLORS[cls], linewidth=1.2)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(CLASS_LABELS_PLOT[cls], fontsize=9)
            if row_idx == 0:
                ax.set_title(f"exemple {col_idx + 1}", fontsize=8)

    fig.tight_layout()
    return fig


def save_results(results: pd.DataFrame, fig: plt.Figure, name: str) -> None:
    """Save results to CSV and PNG.

    Args:
        results: DataFrame to save.
        fig: Matplotlib figure to save.
        name: Base filename without extension.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{name}.csv"
    png_path = RESULTS_DIR / f"{name}.png"
    results.to_csv(csv_path, index=False)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    logger.info("Results saved to %s and %s", csv_path, png_path)


def main() -> None:
    """Run full experiment pipeline."""
    logger.info("Loading data...")
    X_train, y_train, X_test, y_test = load_data()

    logger.info("Running experiment...")
    results = run_experiment(X_train, y_train, X_test, y_test)

    logger.info("Plotting results...")
    fig = plot_results(X_train, y_train)

    logger.info("Saving results...")
    save_results(results, fig, NAME)
    plt.close(fig)
    logger.info("Done.")


if __name__ == "__main__":
    main()
