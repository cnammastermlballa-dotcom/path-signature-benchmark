"""
Experiment 5.1.4 — Load, explore and visualize NATOPS.

Chapter 5, Section 5.1: Datasets
Dataset: NATOPS Gestures (UEA multivariate archive, Bagnall et al. 2018)
         Naval Air Training and Operating Procedures Standardization gestures.
         8 sensors × 3 coordinates = 24 channels, 6 gesture classes.
Input:  data/processed/natops_{X,y}_{train,test}.npy
Output: results/chap5/5_1_4_load_natops.csv
        results/chap5/5_1_4_load_natops.png
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
NAME = "5_1_4_load_natops"

N_CLASSES = 6
CLASS_LABELS = [f"class {i}" for i in range(N_CLASSES)]

# Only the first sensor's 3 coordinates are plotted to keep the figure readable
# (24 overlaid curves would be visually unusable).
SENSOR1_CHANNELS = [("sensor1-x", 0, "red"),
                    ("sensor1-y", 1, "blue"),
                    ("sensor1-z", 2, "green")]


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load NATOPS train and test splits from data/processed/.

    Returns:
        X_train: Train series, shape (n_train, T, d).
        y_train: Train labels, shape (n_train,).
        X_test:  Test series, shape (n_test, T, d).
        y_test:  Test labels, shape (n_test,).
    """
    X_train = np.load(PROCESSED_DIR / "natops_X_train.npy")
    X_test = np.load(PROCESSED_DIR / "natops_X_test.npy")
    y_train = np.load(PROCESSED_DIR / "natops_y_train.npy")
    y_test = np.load(PROCESSED_DIR / "natops_y_test.npy")
    logger.info(
        "Loaded NATOPS: X_train %s, X_test %s", X_train.shape, X_test.shape
    )
    return X_train, y_train, X_test, y_test


def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Build the descriptive-statistics table for NATOPS.

    Args:
        X_train: Train series, shape (n_train, T, d).
        y_train: Train labels.
        X_test:  Test series.
        y_test:  Test labels.

    Returns:
        DataFrame with columns [metric, train, test]. Rows: n_series, T, d,
        n_classes, and one row per class holding sample counts.
    """
    n_train, T_train, d_train = X_train.shape
    n_test, T_test, d_test = X_test.shape
    assert T_train == T_test and d_train == d_test, "Train/test shape mismatch"
    classes = np.unique(np.concatenate([y_train, y_test]))
    n_classes = len(classes)

    rows = [
        {"metric": "n_series",  "train": n_train, "test": n_test},
        {"metric": "T",         "train": T_train, "test": T_test},
        {"metric": "d",         "train": d_train, "test": d_test},
        {"metric": "n_classes", "train": n_classes, "test": n_classes},
    ]
    for cls in classes:
        rows.append(
            {
                "metric": f"class_{int(cls)}_count",
                "train": int(np.sum(y_train == cls)),
                "test": int(np.sum(y_test == cls)),
            }
        )

    results = pd.DataFrame(rows)
    logger.info(
        "NATOPS — n_train=%d, n_test=%d, T=%d, d=%d, n_classes=%d",
        n_train, n_test, T_train, d_train, n_classes,
    )
    logger.info("Dataset statistics:\n%s", results.to_string(index=False))
    return results


def plot_results(X_train: np.ndarray, y_train: np.ndarray) -> plt.Figure:
    """Plot 2 sample gestures per class with sensor 1's (x, y, z) overlaid.

    Args:
        X_train: Train series, shape (n_train, T, 24).
        y_train: Train labels (0..5).

    Returns:
        Matplotlib figure (6 rows × 2 cols).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_samples_per_class = 2
    classes = np.arange(N_CLASSES)

    fig, axes = plt.subplots(
        len(classes),
        n_samples_per_class,
        figsize=(10, 12),
        sharex=True,
        sharey=True,
    )
    fig.suptitle(
        "NATOPS — sample gestures per class (sensor 1: x, y, z overlaid)",
        fontsize=13, y=0.98,
    )

    for row_idx, cls in enumerate(classes):
        indices = np.where(y_train == cls)[0]
        if len(indices) == 0:
            continue
        chosen = rng.choice(
            indices,
            size=min(n_samples_per_class, len(indices)),
            replace=False,
        )
        for col_idx, idx in enumerate(chosen):
            ax = axes[row_idx, col_idx]
            for name, dim, color in SENSOR1_CHANNELS:
                ax.plot(X_train[idx, :, dim], color=color, linewidth=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col_idx == 0:
                ax.set_ylabel(
                    CLASS_LABELS[cls], fontsize=10, rotation=90, labelpad=8
                )
            if row_idx == 0:
                ax.set_title(f"sample {col_idx + 1}", fontsize=9)

    handles = [
        plt.Line2D([0], [0], color=color, linewidth=1.5, label=name)
        for name, _, color in SENSOR1_CHANNELS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.945),
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


def save_results(results: pd.DataFrame, fig: plt.Figure, name: str) -> None:
    """Save descriptive stats to CSV and figure to PNG.

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
