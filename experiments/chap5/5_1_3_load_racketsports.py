"""
Experiment 5.1.3 — Load, explore and visualize RacketSports.

Chapter 5, Section 5.1: Datasets
Dataset: RacketSports (UEA multivariate archive, Bagnall et al. 2018)
         Smartwatch on the wrist of a racket player. Accelerometer +
         gyroscope, 6 channels, 4 racket-stroke classes (badminton clear /
         smash, squash backhand / forehand boast).
Input:  data/processed/racketsports_{X,y}_{train,test}.npy
Output: results/chap5/5_1_3_load_racketsports.csv
        results/chap5/5_1_3_load_racketsports.png
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
NAME = "5_1_3_load_racketsports"

# Labels 0..3 follow the sorted class-string encoding used in 5_0_preprocess.py:
# 0=badminton_clear, 1=badminton_smash, 2=squash_backhandboast, 3=squash_forehandboast.
CLASS_NAMES = ["bad_clear", "bad_smash", "squash_bh", "squash_fh"]
CLASS_LABELS_PLOT = ["badminton\nclear", "badminton\nsmash",
                     "squash\nbackhand", "squash\nforehand"]

# 6 channels overlaid on each subplot.
CHANNEL_NAMES = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
CHANNEL_COLORS = ["red", "blue", "green", "orange", "purple", "brown"]


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load RacketSports train and test splits from data/processed/.

    Returns:
        X_train: Train series, shape (n_train, T, d).
        y_train: Train labels, shape (n_train,).
        X_test:  Test series, shape (n_test, T, d).
        y_test:  Test labels, shape (n_test,).
    """
    X_train = np.load(PROCESSED_DIR / "racketsports_X_train.npy")
    X_test = np.load(PROCESSED_DIR / "racketsports_X_test.npy")
    y_train = np.load(PROCESSED_DIR / "racketsports_y_train.npy")
    y_test = np.load(PROCESSED_DIR / "racketsports_y_test.npy")
    logger.info(
        "Loaded RacketSports: X_train %s, X_test %s", X_train.shape, X_test.shape
    )
    return X_train, y_train, X_test, y_test


def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Build the descriptive-statistics table for RacketSports.

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
        label = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else str(cls)
        rows.append(
            {
                "metric": f"class_{label}_count",
                "train": int(np.sum(y_train == cls)),
                "test": int(np.sum(y_test == cls)),
            }
        )

    results = pd.DataFrame(rows)
    logger.info(
        "RacketSports — n_train=%d, n_test=%d, T=%d, d=%d, n_classes=%d",
        n_train, n_test, T_train, d_train, n_classes,
    )
    logger.info("Dataset statistics:\n%s", results.to_string(index=False))
    return results


def plot_results(X_train: np.ndarray, y_train: np.ndarray) -> plt.Figure:
    """Plot 3 sample time series per class with 6 channels overlaid.

    Args:
        X_train: Train series, shape (n_train, T, 6).
        y_train: Train labels (0..3).

    Returns:
        Matplotlib figure (4 rows × 3 cols).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_samples_per_class = 3
    classes = np.arange(len(CLASS_NAMES))

    fig, axes = plt.subplots(
        len(classes),
        n_samples_per_class,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
    )
    fig.suptitle(
        "RacketSports — sample time series per class (6 dimensions overlaid)",
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
            for dim, (name, color) in enumerate(zip(CHANNEL_NAMES, CHANNEL_COLORS)):
                ax.plot(
                    X_train[idx, :, dim],
                    color=color,
                    linewidth=1.0,
                    label=name if (row_idx == 0 and col_idx == 0) else None,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col_idx == 0:
                ax.set_ylabel(
                    CLASS_LABELS_PLOT[cls], fontsize=9, rotation=90, labelpad=8
                )
            if row_idx == 0:
                ax.set_title(f"sample {col_idx + 1}", fontsize=9)

    handles = [
        plt.Line2D([0], [0], color=color, linewidth=1.5, label=name)
        for name, color in zip(CHANNEL_NAMES, CHANNEL_COLORS)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.90))
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
