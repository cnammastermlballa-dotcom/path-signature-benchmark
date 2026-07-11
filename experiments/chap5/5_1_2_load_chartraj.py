"""
Experiment 5.1.2 — Load, explore and visualize CharacterTrajectories.

Chapter 5, Section 5.1: Datasets
Dataset: CharacterTrajectories (UEA multivariate archive via sktime).
         Pen trajectories with 3 channels (x-velocity, y-velocity, pen force),
         20 character classes, variable-length series.
Input:  data/processed/chartraj_{X,y}_{train,test}.npy
Output: results/chap5/5_1_2_load_chartraj.csv
        results/chap5/5_1_2_load_chartraj.png
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
NAME = "5_1_2_load_chartraj"

# 3 channels: x velocity, y velocity, pen tip force.
N_CHANNELS = 3
CHANNEL_NAMES = ["x-vel", "y-vel", "pen-force"]

# UCI CharacterTrajectories 20-character subset (excludes f, i, j, k, t, x —
# characters that cannot be written without lifting the pen). Ordered to
# match the preprocessor's sorted string→int label encoding (class 0 = 'a').
CLASS_NAMES = list("abcdeghlmnopqrsuvwyz")


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CharacterTrajectories train and test splits from data/processed/.

    Returns:
        X_train: Object array of n_train variable-length (T_i, 3) arrays.
        y_train: Train labels, shape (n_train,).
        X_test:  Object array of n_test variable-length (T_i, 3) arrays.
        y_test:  Test labels, shape (n_test,).
    """
    X_train = np.load(PROCESSED_DIR / "chartraj_X_train.npy", allow_pickle=True)
    X_test = np.load(PROCESSED_DIR / "chartraj_X_test.npy", allow_pickle=True)
    y_train = np.load(PROCESSED_DIR / "chartraj_y_train.npy")
    y_test = np.load(PROCESSED_DIR / "chartraj_y_test.npy")
    logger.info(
        "Loaded CharTraj: n_train=%d, n_test=%d, d=%d",
        len(X_train), len(X_test), X_train[0].shape[1],
    )
    return X_train, y_train, X_test, y_test


def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Build the descriptive-statistics table for CharacterTrajectories.

    Args:
        X_train: Object array of n_train (T_i, 3) arrays.
        y_train: Train labels.
        X_test:  Object array of n_test (T_i, 3) arrays.
        y_test:  Test labels.

    Returns:
        DataFrame with columns [metric, train, test]. Rows: n_series, d,
        n_classes, T_min/max/mean, and one row per class with sample counts.
    """
    len_train = np.array([x.shape[0] for x in X_train])
    len_test = np.array([x.shape[0] for x in X_test])
    classes = np.unique(np.concatenate([y_train, y_test]))
    n_classes = len(classes)
    d = X_train[0].shape[1]

    rows = [
        {"metric": "n_series",  "train": len(X_train),         "test": len(X_test)},
        {"metric": "T_min",     "train": int(len_train.min()), "test": int(len_test.min())},
        {"metric": "T_max",     "train": int(len_train.max()), "test": int(len_test.max())},
        {"metric": "T_mean",    "train": round(float(len_train.mean()), 2),
                                "test":  round(float(len_test.mean()),  2)},
        {"metric": "d",         "train": d,                    "test": d},
        {"metric": "n_classes", "train": n_classes,            "test": n_classes},
    ]
    for cls in classes:
        label = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else str(cls)
        rows.append(
            {
                "metric": f"class_{label}_count",
                "train":  int(np.sum(y_train == cls)),
                "test":   int(np.sum(y_test == cls)),
            }
        )

    results = pd.DataFrame(rows)
    logger.info(
        "CharTraj — n_train=%d, n_test=%d, T=[%d, %d], d=%d, n_classes=%d",
        len(X_train), len(X_test),
        int(min(len_train.min(), len_test.min())),
        int(max(len_train.max(), len_test.max())),
        d, n_classes,
    )
    logger.info("Dataset statistics:\n%s", results.to_string(index=False))
    return results


def plot_results(X_train: np.ndarray, y_train: np.ndarray) -> plt.Figure:
    """Plot 3 sample 2-D pen paths (x-vel vs y-vel) per class from train.

    Args:
        X_train: Object array of (T_i, 3) arrays.
        y_train: Train labels (0..19).

    Returns:
        Matplotlib figure (20 rows × 3 cols).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_classes = len(CLASS_NAMES)
    n_per_class = 3
    cmap = plt.colormaps["tab20"]
    colors = {i: cmap(i / n_classes) for i in range(n_classes)}

    fig, axes = plt.subplots(
        n_classes, n_per_class,
        figsize=(n_per_class * 2.2, n_classes * 2.0),
        squeeze=False,
    )
    fig.suptitle(
        "CharacterTrajectories — 2-D pen path samples per class\n"
        "(x-velocity vs y-velocity)",
        fontsize=13, y=1.005,
    )

    for row_idx in range(n_classes):
        indices = np.where(y_train == row_idx)[0]
        if len(indices) == 0:
            continue
        chosen = rng.choice(
            indices,
            size=min(n_per_class, len(indices)),
            replace=False,
        )
        for col_idx, idx in enumerate(chosen):
            ax = axes[row_idx, col_idx]
            traj = X_train[idx]
            ax.plot(traj[:, 0], traj[:, 1], color=colors[row_idx], linewidth=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            if col_idx == 0:
                ax.set_ylabel(
                    f"'{CLASS_NAMES[row_idx]}'",
                    fontsize=8, rotation=0, labelpad=22,
                )
            if row_idx == 0:
                ax.set_title(f"sample {col_idx + 1}", fontsize=8)

    fig.tight_layout()
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
