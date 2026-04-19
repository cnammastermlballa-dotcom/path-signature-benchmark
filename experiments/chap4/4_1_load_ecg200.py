"""
Experiment 4.1 — Load, explore and visualize ECG200.

Chapter 4: Datasets and Representations
Dataset: ECG200 (UCR Time Series Archive)
Output: results/chap4/4_1_load_ecg200.csv
        results/chap4/4_1_load_ecg200.png
"""

import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
RESULTS_DIR = Path("results/chap4")
DATA_DIR = Path("data/raw/ECG200")
UCR_URL = (
    "https://www.timeseriesclassification.com/aeon-toolkit/ECG200.zip"
)


def download_ecg200() -> None:
    """Download ECG200 from the UCR archive if not already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_file = DATA_DIR / "ECG200_TRAIN.txt"
    if train_file.exists():
        logger.info("ECG200 already downloaded, skipping.")
        return
    zip_path = DATA_DIR / "ECG200.zip"
    logger.info("Downloading ECG200...")
    urllib.request.urlretrieve(UCR_URL, zip_path)
    logger.info("Extracting archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()
    logger.info("ECG200 downloaded and extracted.")


def _parse_ucr_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a UCR .txt file (label in first column).

    Args:
        path: Path to the UCR text file.

    Returns:
        X: Time series array of shape (n_samples, length).
        y: Label array of shape (n_samples,).
    """
    data = np.loadtxt(path)
    y = data[:, 0].astype(int)
    X = data[:, 1:]
    return X, y


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ECG200 train and test splits.

    Returns:
        X_train: Train time series, shape (100, 96).
        y_train: Train labels, shape (100,).
        X_test:  Test time series, shape (100, 96).
        y_test:  Test labels, shape (100,).
    """
    download_ecg200()
    # Locate files regardless of nesting introduced by zip extraction
    train_candidates = list(DATA_DIR.rglob("ECG200_TRAIN.txt")) + list(
        DATA_DIR.rglob("ECG200_TRAIN.tsv")
    )
    test_candidates = list(DATA_DIR.rglob("ECG200_TEST.txt")) + list(
        DATA_DIR.rglob("ECG200_TEST.tsv")
    )
    if not train_candidates or not test_candidates:
        raise FileNotFoundError(
            f"Could not find ECG200 data files under {DATA_DIR}. "
            "Check the download or extract manually."
        )
    X_train, y_train = _parse_ucr_file(train_candidates[0])
    X_test, y_test = _parse_ucr_file(test_candidates[0])
    logger.info(
        "Loaded ECG200: train %s, test %s", X_train.shape, X_test.shape
    )
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
        DataFrame with columns [split, metric, value].
    """
    classes = np.unique(np.concatenate([y_train, y_test]))
    rows = []

    for split_name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        rows.append({"split": split_name, "metric": "n_samples", "value": len(y)})
        rows.append({"split": split_name, "metric": "series_length", "value": X.shape[1]})
        rows.append({"split": split_name, "metric": "n_classes", "value": len(classes)})
        for cls in classes:
            count = int(np.sum(y == cls))
            rows.append(
                {
                    "split": split_name,
                    "metric": f"class_{cls}_count",
                    "value": count,
                }
            )
            rows.append(
                {
                    "split": split_name,
                    "metric": f"class_{cls}_pct",
                    "value": round(100 * count / len(y), 2),
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
    fig.suptitle("ECG200 — sample time series per class", fontsize=14, y=1.01)

    colors = {classes[0]: "#d62728", classes[1]: "#1f77b4"}
    label_map = {classes[0]: "Abnormal (−1)", classes[1]: "Normal (+1)"}

    for row_idx, cls in enumerate(classes):
        indices = np.where(y_train == cls)[0]
        chosen = rng.choice(indices, size=n_samples_per_class, replace=False)
        for col_idx, idx in enumerate(chosen):
            ax = axes[row_idx, col_idx]
            ax.plot(X_train[idx], color=colors[cls], linewidth=1.2)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(label_map[cls], fontsize=9)
            if row_idx == 0:
                ax.set_title(f"sample {col_idx + 1}", fontsize=8)

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
    save_results(results, fig, "4_1_load_ecg200")
    plt.close(fig)
    logger.info("Done.")


if __name__ == "__main__":
    main()
