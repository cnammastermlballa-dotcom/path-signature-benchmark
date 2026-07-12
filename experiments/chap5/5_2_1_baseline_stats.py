"""
Experiment 5.2.1 — Baseline statistical features + LR (L1/L2).

Chapter 5, Axe 1: Signature as representation.
Establishes a naive baseline that discards temporal structure: per-channel
summary statistics (mean, std, min, max) computed on the raw series.
Two linear models are then fit with 5-fold CV hyperparameter selection:
LR L2 (solver=lbfgs) and LR L1 / LASSO (solver=liblinear).

Datasets: ECG200, RacketSports, CharacterTrajectories
Metrics : accuracy, F1 macro, feature dimension, fit+predict time (s)

Input : data/processed/{ecg200,racketsports,chartraj}_{X,y}_{train,test}.npy
Output: results/chap5/5_2_1_baseline_stats.csv
        results/chap5/5_2_1_baseline_stats.png
"""

import logging
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# liblinear + multiclass is deprecated in sklearn but chosen intentionally
# for LASSO (CLAUDE.md, Axe 1&2 conventions). Silence the noisy FutureWarning.
warnings.filterwarnings(
    "ignore",
    message="Using the 'liblinear' solver for multiclass",
    category=FutureWarning,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/chap5")
NAME = "5_2_1_baseline_stats"

DATASETS = ["ecg200", "racketsports", "chartraj", "natops"]
DISPLAY_NAMES = {
    "ecg200": "ECG200",
    "racketsports": "RacketSports",
    "chartraj": "CharTraj",
    "natops": "NATOPS",
}
MODELS = [("LR-L2", "l2", "lbfgs"), ("LR-L1", "l1", "liblinear")]
CS = np.logspace(-3, 3, 7)


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one dataset's train/test splits from data/processed/.

    Args:
        name: Dataset key (matches file prefix, e.g. 'ecg200').

    Returns:
        X_train, y_train, X_test, y_test.
    """
    allow = (name == "chartraj")  # variable-length list stored as object array
    X_train = np.load(PROCESSED_DIR / f"{name}_X_train.npy", allow_pickle=allow)
    X_test = np.load(PROCESSED_DIR / f"{name}_X_test.npy", allow_pickle=allow)
    y_train = np.load(PROCESSED_DIR / f"{name}_y_train.npy")
    y_test = np.load(PROCESSED_DIR / f"{name}_y_test.npy")
    return X_train, y_train, X_test, y_test


def _to_series_list(X: np.ndarray) -> List[np.ndarray]:
    """Normalise input into a list of 2-D (T_i, d) arrays."""
    if isinstance(X, np.ndarray) and X.dtype == object:
        return list(X)
    if X.ndim == 2:
        return [X[i, :, None] for i in range(X.shape[0])]
    return [X[i] for i in range(X.shape[0])]


def stats_features(X: np.ndarray) -> np.ndarray:
    """Extract [mean, std, min, max] per channel for each series.

    Args:
        X: (n, T), (n, T, d), or object array of (T_i, d) arrays.

    Returns:
        Feature matrix of shape (n, 4 * d).
    """
    series = _to_series_list(X)
    d = series[0].shape[1]
    F = np.empty((len(series), 4 * d), dtype=np.float64)
    for i, x in enumerate(series):
        F[i, 0:d]       = x.mean(axis=0)
        F[i, d:2*d]     = x.std(axis=0)
        F[i, 2*d:3*d]   = x.min(axis=0)
        F[i, 3*d:4*d]   = x.max(axis=0)
    return F


def fit_evaluate(
    X_train_f: np.ndarray,
    y_train: np.ndarray,
    X_test_f: np.ndarray,
    y_test: np.ndarray,
    penalty: str,
    solver: str,
) -> dict:
    """Fit LogisticRegressionCV and return test metrics.

    Args:
        X_train_f: Feature matrix, shape (n_train, p).
        y_train: Train labels.
        X_test_f: Feature matrix, shape (n_test, p).
        y_test: Test labels.
        penalty: 'l1' or 'l2'.
        solver: 'liblinear' (l1) or 'lbfgs' (l2).

    Returns:
        Dict with accuracy, f1_macro, feature_dim, time_s, C_best.
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train_f)
    Xte = scaler.transform(X_test_f)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = LogisticRegressionCV(
        Cs=CS,
        cv=cv,
        penalty=penalty,
        solver=solver,
        scoring="accuracy",
        max_iter=5000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    elapsed = time.perf_counter() - t0
    return {
        "accuracy":    accuracy_score(y_test, y_pred),
        "f1_macro":    f1_score(y_test, y_pred, average="macro"),
        "feature_dim": X_train_f.shape[1],
        "time_s":      elapsed,
        "C_best":      float(np.mean(model.C_)),
    }


def run_experiment() -> pd.DataFrame:
    """Run baseline evaluation across all datasets × models.

    Returns:
        Long-format DataFrame, one row per (dataset, model).
    """
    rows = []
    for ds in DATASETS:
        logger.info("--- %s ---", DISPLAY_NAMES[ds])
        X_train, y_train, X_test, y_test = load_dataset(ds)
        X_train_f = stats_features(X_train)
        X_test_f = stats_features(X_test)
        logger.info(
            "  features: train %s, test %s", X_train_f.shape, X_test_f.shape
        )
        for name, penalty, solver in MODELS:
            metrics = fit_evaluate(
                X_train_f, y_train, X_test_f, y_test, penalty, solver
            )
            metrics["dataset"] = DISPLAY_NAMES[ds]
            metrics["model"] = name
            rows.append(metrics)
            logger.info(
                "  %s: acc=%.3f  f1=%.3f  time=%.2fs  C*=%.3g",
                name, metrics["accuracy"], metrics["f1_macro"],
                metrics["time_s"], metrics["C_best"],
            )
    df = pd.DataFrame(rows)[
        ["dataset", "model", "accuracy", "f1_macro",
         "feature_dim", "time_s", "C_best"]
    ]
    return df


def plot_results(df: pd.DataFrame) -> plt.Figure:
    """2x2 grouped-bar figure, one subplot per metric.

    Args:
        df: DataFrame from run_experiment.

    Returns:
        Matplotlib figure.
    """
    metrics = [
        ("accuracy",    "Accuracy",         "%.3f"),
        ("f1_macro",    "F1 (macro)",       "%.3f"),
        ("feature_dim", "Feature dimension", "%d"),
        ("time_s",      "Fit + predict (s)", "%.2f"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "Baseline (mean / std / min / max per channel) — LR L2 vs L1",
        fontsize=13, y=0.99,
    )

    datasets = [DISPLAY_NAMES[d] for d in DATASETS]
    x = np.arange(len(datasets))
    width = 0.38
    colors = {"LR-L2": "#1f77b4", "LR-L1": "#ff7f0e"}
    model_names = [m[0] for m in MODELS]

    for ax, (col, title, fmt) in zip(axes.ravel(), metrics):
        for i, model in enumerate(model_names):
            values = [
                df[(df.dataset == d) & (df.model == model)][col].iloc[0]
                for d in datasets
            ]
            offset = (i - 0.5) * width
            bars = ax.bar(
                x + offset, values, width, label=model, color=colors[model]
            )
            for bar, v in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    fmt % v,
                    ha="center", va="bottom", fontsize=8,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
    axes[0, 0].legend(loc="lower right", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def save_results(df: pd.DataFrame, fig: plt.Figure, name: str) -> None:
    """Save results DataFrame to CSV and figure to PNG."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{name}.csv"
    png_path = RESULTS_DIR / f"{name}.png"
    df.to_csv(csv_path, index=False)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    logger.info("Results saved to %s and %s", csv_path, png_path)


def main() -> None:
    """Run full baseline experiment."""
    logger.info("Running baseline stats experiment...")
    df = run_experiment()
    logger.info("Summary:\n%s", df.to_string(index=False))
    logger.info("Plotting...")
    fig = plot_results(df)
    save_results(df, fig, NAME)
    plt.close(fig)
    logger.info("Done.")


if __name__ == "__main__":
    main()
