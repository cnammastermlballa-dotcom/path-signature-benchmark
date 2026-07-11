"""
Experiment 5.2.2 — Truncated signature (lin+time) + LR (L1/L2), N=1..10.

Chapter 5, Axe 1: Signature as representation.
Pipeline per dataset:
    1. Prepend a normalized-time channel t ∈ [0, 1] to each series
       (canonical "lin+time" embedding; piecewise-linear interpolation
       between consecutive samples is implicit in the signature backend).
    2. Compute truncated signatures at depth N = 1..10 via iisignature
       (esig backend errors on CharTraj; see comment near the import).
       Capped when the signature dimension would exceed MAX_FEATURE_DIM.
    3. Fit LR L2 (lbfgs) and LR L1 / LASSO (liblinear) with 5-fold
       LogisticRegressionCV hyperparameter selection.

For each dataset, N* is the truncation depth with the highest CV
accuracy across the two models. It is used downstream in Axe 2 (5_3).

Datasets: ECG200, RacketSports, CharacterTrajectories
Metrics : accuracy, F1 macro, feature dimension, CV score, time (s)

Input : data/processed/{ecg200,racketsports,chartraj,natops}_{X,y}_{train,test}.npy
Output: results/chap5/5_2_2_signature_lr.csv
        results/chap5/5_2_2_signature_lr.png
        results/chap5/optimal_N_{ecg200,racketsports,chartraj,natops}.txt
"""

import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# NOTE: CLAUDE.md declares `esig` as the default signature library, but its
# RoughPy backend errors on the CharTraj batch at depth ≥ 2 ("mismatch between
# number of rows in data and number of indices"). iisignature is used here as
# a drop-in replacement; its `sig(path, N)` returns sum_{k=1..N} d^k features
# in the same tensor-basis order and excludes the level-0 constant.
import iisignature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# liblinear + multiclass is deprecated but chosen intentionally for LASSO
# (CLAUDE.md, Axe 1&2 conventions). Silence the noisy FutureWarning.
warnings.filterwarnings(
    "ignore",
    message="Using the 'liblinear' solver for multiclass",
    category=FutureWarning,
)

RANDOM_STATE = 42
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/chap5")
NAME = "5_2_2_signature_lr"

DATASETS = ["ecg200", "racketsports", "chartraj", "natops"]
DISPLAY_NAMES = {
    "ecg200":       "ECG200",
    "racketsports": "RacketSports",
    "chartraj":     "CharTraj",
    "natops":       "NATOPS",
}
MODELS = [("LR-L2", "l2", "lbfgs"), ("LR-L1", "l1", "liblinear")]
CS = np.logspace(-3, 3, 7)
N_RANGE = list(range(1, 11))
MAX_FEATURE_DIM = 50_000


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one dataset's train/test splits from data/processed/."""
    allow = (name == "chartraj")
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


def add_time_channel(x: np.ndarray) -> np.ndarray:
    """Prepend a normalized-time channel to a (T, d) series → (T, d+1)."""
    T = x.shape[0]
    t = np.linspace(0.0, 1.0, T, dtype=np.float64).reshape(-1, 1)
    return np.hstack([t, x.astype(np.float64)])


def sig_dim(d: int, depth: int) -> int:
    """Signature feature count (excluding the level-0 constant = 1)."""
    if d == 1:
        return depth
    return int((d ** (depth + 1) - d) // (d - 1))


def compute_signatures(series: List[np.ndarray], depth: int) -> np.ndarray:
    """Compute truncated signatures at the given depth for a list of series.

    Args:
        series: List of time-augmented (T_i, d+1) arrays.
        depth: Truncation depth N.

    Returns:
        Array of shape (n, sig_dim(d+1, depth)). iisignature.sig already
        excludes the level-0 constant.
    """
    return np.stack([iisignature.sig(x, depth) for x in series])


def _cv_score_from_model(
    model: LogisticRegressionCV,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    penalty: str,
    solver: str,
) -> float:
    """Return a valid multiclass CV accuracy.

    LogisticRegressionCV.scores_ stores per-class OvR accuracy when
    solver='liblinear' and n_classes > 2 — averaging that gives an
    upward-biased value (each per-class score is dominated by the trivial
    "class is negative" prediction on a heavily imbalanced OvR problem).
    For that case, we recompute a proper multiclass accuracy via
    cross_val_score on a plain LogisticRegression at the chosen best C.
    For binary and multinomial (lbfgs) fits, scores_ already reflects the
    true accuracy and can be averaged directly.
    """
    per_class = list(model.scores_.values())  # (n_folds, n_Cs) each
    is_ovr_multiclass = solver == "liblinear" and len(per_class) > 1
    if not is_ovr_multiclass:
        mean_over_folds_per_C = np.mean(
            [arr.mean(axis=0) for arr in per_class], axis=0
        )
        return float(mean_over_folds_per_C.max())
    C = float(np.mean(model.C_))
    est = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    scores = cross_val_score(est, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean())


def fit_evaluate(
    X_train_f: np.ndarray,
    y_train: np.ndarray,
    X_test_f: np.ndarray,
    y_test: np.ndarray,
    penalty: str,
    solver: str,
) -> dict:
    """Standardize, fit LogisticRegressionCV, return test + CV metrics."""
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
        "cv_score":    _cv_score_from_model(
            model, Xtr, y_train, cv, penalty, solver
        ),
        "C_best":      float(np.mean(model.C_)),
    }


def run_experiment() -> Tuple[pd.DataFrame, Dict[str, Optional[int]]]:
    """Signature × LR grid over datasets × N × models.

    Returns:
        df: Long-format results, one row per (dataset, model, N).
        optimal_N: Mapping dataset key → N* (highest CV score), or None
                   if no N was evaluable.
    """
    rows = []
    optimal_N: Dict[str, Optional[int]] = {}

    for ds in DATASETS:
        logger.info("=== %s ===", DISPLAY_NAMES[ds])
        X_train, y_train, X_test, y_test = load_dataset(ds)
        train_list = [add_time_channel(x) for x in _to_series_list(X_train)]
        test_list = [add_time_channel(x) for x in _to_series_list(X_test)]
        d_aug = train_list[0].shape[1]
        logger.info("  augmented dim d=%d (incl. time channel)", d_aug)

        best_cv = -np.inf
        best_N: Optional[int] = None
        for N in N_RANGE:
            dim = sig_dim(d_aug, N)
            if dim > MAX_FEATURE_DIM:
                logger.info(
                    "  N=%d: skipped (sig_dim=%d > %d)",
                    N, dim, MAX_FEATURE_DIM,
                )
                continue
            t0 = time.perf_counter()
            X_train_f = compute_signatures(train_list, N)
            X_test_f = compute_signatures(test_list, N)
            sig_time = time.perf_counter() - t0
            logger.info(
                "  N=%d (sig_dim=%d, sig_time=%.1fs):",
                N, X_train_f.shape[1], sig_time,
            )
            for model_name, penalty, solver in MODELS:
                metrics = fit_evaluate(
                    X_train_f, y_train, X_test_f, y_test, penalty, solver
                )
                metrics["dataset"] = DISPLAY_NAMES[ds]
                metrics["model"] = model_name
                metrics["N"] = N
                rows.append(metrics)
                logger.info(
                    "    %s: acc=%.3f  f1=%.3f  cv=%.3f  fit=%.2fs",
                    model_name, metrics["accuracy"], metrics["f1_macro"],
                    metrics["cv_score"], metrics["time_s"],
                )
                if metrics["cv_score"] > best_cv:
                    best_cv = metrics["cv_score"]
                    best_N = N
        optimal_N[ds] = best_N
        logger.info(
            "  N* = %s for %s (CV=%.3f)",
            best_N, DISPLAY_NAMES[ds], best_cv,
        )

    df = pd.DataFrame(rows)[
        ["dataset", "model", "N", "accuracy", "f1_macro",
         "feature_dim", "cv_score", "time_s", "C_best"]
    ]
    return df, optimal_N


def plot_results(
    df: pd.DataFrame,
    optimal_N: Dict[str, Optional[int]],
) -> plt.Figure:
    """2×n_datasets grid: top row test accuracy vs N, bottom sig-dim (log)."""
    n_cols = len(DATASETS)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8), squeeze=False)
    fig.suptitle(
        "Signature (lin+time) + LR — accuracy and dimension vs "
        "truncation depth N",
        fontsize=13, y=0.995,
    )
    colors = {"LR-L2": "#1f77b4", "LR-L1": "#ff7f0e"}
    model_names = [m[0] for m in MODELS]

    for col, ds in enumerate(DATASETS):
        sub = df[df.dataset == DISPLAY_NAMES[ds]]

        ax_a = axes[0, col]
        for model in model_names:
            g = sub[sub.model == model].sort_values("N")
            ax_a.plot(
                g.N, g.accuracy,
                marker="o", color=colors[model], label=model, linewidth=1.6,
            )
        if optimal_N[ds] is not None:
            ax_a.axvline(
                optimal_N[ds], color="grey", linestyle="--", alpha=0.6,
                label=f"N*={optimal_N[ds]}",
            )
        ax_a.set_title(DISPLAY_NAMES[ds], fontsize=11)
        ax_a.set_ylabel("Test accuracy" if col == 0 else "")
        ax_a.grid(True, linestyle=":", alpha=0.5)
        ax_a.set_axisbelow(True)
        ax_a.legend(fontsize=8, loc="lower right")

        ax_d = axes[1, col]
        g = sub.drop_duplicates(subset=["N"]).sort_values("N")
        ax_d.plot(g.N, g.feature_dim, marker="s", color="black", linewidth=1.4)
        ax_d.set_yscale("log")
        ax_d.set_xlabel("Truncation depth N")
        ax_d.set_ylabel("Signature dim (log)" if col == 0 else "")
        ax_d.grid(True, which="both", linestyle=":", alpha=0.4)
        ax_d.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def save_results(
    df: pd.DataFrame,
    fig: plt.Figure,
    optimal_N: Dict[str, Optional[int]],
    name: str,
) -> None:
    """Save results DataFrame (CSV), figure (PNG), and one N* file per dataset."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{name}.csv"
    png_path = RESULTS_DIR / f"{name}.png"
    df.to_csv(csv_path, index=False)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    for ds, N in optimal_N.items():
        txt = RESULTS_DIR / f"optimal_N_{ds}.txt"
        txt.write_text(f"{N}\n" if N is not None else "None\n")
    logger.info(
        "Results saved to %s, %s, and optimal_N_{%s}.txt",
        csv_path, png_path, ",".join(DATASETS),
    )


def main() -> None:
    """Run signature × LR grid, then save CSV / PNG / N* files."""
    logger.info("Running signature+LR experiment (N=1..10)...")
    df, optimal_N = run_experiment()
    logger.info("Summary:\n%s", df.to_string(index=False))
    fig = plot_results(df, optimal_N)
    save_results(df, fig, optimal_N, NAME)
    plt.close(fig)
    for ds, N in optimal_N.items():
        logger.info("N* [%s] = %s", DISPLAY_NAMES[ds], N)
    logger.info("Done.")


if __name__ == "__main__":
    main()
