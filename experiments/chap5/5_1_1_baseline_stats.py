"""
Experiment 5.1.1 — Baseline statistical features + LR (L1/L2).

Chapter 5, Axe 1: Signature as representation.
Establishes a naive baseline that discards temporal structure: per-channel
summary statistics (mean, std, min, max) computed on the raw series.
Two linear models are then fit: Ridge (LR-L2, lbfgs) and LASSO (LR-L1,
liblinear), each as GridSearchCV over Pipeline([StandardScaler,
LogisticRegression]) with C ∈ CS and 5-fold stratified CV (same protocol
as 5_1_2, so that the Axe 1 McNemar tests compare the same model).
On multiclass data the two models differ in more than the penalty: lbfgs fits
one multinomial model, liblinear fits one-vs-rest (K binary L1 problems).
ECG200 is binary, so there the distinction does not arise.

Datasets: ECG200, RacketSports, CharacterTrajectories, NATOPS
Metrics : accuracy, F1 macro, n_features, CV score (mean, std), C,
          feat_time_s (feature extraction, train + test; shared by both
          models), fit_time_s (GridSearchCV + refit + test predict)
Diagnostics (CSV only): n_convergence_warnings (CV folds × C grid + refit),
          c_at_grid_edge (1 if the selected C is the smallest or largest in CS)

Input : data/processed/{ecg200,racketsports,chartraj,natops}_{X,y}_{train,test}.npy
Output: results/chap5/5_1_1_baseline_stats.csv
        results/chap5/5_1_1_baseline_stats.png
        results/chap5/5_1_1_baseline_stats_predictions.csv
            (per-sample test predictions, every dataset × model)
"""

import logging
import os
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# liblinear + multiclass is deprecated in sklearn but chosen intentionally
# for LASSO (CLAUDE.md, Axe 1&2 conventions): it fits one-vs-rest, whereas
# lbfgs (Ridge) is multinomial. scikit-learn >= 1.8 raises an error here,
# hence the pin scikit-learn==1.7.2. Silence the noisy FutureWarning.
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
NAME = "5_1_1_baseline_stats"

DATASETS = ["ecg200", "racketsports", "chartraj", "natops"]
DISPLAY_NAMES = {
    "ecg200": "ECG200",
    "racketsports": "RacketSports",
    "chartraj": "CharTraj",
    "natops": "NATOPS",
}
MODELS = [("LR-L2", "l2", "lbfgs"), ("LR-L1", "l1", "liblinear")]
MODEL_LABELS = {"LR-L2": "Ridge", "LR-L1": "LASSO"}  # names used in figures
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
        return [x.astype(np.float64) for x in X]
    if X.ndim == 2:
        return [X[i, :, None].astype(np.float64) for i in range(X.shape[0])]
    return [X[i].astype(np.float64) for i in range(X.shape[0])]


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


class CountingLogisticRegression(LogisticRegression):
    """LogisticRegression that counts the ConvergenceWarnings raised by its own fit.

    GridSearchCV(n_jobs=-1) fits in worker processes, where this script cannot
    catch warnings: the count is stored on the fitted estimator and read back
    through the `n_conv_warnings` scorer. Other warnings are re-emitted.
    """

    def fit(self, X, y, sample_weight=None):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            super().fit(X, y, sample_weight=sample_weight)
        self.n_convergence_warnings_ = 0
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                self.n_convergence_warnings_ += 1
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        return self


def _n_conv_warnings(estimator, X, y) -> int:
    """GridSearchCV scorer: ConvergenceWarnings raised by this fold's fit."""
    return estimator.named_steps["clf"].n_convergence_warnings_


def _fit_diagnostics(search: GridSearchCV) -> dict:
    """ConvergenceWarnings over CV folds × C grid + refit; selected C at the grid edge."""
    n_cv = sum(
        int(search.cv_results_[f"split{k}_test_n_conv_warnings"].sum())
        for k in range(search.n_splits_)
    )
    n_refit = search.best_estimator_.named_steps["clf"].n_convergence_warnings_
    C = search.best_params_["clf__C"]
    return {
        "n_convergence_warnings": n_cv + n_refit,
        "c_at_grid_edge":         int(np.isclose(C, CS[0]) or np.isclose(C, CS[-1])),
    }


def fit_evaluate(
    X_train_f: np.ndarray,
    y_train: np.ndarray,
    X_test_f: np.ndarray,
    y_test: np.ndarray,
    penalty: str,
    solver: str,
) -> Tuple[dict, np.ndarray]:
    """Grid-search C on Pipeline(scaler, LR) by CV; evaluate best_estimator_ on test.

    Returns:
        metrics: Test and CV metrics for the selected C.
        y_pred: Test predictions of best_estimator_.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CountingLogisticRegression(
            penalty=penalty,
            solver=solver,
            max_iter=5000,
            random_state=RANDOM_STATE,
        )),
    ])
    search = GridSearchCV(
        pipe,
        {"clf__C": CS},
        scoring={"accuracy": "accuracy", "n_conv_warnings": _n_conv_warnings},
        refit="accuracy",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    search.fit(X_train_f, y_train)
    y_pred = search.best_estimator_.predict(X_test_f)
    elapsed = time.perf_counter() - t0
    metrics = {
        "accuracy":           accuracy_score(y_test, y_pred),
        "f1_macro":           f1_score(y_test, y_pred, average="macro"),
        "n_features":         X_train_f.shape[1],
        "cv_score":           float(search.best_score_),
        "cv_std":             float(search.cv_results_["std_test_accuracy"][search.best_index_]),
        "C_best":             float(search.best_params_["clf__C"]),
        "fit_time_s":         elapsed,
        **_fit_diagnostics(search),
    }
    return metrics, y_pred


def warm_up() -> None:
    """Untimed dummy fits, so the first timed fit does not pay joblib's worker start-up."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(50, 3))
    y = np.arange(50) % 2
    for _, penalty, solver in MODELS:
        fit_evaluate(X, y, X, y, penalty, solver)


def run_experiment() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run baseline evaluation across all datasets × models.

    Returns:
        df: Long-format results, one row per (dataset, model).
        preds: Per-sample test predictions, one row per
               (dataset, model, test sample).
    """
    rows = []
    pred_rows = []
    for ds in DATASETS:
        logger.info("--- %s ---", DISPLAY_NAMES[ds])
        X_train, y_train, X_test, y_test = load_dataset(ds)
        t0 = time.perf_counter()
        X_train_f = stats_features(X_train)
        X_test_f = stats_features(X_test)
        feat_time = time.perf_counter() - t0
        logger.info(
            "  features: train %s, test %s (feat_time=%.3fs)",
            X_train_f.shape, X_test_f.shape, feat_time,
        )
        for name, penalty, solver in MODELS:
            metrics, y_pred = fit_evaluate(
                X_train_f, y_train, X_test_f, y_test, penalty, solver
            )
            metrics["dataset"] = DISPLAY_NAMES[ds]
            metrics["model"] = name
            metrics["feat_time_s"] = feat_time
            rows.append(metrics)
            pred_rows.append(pd.DataFrame({
                "dataset": DISPLAY_NAMES[ds],
                "model":   name,
                "sample":  np.arange(len(y_test)),
                "y_true":  y_test,
                "y_pred":  y_pred,
            }))
            logger.info(
                "  %s: acc=%.3f  f1=%.3f  cv=%.3f±%.3f  C=%g  time=%.2fs  conv_warn=%d",
                MODEL_LABELS[name], metrics["accuracy"], metrics["f1_macro"],
                metrics["cv_score"], metrics["cv_std"], metrics["C_best"],
                metrics["fit_time_s"], metrics["n_convergence_warnings"],
            )
    df = pd.DataFrame(rows)[
        ["dataset", "model", "accuracy", "f1_macro", "n_features",
         "cv_score", "cv_std", "C_best", "feat_time_s", "fit_time_s",
         "n_convergence_warnings", "c_at_grid_edge"]
    ]
    return df, pd.concat(pred_rows, ignore_index=True)


def plot_results(df: pd.DataFrame) -> plt.Figure:
    """2x2 grouped-bar figure, one subplot per metric.

    Args:
        df: DataFrame from run_experiment.

    Returns:
        Matplotlib figure.
    """
    metrics = [
        ("accuracy",   "Taux de bonne classification (test)", "%.3f"),
        ("f1_macro",   "F1 macro (test)",                     "%.3f"),
        ("n_features", "Nombre de caractéristiques",          "%d"),
        ("fit_time_s", "Temps d'ajustement (s)",              "%.2f"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "Référence statistique (moyenne / écart-type / min / max par canal) — Ridge vs LASSO",
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
                x + offset, values, width, label=MODEL_LABELS[model],
                color=colors[model],
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


def save_results(
    df: pd.DataFrame, preds: pd.DataFrame, fig: plt.Figure, name: str
) -> None:
    """Save results (CSV), per-sample test predictions (CSV) and figure (PNG)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{name}.csv"
    pred_path = RESULTS_DIR / f"{name}_predictions.csv"
    png_path = RESULTS_DIR / f"{name}.png"
    df.to_csv(csv_path, index=False)
    preds.to_csv(pred_path, index=False)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    logger.info("Results saved to %s, %s and %s", csv_path, pred_path, png_path)


def main() -> None:
    """Run full baseline experiment."""
    logger.info("Running baseline stats experiment...")
    logger.info("CPU cores available: %s (os.cpu_count()); GridSearchCV n_jobs=-1", os.cpu_count())
    warm_up()
    df, preds = run_experiment()
    logger.info("Summary:\n%s", df.to_string(index=False))
    logger.info("Plotting...")
    fig = plot_results(df)
    save_results(df, preds, fig, NAME)
    plt.close(fig)
    logger.info("Done.")


if __name__ == "__main__":
    main()
