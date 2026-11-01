"""
Experiment 5.1.2 — Truncated signature (lin+time) + LR (L1/L2), N ∈ N_RANGE (1..10).

Chapter 5, Axe 1: Signature as representation.
Pipeline per dataset:
    1. Prepend a normalized-time channel t ∈ [0, 1] to each series
       (canonical "lin+time" embedding; piecewise-linear interpolation
       between consecutive samples is implicit in the signature backend).
    2. Compute truncated signatures at depth N = 1..10 via iisignature
       (esig backend errors on CharTraj; see comment near the import).
       Capped when the signature dimension would exceed MAX_FEATURE_DIM.
    3. Fit Ridge (LR-L2, lbfgs) and LASSO (LR-L1, liblinear), each as
       GridSearchCV over Pipeline([StandardScaler, LogisticRegression])
       with C ∈ CS and 5-fold stratified CV (the scaler is refitted on each
       training fold). One C per model; CV score = best_score_; the test
       predictions come from best_estimator_. On multiclass data, lbfgs
       fits one multinomial model and liblinear fits one-vs-rest (K binary
       L1 problems), so Ridge and LASSO also differ in the multiclass scheme.

For each dataset and model, N* is the depth with the highest CV accuracy,
ties → smallest N. CV scores are rounded to 1e-9 before comparison, so two
scores that differ only by floating-point noise count as a tie. The LASSO
N* (LR-L1) is used by Axe 3 (5_3_1); both are read by the synthesis (5_4_1).

`is_N_star` marks each model's N*. `n_star_at_cap` = 1 when N* is the
largest N evaluated (the true optimum may lie beyond the grid);
`cap_reason` then says which limit applied: "n_range" (N* = max N_RANGE)
or "max_feature_dim" (the next depth would exceed MAX_FEATURE_DIM). These
N* points are starred in the figure.

Datasets: ECG200, RacketSports, CharacterTrajectories, NATOPS
Metrics : accuracy, F1 macro, n_features, CV score (mean, std), C,
          feat_time_s (signatures, train + test; shared by both models at
          a given N), fit_time_s (GridSearchCV + refit + test predict)
Diagnostics (CSV only): n_convergence_warnings (CV folds × C grid + refit),
          c_at_grid_edge (1 if the selected C is the smallest or largest in CS)

Input : data/processed/{ecg200,racketsports,chartraj,natops}_{X,y}_{train,test}.npy
Output: results/chap5/5_1_2_signature_lr.csv
        results/chap5/5_1_2_signature_lr.png
        results/chap5/5_1_2_signature_lr_predictions.csv
            (per-sample test predictions, every dataset × model × N)
        results/chap5/optimal_N_{ecg200,racketsports,chartraj,natops}_{LR-L2,LR-L1}.txt
"""

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# NOTE: iisignature is used rather than esig, whose RoughPy backend errors on
# the CharTraj batch at depth ≥ 2 ("mismatch between number of rows in data
# and number of indices"). `iisignature.sig(path, N)` returns
# sum_{k=1..N} d^k features and excludes the level-0 constant.
import iisignature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from embeddings import lin_time

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# liblinear + multiclass is deprecated but chosen intentionally for LASSO
# (CLAUDE.md, Axe 1&2 conventions): it fits one-vs-rest, whereas lbfgs (Ridge)
# is multinomial. scikit-learn >= 1.8 raises an error here, hence the pin
# scikit-learn==1.7.2. Silence the noisy FutureWarning.
warnings.filterwarnings(
    "ignore",
    message="Using the 'liblinear' solver for multiclass",
    category=FutureWarning,
)

RANDOM_STATE = 42
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/chap5")
NAME = "5_1_2_signature_lr"

DATASETS = ["ecg200", "racketsports", "chartraj", "natops"]
DISPLAY_NAMES = {
    "ecg200":       "ECG200",
    "racketsports": "RacketSports",
    "chartraj":     "CharTraj",
    "natops":       "NATOPS",
}
MODELS = [("LR-L2", "l2", "lbfgs"), ("LR-L1", "l1", "liblinear")]
MODEL_LABELS = {"LR-L2": "Ridge", "LR-L1": "LASSO"}  # names used in figures
CS = np.logspace(-3, 3, 7)
N_RANGE = list(range(1, 11))
MAX_FEATURE_DIM = 50_000
CV_DECIMALS = 9   # CV scores rounded to 1e-9 before tie comparisons


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
        return [x.astype(np.float64) for x in X]
    if X.ndim == 2:
        return [X[i, :, None].astype(np.float64) for i in range(X.shape[0])]
    return [X[i].astype(np.float64) for i in range(X.shape[0])]


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


def run_experiment() -> Tuple[
    pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Optional[int]]]
]:
    """Signature × LR grid over datasets × N × models.

    Returns:
        df: Long-format results, one row per (dataset, model, N).
        preds: Per-sample test predictions, one row per
               (dataset, model, N, test sample).
        optimal_N: dataset key → model → N* (highest CV score, ties →
                   smallest N), or None if no N was evaluable.
    """
    rows = []
    pred_rows = []
    optimal_N: Dict[str, Dict[str, Optional[int]]] = {}

    for ds in DATASETS:
        logger.info("=== %s ===", DISPLAY_NAMES[ds])
        X_train, y_train, X_test, y_test = load_dataset(ds)
        train_list = [lin_time(x) for x in _to_series_list(X_train)]
        test_list = [lin_time(x) for x in _to_series_list(X_test)]
        d_aug = train_list[0].shape[1]
        logger.info("  augmented dim d=%d (incl. time channel)", d_aug)

        # N_RANGE is increasing and only a strictly better (rounded) CV score
        # replaces the current best, so ties go to the smallest N.
        best_cv = {m: -np.inf for m, _, _ in MODELS}
        best_N: Dict[str, Optional[int]] = {m: None for m, _, _ in MODELS}
        evaluated_N: List[int] = []
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
            evaluated_N.append(N)
            for model_name, penalty, solver in MODELS:
                metrics, y_pred = fit_evaluate(
                    X_train_f, y_train, X_test_f, y_test, penalty, solver
                )
                metrics.update(
                    dataset=DISPLAY_NAMES[ds], model=model_name, N=N,
                    feat_time_s=sig_time, is_N_star=False, n_star_at_cap=0,
                    cap_reason="",
                )
                rows.append(metrics)
                pred_rows.append(pd.DataFrame({
                    "dataset": DISPLAY_NAMES[ds],
                    "model":   model_name,
                    "N":       N,
                    "sample":  np.arange(len(y_test)),
                    "y_true":  y_test,
                    "y_pred":  y_pred,
                }))
                logger.info(
                    "    %s: acc=%.3f  f1=%.3f  cv=%.3f±%.3f  C=%g  fit=%.2fs  "
                    "conv_warn=%d",
                    MODEL_LABELS[model_name], metrics["accuracy"],
                    metrics["f1_macro"], metrics["cv_score"],
                    metrics["cv_std"], metrics["C_best"],
                    metrics["fit_time_s"], metrics["n_convergence_warnings"],
                )
                cv = round(metrics["cv_score"], CV_DECIMALS)
                if cv > best_cv[model_name]:
                    best_cv[model_name] = cv
                    best_N[model_name] = N
        optimal_N[ds] = best_N
        if evaluated_N:
            # Largest N evaluated: the end of N_RANGE, or the last depth
            # within MAX_FEATURE_DIM.
            N_max = max(evaluated_N)
            reason = "n_range" if N_max == N_RANGE[-1] else "max_feature_dim"
            for r in rows:
                if r["dataset"] == DISPLAY_NAMES[ds] and r["N"] == best_N[r["model"]]:
                    r["is_N_star"] = True
                    if r["N"] == N_max:
                        r["n_star_at_cap"] = 1
                        r["cap_reason"] = reason
        for model_name, _, _ in MODELS:
            logger.info(
                "  N* = %s for %s / %s (CV=%.3f)",
                best_N[model_name], DISPLAY_NAMES[ds],
                MODEL_LABELS[model_name], best_cv[model_name],
            )

    df = pd.DataFrame(rows)[
        ["dataset", "model", "N", "accuracy", "f1_macro", "n_features",
         "cv_score", "cv_std", "C_best", "feat_time_s", "fit_time_s",
         "is_N_star", "n_star_at_cap", "cap_reason",
         "n_convergence_warnings", "c_at_grid_edge"]
    ]
    return df, pd.concat(pred_rows, ignore_index=True), optimal_N


def plot_results(
    df: pd.DataFrame,
    optimal_N: Dict[str, Dict[str, Optional[int]]],
) -> plt.Figure:
    """2×n_datasets grid: top row test accuracy vs N, bottom sig-dim (log).

    A star marks an N* at the cap of the evaluated grid (n_star_at_cap).
    """
    n_cols = len(DATASETS)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8), squeeze=False)
    fig.suptitle(
        "Signature (lin+time) + régression logistique — taux de bonne "
        "classification et dimension selon l'ordre de troncature N",
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
                g.N, g.accuracy, marker="o", color=colors[model],
                label=MODEL_LABELS[model], linewidth=1.6,
            )
            N_star = optimal_N[ds][model]
            if N_star is not None:
                ax_a.axvline(
                    N_star, color=colors[model], linestyle="--", alpha=0.6,
                    label=f"N* {MODEL_LABELS[model]} = {N_star}",
                )
            cap = g[g.n_star_at_cap == 1]
            ax_a.scatter(
                cap.N, cap.accuracy, marker="*", s=220, color=colors[model],
                edgecolors="black", linewidths=0.8, zorder=4,
            )
        ax_a.set_title(DISPLAY_NAMES[ds], fontsize=11)
        ax_a.set_ylabel("Taux de bonne classification (test)" if col == 0 else "")
        ax_a.grid(True, linestyle=":", alpha=0.5)
        ax_a.set_axisbelow(True)
        ax_a.legend(fontsize=8, loc="lower right")

        ax_d = axes[1, col]
        g = sub.drop_duplicates(subset=["N"]).sort_values("N")
        ax_d.plot(g.N, g.n_features, marker="s", color="black", linewidth=1.4)
        ax_d.set_yscale("log")
        ax_d.set_xlabel("Ordre de troncature N")
        ax_d.set_ylabel("Nombre de caractéristiques (échelle log)" if col == 0 else "")
        ax_d.grid(True, which="both", linestyle=":", alpha=0.4)
        ax_d.set_axisbelow(True)

    fig.text(
        0.5, 0.005,
        "★ : N* égal au plus grand N évalué (plafond de la grille : "
        "N max ou dimension maximale).",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return fig


def save_results(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    fig: plt.Figure,
    optimal_N: Dict[str, Dict[str, Optional[int]]],
    name: str,
) -> None:
    """Save results (CSV), predictions (CSV), figure (PNG), one N* file per dataset × model."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{name}.csv"
    pred_path = RESULTS_DIR / f"{name}_predictions.csv"
    png_path = RESULTS_DIR / f"{name}.png"
    df.to_csv(csv_path, index=False)
    preds.to_csv(pred_path, index=False)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    for ds, per_model in optimal_N.items():
        for model_name, N in per_model.items():
            txt = RESULTS_DIR / f"optimal_N_{ds}_{model_name}.txt"
            txt.write_text(f"{N}\n" if N is not None else "None\n")
    logger.info(
        "Results saved to %s, %s, %s, and optimal_N_{%s}_{%s}.txt",
        csv_path, pred_path, png_path, ",".join(DATASETS),
        ",".join(m for m, _, _ in MODELS),
    )


def main() -> None:
    """Run signature × LR grid, then save CSV / PNG / N* files."""
    logger.info(
        "Running signature+LR experiment (N=%d..%d)...", N_RANGE[0], N_RANGE[-1]
    )
    logger.info("CPU cores available: %s (os.cpu_count()); GridSearchCV n_jobs=-1", os.cpu_count())
    warm_up()
    df, preds, optimal_N = run_experiment()
    logger.info("Summary:\n%s", df.to_string(index=False))
    fig = plot_results(df, optimal_N)
    save_results(df, preds, fig, optimal_N, NAME)
    plt.close(fig)
    for ds, per_model in optimal_N.items():
        for model_name, N in per_model.items():
            logger.info(
                "N* [%s / %s] = %s",
                DISPLAY_NAMES[ds], MODEL_LABELS[model_name], N,
            )
    logger.info("Done.")


if __name__ == "__main__":
    main()
