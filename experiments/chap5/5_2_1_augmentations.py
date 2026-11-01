"""
Experiment 5.2.1 — Effect of path augmentations on signature + LASSO.

Chapter 5, Axe 2: Effect of augmentations.
Compare four path configurations (defined in embeddings.py) under LASSO on
each dataset:

    1. lin+time              — canonical baseline
    2. lin+time+lead-lag     — Fermanian (2021) delayed-copy lead-lag, lag 1
    3. rectilinear+time      — Fermanian (2021) rectilinear, axis by axis,
                                time last
    4. lin+time+cumsum       — cumulative sum applied before time augmentation

Selection, entirely by 5-fold CV on the training set:
    1. For each configuration, the depth N ∈ N_RANGE (within MAX_FEATURE_DIM)
       with the highest CV accuracy, ties → smallest N (the config's N*).
    2. The configuration with the highest CV accuracy at its N*, ties → the
       earlier configuration in CONFIGS (so lin+time wins a tie).
CV scores are rounded to 1e-9 before both comparisons, so two scores that
differ only by floating-point noise count as a tie.
The selected configuration is saved to results/chap5/best_config_{dataset}.txt.

`n_star_at_cap` = 1 on a config's N* row when N* is the largest N evaluated
for that config; `cap_reason` says which limit applied: "n_range"
(N* = max N_RANGE) or "max_feature_dim" (the next depth would exceed
MAX_FEATURE_DIM). These points are starred in both figures.
Diagnostics (CSV only): n_convergence_warnings (CV folds × C grid + refit),
c_at_grid_edge (1 if the selected C is the smallest or largest in CS).

Memory: a MemoryError while computing the signatures or fitting a (config, N)
is logged as a WARNING and that row is written with NaN metrics; the sweep
goes on. Any other exception stops the script. At the end the script logs the
number of rows skipped for memory reasons, and exits with code 1 (after
saving its outputs) if a row it needs is missing or NaN: the N* row of every
configuration, or the selected configuration of every dataset.

Datasets: ECG200, RacketSports, CharacterTrajectories, NATOPS
Model   : LASSO (LR-L1, liblinear), GridSearchCV over
          Pipeline([StandardScaler, LogisticRegression]) with C ∈ CS.
          On multiclass data liblinear fits one-vs-rest (K binary L1
          problems), unlike the multinomial Ridge (lbfgs) of 5_1_x;
          scikit-learn >= 1.8 raises on it, hence the 1.7.2 pin.
CV      : StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

Input : data/processed/{ecg200,racketsports,chartraj,natops}_{X,y}_{train,test}.npy
Output: results/chap5/5_2_1_augmentations.csv
            one row per (dataset, config, N); `is_config_N_star` marks each
            config's N*, `is_selected` the selected config at its N*
        results/chap5/5_2_1_augmentations.png
            test accuracy and nonzero features at each config's N*
        results/chap5/5_2_1_augmentations_acc_vs_features.png
            test accuracy vs number of features (log), one curve per config
        results/chap5/5_2_1_augmentations_predictions.csv
            per-sample test predictions, every dataset × config × N
        results/chap5/best_config_{ecg200,racketsports,chartraj,natops}.txt

Note: signature backend is iisignature (esig hits an internal error on the
CharTraj batch; see 5_1_2 header comment).
"""

import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import iisignature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from embeddings import TRANSFORMS, path_dim
from runinfo import log_versions

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
NAME = "5_2_1_augmentations"

DATASETS = ["ecg200", "racketsports", "chartraj", "natops"]
DISPLAY_NAMES = {
    "ecg200":       "ECG200",
    "racketsports": "RacketSports",
    "chartraj":     "CharTraj",
    "natops":       "NATOPS",
}

# Order matters: ties between configurations go to the earlier one.
CONFIGS = ["lin+time", "lin+time+lead-lag", "rectilinear+time", "lin+time+cumsum"]
CONFIG_COLORS = {
    "lin+time":          "blue",
    "lin+time+lead-lag": "red",
    "rectilinear+time":  "green",
    "lin+time+cumsum":   "orange",
}
# Distinct markers / line styles so that overlapping curves stay visible
# (lin+time and rectilinear+time share n_features at every N).
CONFIG_STYLES = {
    "lin+time":          ("o", "-"),
    "lin+time+lead-lag": ("s", "-"),
    "rectilinear+time":  ("^", "--"),
    "lin+time+cumsum":   ("D", ":"),
}

CS = np.logspace(-3, 3, 7)
N_RANGE = list(range(1, 11))
MAX_FEATURE_DIM = 50_000
CV_DECIMALS = 9   # CV scores rounded to 1e-9 before tie comparisons


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------

def sig_dim(d: int, depth: int) -> int:
    """Signature feature count (level-0 = 1 already excluded)."""
    if d == 1:
        return depth
    return int((d ** (depth + 1) - d) // (d - 1))


def compute_signatures(
    series: List[np.ndarray], config: str, depth: int
) -> np.ndarray:
    """Apply the config transform then compute truncated signatures."""
    transform = TRANSFORMS[config]
    return np.stack([iisignature.sig(transform(x), depth) for x in series])


# ---------------------------------------------------------------------------
# Fit / evaluate
# ---------------------------------------------------------------------------

def _count_nonzero_features(coef: np.ndarray) -> int:
    """Number of features nonzero in at least one class (union support)."""
    return int(np.any(np.atleast_2d(coef) != 0.0, axis=0).sum())


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
) -> Tuple[Dict[str, float], np.ndarray]:
    """Grid-search C on Pipeline(scaler, LASSO) by CV; evaluate best_estimator_ on test.

    Returns:
        metrics: Test and CV metrics for the selected C.
        y_pred: Test predictions of best_estimator_.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CountingLogisticRegression(
            penalty="l1",
            solver="liblinear",
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
        "cv_score":           float(search.best_score_),
        "cv_std":             float(search.cv_results_["std_test_accuracy"][search.best_index_]),
        "C_best":             float(search.best_params_["clf__C"]),
        "test_accuracy":      accuracy_score(y_test, y_pred),
        "test_f1_macro":      f1_score(y_test, y_pred, average="macro"),
        "n_nonzero_features": _count_nonzero_features(
            search.best_estimator_.named_steps["clf"].coef_
        ),
        "fit_time_s":         elapsed,
        **_fit_diagnostics(search),
    }
    return metrics, y_pred


def warm_up() -> None:
    """Untimed dummy fit, so the first timed fit does not pay joblib's worker start-up."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(50, 3))
    y = np.arange(50) % 2
    fit_evaluate(X, y, X, y)


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

def _nan_metrics() -> Dict[str, float]:
    """Placeholder metrics for a failed fit."""
    return {
        "cv_score":           np.nan,
        "cv_std":             np.nan,
        "C_best":             np.nan,
        "test_accuracy":      np.nan,
        "test_f1_macro":      np.nan,
        "n_nonzero_features": np.nan,
        "fit_time_s":         np.nan,
        "n_convergence_warnings": np.nan,
        "c_at_grid_edge":     np.nan,
    }


def run_experiment() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Optional[str]]]:
    """Sweep datasets × configs × N, select N* per config then the config, all by CV.

    Returns:
        df: One row per (dataset, config, N), with the selection flags.
        preds: Per-sample test predictions, one row per
               (dataset, config, N, test sample).
        best_configs: dataset key → selected config (None if nothing fitted).
    """
    rows = []
    pred_rows = []
    best_configs: Dict[str, Optional[str]] = {}

    for ds in DATASETS:
        logger.info("=== %s ===", DISPLAY_NAMES[ds])
        t_ds = time.perf_counter()
        X_train, y_train, X_test, y_test = load_dataset(ds)
        train_list = _to_series_list(X_train)
        test_list = _to_series_list(X_test)
        d_base = train_list[0].shape[1]
        logger.info("  base d=%d", d_base)

        # (config, N*, rounded CV at N*) for each config that produced a fit.
        config_best: List[Tuple[str, int, float]] = []

        for config in CONFIGS:
            t_cfg = time.perf_counter()
            d_path = path_dim(config, d_base)
            # N_RANGE is increasing and only a strictly better (rounded) CV
            # score replaces the current best, so ties go to the smallest N.
            best_cv, best_N = -np.inf, None
            fitted_N: List[int] = []
            for N in N_RANGE:
                dim = sig_dim(d_path, N)
                if dim > MAX_FEATURE_DIM:
                    logger.info(
                        "  %s: N≥%d skipped (sig_dim=%d > %d)",
                        config, N, dim, MAX_FEATURE_DIM,
                    )
                    break  # sig_dim grows with N
                row_stub = {
                    "dataset":    DISPLAY_NAMES[ds],
                    "config":     config,
                    "N":          N,
                    "n_features": dim,
                    "is_config_N_star": False,
                    "is_selected":      False,
                    "n_star_at_cap":    0,
                    "cap_reason":       "",
                }
                try:
                    t0 = time.perf_counter()
                    X_train_f = compute_signatures(train_list, config, N)
                    X_test_f = compute_signatures(test_list, config, N)
                    sig_t = time.perf_counter() - t0
                    metrics, y_pred = fit_evaluate(
                        X_train_f, y_train, X_test_f, y_test
                    )
                except MemoryError:
                    # Only memory errors are tolerated; anything else stops the script.
                    logger.warning(
                        "  %s N=%d skipped: MemoryError (row written as NaN)",
                        config, N,
                    )
                    rows.append({**row_stub, "feat_time_s": np.nan, **_nan_metrics()})
                    continue
                rows.append({**row_stub, "feat_time_s": sig_t, **metrics})
                fitted_N.append(N)
                pred_rows.append(pd.DataFrame({
                    "dataset": DISPLAY_NAMES[ds],
                    "config":  config,
                    "N":       N,
                    "sample":  np.arange(len(y_test)),
                    "y_true":  y_test,
                    "y_pred":  y_pred,
                }))
                logger.info(
                    "  %s N=%d (dim=%d): cv=%.3f±%.3f  C=%g  acc=%.3f  "
                    "nnz=%d  sig=%.1fs  fit=%.1fs  conv_warn=%d",
                    config, N, dim, metrics["cv_score"], metrics["cv_std"],
                    metrics["C_best"], metrics["test_accuracy"],
                    metrics["n_nonzero_features"], sig_t,
                    metrics["fit_time_s"], metrics["n_convergence_warnings"],
                )
                cv = round(metrics["cv_score"], CV_DECIMALS)
                if cv > best_cv:
                    best_cv, best_N = cv, N
            if best_N is not None:
                config_best.append((config, best_N, best_cv))
                N_max = max(fitted_N)
                if best_N == N_max:
                    if N_max == N_RANGE[-1]:
                        reason = "n_range"
                    elif sig_dim(d_path, N_max + 1) > MAX_FEATURE_DIM:
                        reason = "max_feature_dim"
                    else:
                        reason = "fit_failed"   # the next depth raised MemoryError
                    for r in rows:
                        if (r["dataset"] == DISPLAY_NAMES[ds] and r["config"] == config
                                and r["N"] == best_N):
                            r["n_star_at_cap"] = 1
                            r["cap_reason"] = reason
            logger.info(
                "  %s: N*=%s (CV=%.3f), config time %.1fs",
                config, best_N, best_cv, time.perf_counter() - t_cfg,
            )

        # Configs are scanned in CONFIGS order and only a strictly better
        # (rounded) CV score replaces the current best, so ties go to the
        # earlier config.
        best = None
        for cand in config_best:
            if best is None or cand[2] > best[2]:
                best = cand
        best_configs[ds] = best[0] if best else None
        for config, N_star, _ in config_best:
            for r in rows:
                if (r["dataset"] == DISPLAY_NAMES[ds] and r["config"] == config
                        and r["N"] == N_star):
                    r["is_config_N_star"] = True
                    r["is_selected"] = config == best[0]
        logger.info(
            "  selected config [%s] = %s (N*=%s, CV=%.3f); dataset time %.1fs",
            DISPLAY_NAMES[ds], best_configs[ds],
            best[1] if best else None, best[2] if best else np.nan,
            time.perf_counter() - t_ds,
        )

    df = pd.DataFrame(rows)[
        ["dataset", "config", "N", "n_features", "n_nonzero_features",
         "cv_score", "cv_std", "C_best", "test_accuracy", "test_f1_macro",
         "feat_time_s", "fit_time_s", "is_config_N_star", "is_selected",
         "n_star_at_cap", "cap_reason", "n_convergence_warnings", "c_at_grid_edge"]
    ]
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    return df, preds, best_configs


def selection_problems(df: pd.DataFrame) -> List[str]:
    """Selected rows that are missing or NaN: each config's N*, each dataset's selection."""
    problems = []
    for ds in DATASETS:
        sub = df[df.dataset == DISPLAY_NAMES[ds]]
        for config in CONFIGS:
            star = sub[(sub.config == config) & sub.is_config_N_star]
            if len(star) != 1 or star[["cv_score", "test_accuracy"]].isna().any(axis=None):
                problems.append(f"{DISPLAY_NAMES[ds]} / {config}: no valid N* row")
        sel = sub[sub.is_selected]
        if len(sel) != 1 or sel[["cv_score", "test_accuracy"]].isna().any(axis=None):
            problems.append(f"{DISPLAY_NAMES[ds]}: no valid selected configuration")
    return problems


# ---------------------------------------------------------------------------
# Plots + save
# ---------------------------------------------------------------------------

def plot_results(df: pd.DataFrame) -> plt.Figure:
    """2×n_datasets grid at each config's N*: top test accuracy, bottom nonzero features.

    The CV-selected configuration is outlined in black; a star in the tick
    label marks an N* at the cap of the evaluated grid. Nothing on this
    figure is chosen on test accuracy.
    """
    n_cols = len(DATASETS)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 8), squeeze=False)
    fig.suptitle(
        "Axe 2 — effet des augmentations (LASSO, chaque configuration à son N* "
        "choisi par validation croisée)",
        fontsize=13, y=0.995,
    )

    for col, ds in enumerate(DATASETS):
        sub = df[(df.dataset == DISPLAY_NAMES[ds]) & df.is_config_N_star]
        sub = sub.set_index("config").reindex(
            [c for c in CONFIGS if c in set(sub.config)]
        ).reset_index()
        x = np.arange(len(sub))
        colors = [CONFIG_COLORS.get(c, "grey") for c in sub.config]
        edges = ["black" if s else "none" for s in sub.is_selected]
        labels = [
            f"{c}\n(N*={int(n)}{' ★' if cap == 1 else ''})"
            for c, n, cap in zip(sub.config, sub.N, sub.n_star_at_cap)
        ]

        for row, (col_name, ylabel, fmt) in enumerate([
            ("test_accuracy",      "Taux de bonne classification (test)",   "{:.3f}"),
            ("n_nonzero_features", "Nombre de caractéristiques non nulles", "{:.0f}"),
        ]):
            ax = axes[row, col]
            vals = sub[col_name].values.astype(float)
            ax.bar(x, vals, color=colors, edgecolor=edges, linewidth=2)
            for i, v in enumerate(vals):
                if not np.isnan(v):
                    ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
            ax.set_ylabel(ylabel if col == 0 else "")
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            ax.set_axisbelow(True)
        axes[0, col].set_title(DISPLAY_NAMES[ds], fontsize=11)

    fig.text(
        0.5, 0.005,
        "Contour noir : configuration choisie par validation croisée. "
        "★ : N* égal au plus grand N évalué (plafond de la grille : N max ou "
        "dimension maximale).",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return fig


def plot_acc_vs_features(df: pd.DataFrame) -> plt.Figure:
    """Test accuracy vs number of signature features (log x), one curve per config.

    Each point is one truncation depth N (as in Fermanian, 2021). The
    hollow marker is the config's CV-selected N*; a star marks an N* at the
    cap of the evaluated grid.
    """
    n_cols = len(DATASETS)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.2), squeeze=False)
    fig.suptitle(
        "Axe 2 — taux de bonne classification (test) du LASSO selon le nombre "
        "de caractéristiques (un point par N)",
        fontsize=13, y=1.0,
    )
    for col, ds in enumerate(DATASETS):
        ax = axes[0, col]
        sub = df[df.dataset == DISPLAY_NAMES[ds]].dropna(subset=["test_accuracy"])
        for config in CONFIGS:
            g = sub[sub.config == config].sort_values("N")
            if g.empty:
                continue
            color = CONFIG_COLORS.get(config, "grey")
            marker, linestyle = CONFIG_STYLES.get(config, ("o", "-"))
            ax.plot(
                g.n_features, g.test_accuracy, marker=marker, markersize=5,
                linestyle=linestyle, color=color, label=config,
                linewidth=1.4, alpha=0.85,
            )
            star = g[g.is_config_N_star]
            ax.scatter(
                star.n_features, star.test_accuracy, s=110,
                facecolors="none", edgecolors=color, linewidths=1.8, zorder=3,
            )
            cap = star[star.n_star_at_cap == 1]
            ax.scatter(
                cap.n_features, cap.test_accuracy, marker="*", s=160,
                color=color, edgecolors="black", linewidths=0.8, zorder=4,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(DISPLAY_NAMES[ds], fontsize=11)
        ax.set_xlabel("Nombre de caractéristiques (échelle log)")
        ax.set_ylabel("Taux de bonne classification (test)" if col == 0 else "")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.text(
        0.5, -0.04,
        "Marqueur creux : N* choisi par validation croisée pour chaque "
        "configuration.\n★ : N* égal au plus grand N évalué (plafond de la "
        "grille : N max ou dimension maximale).",
        ha="center", fontsize=9,
    )
    fig.tight_layout()
    return fig


def save_results(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    figs: Dict[str, plt.Figure],
    best_configs: Dict[str, Optional[str]],
    name: str,
) -> None:
    """Save CSV, predictions CSV, figures, and one best_config txt per dataset."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{name}.csv"
    pred_path = RESULTS_DIR / f"{name}_predictions.csv"
    df.to_csv(csv_path, index=False)
    preds.to_csv(pred_path, index=False)
    for suffix, fig in figs.items():
        fig.savefig(RESULTS_DIR / f"{name}{suffix}.png", dpi=150, bbox_inches="tight")
    for ds, cfg in best_configs.items():
        (RESULTS_DIR / f"best_config_{ds}.txt").write_text(
            f"{cfg}\n" if cfg else "None\n"
        )
    logger.info(
        "Results saved to %s, %s, %d figure(s), and best_config_{%s}.txt",
        csv_path, pred_path, len(figs), ",".join(DATASETS),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run augmentation × N grid, save CSVs / PNGs / best_config txts."""
    log_versions(logger)
    t_start = time.perf_counter()
    logger.info("Running augmentation experiment (N selected per config by CV)...")
    logger.info(
        "CPU cores available: %s (os.cpu_count()); GridSearchCV n_jobs=-1",
        os.cpu_count(),
    )
    warm_up()
    df, preds, best_configs = run_experiment()
    logger.info("Summary:\n%s", df.to_string(index=False))
    figs = {"": plot_results(df), "_acc_vs_features": plot_acc_vs_features(df)}
    save_results(df, preds, figs, best_configs, NAME)
    for fig in figs.values():
        plt.close(fig)

    for ds in DATASETS:
        sel = df[(df.dataset == DISPLAY_NAMES[ds]) & df.is_selected]
        if sel.empty:
            logger.info("%s — no valid config", DISPLAY_NAMES[ds])
            continue
        row = sel.iloc[0]
        logger.info(
            "%s — selected by CV: %s at N*=%d (CV=%.3f) → test acc=%.3f, "
            "F1=%.3f, features=%d, nonzero=%d",
            DISPLAY_NAMES[ds], row.config, int(row.N), row.cv_score,
            row.test_accuracy, row.test_f1_macro,
            int(row.n_features), int(row.n_nonzero_features),
        )
    logger.info(
        "Rows skipped for memory reasons (MemoryError, NaN in the CSV): %d",
        int(df.cv_score.isna().sum()),
    )
    logger.info("Total run time: %.1f min", (time.perf_counter() - t_start) / 60)
    problems = selection_problems(df)
    if problems:
        for msg in problems:
            logger.error("Selection incomplete: %s", msg)
        sys.exit(1)
    logger.info("Done.")


if __name__ == "__main__":
    main()
