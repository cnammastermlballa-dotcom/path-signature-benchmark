"""
Experiment 5.3.1 — Effect of path augmentations on signature + LR-L1.

Chapter 5, Axe 2: Effect of augmentations.
At the optimal truncation N* identified in Axe 1 (5_2_2), compare four
"lin+time"-based path configurations under LASSO on each dataset:

    1. lin+time              — canonical baseline
    2. lin+time+lead-lag     — lead-lag with lag=1 on the time-augmented path
    3. rectilinear+time      — rectilinear interpolation (time-jump then
                                value-jump per transition)
    4. lin+time+cumsum       — cumulative sum applied before time augmentation

For each dataset the best config is the one with the highest 5-fold CV
accuracy and is saved to results/chap5/best_config_{dataset}.txt.

Datasets: ECG200, RacketSports, CharacterTrajectories
Model   : LR L1 / LASSO (liblinear), LogisticRegressionCV hyperparameter
          selection over Cs = np.logspace(-3, 3, 7)
CV      : StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

Input : data/processed/{ecg200,racketsports,chartraj,natops}_{X,y}_{train,test}.npy
        results/chap5/optimal_N_{ecg200,racketsports,chartraj,natops}.txt
Output: results/chap5/5_3_1_augmentations.csv
        results/chap5/5_3_1_augmentations.png
        results/chap5/best_config_{ecg200,racketsports,chartraj,natops}.txt

Note: signature backend is iisignature (esig hits an internal error on the
CharTraj batch; see 5_2_2 header comment).
"""

import logging
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
NAME = "5_3_1_augmentations"

DATASETS = ["ecg200", "racketsports", "chartraj", "natops"]
DISPLAY_NAMES = {
    "ecg200":       "ECG200",
    "racketsports": "RacketSports",
    "chartraj":     "CharTraj",
    "natops":       "NATOPS",
}

CONFIGS = ["lin+time", "lin+time+lead-lag", "rectilinear+time", "lin+time+cumsum"]
CONFIG_COLORS = {
    "lin+time":          "blue",
    "lin+time+lead-lag": "red",
    "rectilinear+time":  "green",
    "lin+time+cumsum":   "orange",
}

CS = np.logspace(-3, 3, 7)
MAX_FEATURE_DIM = 50_000


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


def load_optimal_N(name: str) -> Optional[int]:
    """Read N* from results/chap5/optimal_N_{name}.txt."""
    val = (RESULTS_DIR / f"optimal_N_{name}.txt").read_text().strip()
    return None if val == "None" else int(val)


def _to_series_list(X: np.ndarray) -> List[np.ndarray]:
    """Normalise input into a list of 2-D (T_i, d) arrays."""
    if isinstance(X, np.ndarray) and X.dtype == object:
        return [x.astype(np.float64) for x in X]
    if X.ndim == 2:
        return [X[i, :, None].astype(np.float64) for i in range(X.shape[0])]
    return [X[i].astype(np.float64) for i in range(X.shape[0])]


# ---------------------------------------------------------------------------
# Path transformations
# ---------------------------------------------------------------------------

def lin_time(x: np.ndarray) -> np.ndarray:
    """Prepend a normalized-time channel to (T, d) → (T, d+1)."""
    T = x.shape[0]
    t = np.linspace(0.0, 1.0, T, dtype=np.float64).reshape(-1, 1)
    return np.hstack([t, x])


def lin_time_lead_lag(x: np.ndarray) -> np.ndarray:
    """lin+time followed by lead-lag with lag=1: (T, d+1) → (T-1, 2*(d+1))."""
    x_time = lin_time(x)
    x_lag = x_time[:-1]
    x_lead = x_time[1:]
    return np.hstack([x_lag, x_lead])


def rectilinear_time(x: np.ndarray) -> np.ndarray:
    """Rectilinear interpolation on the time-augmented path.

    Each transition (X_time[i] → X_time[i+1]) is split into a purely-time
    step then a purely-value step. Total length: 3*(T-1) points (duplicates
    at fold boundaries are harmless — signatures of zero-increment steps
    contribute nothing).
    """
    x_time = lin_time(x)
    T = x_time.shape[0]
    out = []
    for i in range(T - 1):
        step1 = x_time[i].copy()
        step1[0] = x_time[i + 1, 0]  # time jumps to i+1's timestamp
        step2 = x_time[i + 1].copy()  # values jump to i+1's values
        out.extend([x_time[i], step1, step2])
    return np.array(out, dtype=np.float64)


def lin_time_cumsum(x: np.ndarray) -> np.ndarray:
    """Cumulative sum on the raw series then time-augment: (T, d) → (T, d+1)."""
    x_cum = np.cumsum(x, axis=0)
    T = x_cum.shape[0]
    t = np.linspace(0.0, 1.0, T, dtype=np.float64).reshape(-1, 1)
    return np.hstack([t, x_cum])


TRANSFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "lin+time":          lin_time,
    "lin+time+lead-lag": lin_time_lead_lag,
    "rectilinear+time":  rectilinear_time,
    "lin+time+cumsum":   lin_time_cumsum,
}


# ---------------------------------------------------------------------------
# Signature dimensions
# ---------------------------------------------------------------------------

def sig_dim(d: int, depth: int) -> int:
    """Signature feature count (level-0 = 1 already excluded)."""
    if d == 1:
        return depth
    return int((d ** (depth + 1) - d) // (d - 1))


def config_effective_d(config: str, d_base: int) -> int:
    """Effective path dimension after applying the given config to base d."""
    if config == "lin+time+lead-lag":
        return 2 * (d_base + 1)
    return d_base + 1  # lin+time, rectilinear+time, lin+time+cumsum all keep d+1


def compute_signatures(
    series: List[np.ndarray], config: str, depth: int
) -> np.ndarray:
    """Apply the config transform then compute truncated signatures."""
    transform = TRANSFORMS[config]
    return np.stack([iisignature.sig(transform(x), depth) for x in series])


# ---------------------------------------------------------------------------
# CV score + fit/evaluate
# ---------------------------------------------------------------------------

def _cv_score_from_model(
    model: LogisticRegressionCV,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> float:
    """Multiclass-safe CV accuracy (same pattern as 5_2_2)."""
    per_class = list(model.scores_.values())
    is_ovr_multiclass = len(per_class) > 1  # liblinear + multiclass
    if not is_ovr_multiclass:
        mean_over_folds_per_C = np.mean(
            [arr.mean(axis=0) for arr in per_class], axis=0
        )
        return float(mean_over_folds_per_C.max())
    C = float(np.mean(model.C_))
    est = LogisticRegression(
        C=C, penalty="l1", solver="liblinear",
        max_iter=2000, random_state=RANDOM_STATE,
    )
    scores = cross_val_score(est, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean())


def _count_nonzero_features(model: LogisticRegressionCV) -> int:
    """Number of features nonzero in at least one class (union support)."""
    coef = model.coef_
    if coef.ndim == 1:
        active = coef != 0.0
    else:
        active = np.any(coef != 0.0, axis=0)
    return int(active.sum())


def fit_evaluate(
    X_train_f: np.ndarray,
    y_train: np.ndarray,
    X_test_f: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Standardize, fit LogisticRegressionCV L1, return metrics."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train_f)
    Xte = scaler.transform(X_test_f)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model = LogisticRegressionCV(
        Cs=CS,
        cv=cv,
        penalty="l1",
        solver="liblinear",
        scoring="accuracy",
        max_iter=2000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    elapsed = time.perf_counter() - t0
    return {
        "cv_score_mean":       _cv_score_from_model(model, Xtr, y_train, cv),
        "test_accuracy":       accuracy_score(y_test, y_pred),
        "test_f1_macro":       f1_score(y_test, y_pred, average="macro"),
        "n_nonzero_features":  _count_nonzero_features(model),
        "fit_predict_time_s":  elapsed,
    }


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

def _nan_metrics(dim: int) -> Dict[str, float]:
    """Placeholder metrics dict for a skipped or failed config."""
    return {
        "cv_score_mean":      np.nan,
        "test_accuracy":      np.nan,
        "test_f1_macro":      np.nan,
        "n_features":         dim,
        "n_nonzero_features": np.nan,
        "fit_predict_time_s": np.nan,
    }


def run_experiment() -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """Loop over datasets × 4 configs at each dataset's N*."""
    rows = []
    best_configs: Dict[str, Optional[str]] = {}

    for ds in DATASETS:
        logger.info("=== %s ===", DISPLAY_NAMES[ds])
        N = load_optimal_N(ds)
        if N is None:
            logger.warning("  No N* available for %s, skipping dataset", ds)
            best_configs[ds] = None
            continue

        X_train, y_train, X_test, y_test = load_dataset(ds)
        train_list = _to_series_list(X_train)
        test_list = _to_series_list(X_test)
        d_base = train_list[0].shape[1]
        logger.info("  N*=%d, base d=%d", N, d_base)

        best_cv = -np.inf
        best_config: Optional[str] = None

        for config in CONFIGS:
            d_eff = config_effective_d(config, d_base)
            dim = sig_dim(d_eff, N)
            row_stub = {
                "dataset": DISPLAY_NAMES[ds],
                "config":  config,
                "N":       N,
                "n_features": dim,
            }
            if dim > MAX_FEATURE_DIM:
                logger.info(
                    "  %s: skipped (sig_dim=%d > %d)",
                    config, dim, MAX_FEATURE_DIM,
                )
                rows.append({**row_stub, **_nan_metrics(dim)})
                continue

            try:
                t0 = time.perf_counter()
                X_train_f = compute_signatures(train_list, config, N)
                X_test_f = compute_signatures(test_list, config, N)
                sig_t = time.perf_counter() - t0
                logger.info(
                    "  %s (sig_dim=%d, sig_time=%.1fs):",
                    config, X_train_f.shape[1], sig_t,
                )
                metrics = fit_evaluate(X_train_f, y_train, X_test_f, y_test)
                rows.append({**row_stub, **metrics})
                logger.info(
                    "    cv=%.3f  acc=%.3f  f1=%.3f  nnz=%d  fit=%.2fs",
                    metrics["cv_score_mean"], metrics["test_accuracy"],
                    metrics["test_f1_macro"], metrics["n_nonzero_features"],
                    metrics["fit_predict_time_s"],
                )
                if metrics["cv_score_mean"] > best_cv:
                    best_cv = metrics["cv_score_mean"]
                    best_config = config
            except Exception as exc:  # noqa: BLE001 — spec asks for graceful failure
                logger.warning("  %s failed: %s", config, exc)
                rows.append({**row_stub, **_nan_metrics(dim)})

        best_configs[ds] = best_config
        logger.info(
            "  best_config[%s] = %s (CV=%.3f)",
            DISPLAY_NAMES[ds], best_config, best_cv,
        )

    df = pd.DataFrame(rows)[
        ["dataset", "config", "N", "cv_score_mean",
         "test_accuracy", "test_f1_macro",
         "n_features", "n_nonzero_features", "fit_predict_time_s"]
    ]
    return df, best_configs


# ---------------------------------------------------------------------------
# Plot + save
# ---------------------------------------------------------------------------

def plot_results(
    df: pd.DataFrame,
    best_configs: Dict[str, Optional[str]],
) -> plt.Figure:
    """2×n_datasets grid: top row test accuracy, bottom nonzero-feature count."""
    n_cols = len(DATASETS)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 8), squeeze=False)
    fig.suptitle(
        "Axe 2 — Effect of augmentations at N* per dataset",
        fontsize=13, y=0.995,
    )

    for col, ds in enumerate(DATASETS):
        sub = df[df.dataset == DISPLAY_NAMES[ds]]
        configs = sub.config.tolist()
        colors = [CONFIG_COLORS.get(c, "grey") for c in configs]
        x = np.arange(len(configs))
        N = int(sub.N.iloc[0]) if not sub.empty else 0

        # Top: test accuracy
        ax_a = axes[0, col]
        acc = sub.test_accuracy.values
        ax_a.bar(x, acc, color=colors)
        for i, v in enumerate(acc):
            if not np.isnan(v):
                ax_a.text(
                    i, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8,
                )
        ax_a.set_title(f"{DISPLAY_NAMES[ds]} (N*={N})", fontsize=11)
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(configs, rotation=25, ha="right", fontsize=8)
        ax_a.set_ylabel("Test accuracy" if col == 0 else "")
        ax_a.grid(axis="y", linestyle=":", alpha=0.5)
        ax_a.set_axisbelow(True)

        # Bottom: nonzero feature count (log if wide range)
        ax_n = axes[1, col]
        nnz = sub.n_nonzero_features.values
        ax_n.bar(x, nnz, color=colors)
        for i, v in enumerate(nnz):
            if not np.isnan(v):
                ax_n.text(
                    i, v, f"{int(v)}",
                    ha="center", va="bottom", fontsize=8,
                )
        ax_n.set_xticks(x)
        ax_n.set_xticklabels(configs, rotation=25, ha="right", fontsize=8)
        ax_n.set_ylabel("# nonzero features" if col == 0 else "")
        ax_n.grid(axis="y", linestyle=":", alpha=0.5)
        ax_n.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def save_results(
    df: pd.DataFrame,
    fig: plt.Figure,
    best_configs: Dict[str, Optional[str]],
    name: str,
) -> None:
    """Save CSV, PNG, and one best_config txt per dataset."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{name}.csv"
    png_path = RESULTS_DIR / f"{name}.png"
    df.to_csv(csv_path, index=False)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    for ds, cfg in best_configs.items():
        (RESULTS_DIR / f"best_config_{ds}.txt").write_text(
            f"{cfg}\n" if cfg else "None\n"
        )
    logger.info(
        "Results saved to %s, %s, and best_config_{%s}.txt",
        csv_path, png_path, ",".join(DATASETS),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run augmentation grid, save CSV / PNG / best_config txts."""
    logger.info("Running augmentation experiment (N=N* per dataset)...")
    df, best_configs = run_experiment()
    logger.info("Summary:\n%s", df.to_string(index=False))
    fig = plot_results(df, best_configs)
    save_results(df, fig, best_configs, NAME)
    plt.close(fig)

    for ds in DATASETS:
        sub = df[df.dataset == DISPLAY_NAMES[ds]]
        if sub.empty:
            continue
        N = int(sub.N.iloc[0])
        best = best_configs.get(ds)
        if best is None:
            logger.info("%s (N*=%d) — no valid config", DISPLAY_NAMES[ds], N)
            continue
        row = sub[sub.config == best].iloc[0]
        logger.info(
            "%s (N*=%d) — best: %s with acc=%.3f, F1=%.3f, "
            "features=%d, selected=%d",
            DISPLAY_NAMES[ds], N, best,
            row.test_accuracy, row.test_f1_macro,
            int(row.n_features), int(row.n_nonzero_features),
        )
    logger.info("Done.")


if __name__ == "__main__":
    main()
