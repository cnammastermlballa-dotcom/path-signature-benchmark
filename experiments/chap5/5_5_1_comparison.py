"""
Experiment 5.5.1 — Final synthesis across Axes 1-3.

Aggregates the best result per (dataset, method) from each Axe's CSV
into a single comparison table + plot. No new training runs — pure
aggregation, so the script is cheap to rerun as upstream results
evolve.

Methods (ordered by sophistication):
  1. Baseline stats + LR                (5_2_1_baseline_stats.csv)
  2. Signature (lin+time) + LR at N*    (5_2_2_signature_lr.csv at N*)
  3. Signature (best_config) + LASSO    (5_3_1_augmentations.csv)
  4. RFormer (multi-view sig + T-enc)   (5_4_1_rformer.csv)

Datasets: ECG200, RacketSports, CharTraj (methods 1-3), NATOPS (all four).
RFormer entries only exist for CharTraj + NATOPS per Axe 3 scope.

Selection rules per method:
  1. Baseline: best test-accuracy row across LR-L1 / LR-L2.
  2. Sig+LR : highest CV row at the dataset's N* (best model wins the tie).
  3. Sig+aug: highest CV row across configs (matches Axe 2's own
              best_config selection).
  4. RFormer: single row per dataset.

Note: fit_time_s is measured on the CPU for methods 1-3 and on a T4 GPU
for RFormer — comparable as "wall-clock cost you paid", not as a
same-hardware benchmark.

Input : results/chap5/{5_2_1_baseline_stats,5_2_2_signature_lr,
                       5_3_1_augmentations,5_4_1_rformer}.csv
        results/chap5/optimal_N_{ecg200,racketsports,chartraj,natops}.txt

Output: results/chap5/5_5_1_comparison.csv
        results/chap5/5_5_1_comparison.png
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/chap5")
NAME = "5_5_1_comparison"

DATASETS = ["ECG200", "RacketSports", "CharTraj", "NATOPS"]
DATASET_KEYS = {
    "ECG200":       "ecg200",
    "RacketSports": "racketsports",
    "CharTraj":     "chartraj",
    "NATOPS":       "natops",
}

METHODS = [
    "Baseline stats",
    "Sig+LR (N*)",
    "Sig+aug (best_config)",
    "RFormer",
]
METHOD_COLORS = {
    "Baseline stats":        "#7f7f7f",
    "Sig+LR (N*)":           "#1f77b4",
    "Sig+aug (best_config)": "#ff7f0e",
    "RFormer":               "#2ca02c",
}
DATASET_COLORS = {
    "ECG200":       "#d62728",
    "RacketSports": "#9467bd",
    "CharTraj":     "#8c564b",
    "NATOPS":       "#17becf",
}


# ---------------------------------------------------------------------------
# Row extractors
# ---------------------------------------------------------------------------

def load_optimal_N(key: str) -> int:
    return int((RESULTS_DIR / f"optimal_N_{key}.txt").read_text().strip())


def _row_stub(method: str, dataset: str) -> dict:
    return {
        "method":        method,
        "dataset":       dataset,
        "config":        None,
        "N":             None,
        "feature_dim":   None,
        "nnz_features":  None,
        "test_accuracy": None,
        "test_f1_macro": None,
        "fit_time_s":    None,
    }


def baseline_row(df: pd.DataFrame, dataset: str) -> dict:
    """Best (highest test acc) baseline row across LR-L1 / LR-L2."""
    sub = df[df.dataset == dataset]
    row = sub.loc[sub.accuracy.idxmax()]
    stub = _row_stub("Baseline stats", dataset)
    stub.update(
        config=f"stats+{row.model}",
        feature_dim=int(row.feature_dim),
        test_accuracy=float(row.accuracy),
        test_f1_macro=float(row.f1_macro),
        fit_time_s=float(row.time_s),
    )
    return stub


def sig_lr_row(df: pd.DataFrame, dataset: str, N_star: int) -> dict:
    """Highest-CV row at dataset's N* (best-model tie broken by CV)."""
    sub = df[(df.dataset == dataset) & (df.N == N_star)]
    row = sub.loc[sub.cv_score.idxmax()]
    stub = _row_stub("Sig+LR (N*)", dataset)
    stub.update(
        config=f"lin+time+{row.model}",
        N=int(N_star),
        feature_dim=int(row.feature_dim),
        test_accuracy=float(row.accuracy),
        test_f1_macro=float(row.f1_macro),
        fit_time_s=float(row.time_s),
    )
    return stub


def aug_row(df: pd.DataFrame, dataset: str) -> dict:
    """Highest-CV row across the 4 augmentation configs at N*."""
    sub = df[df.dataset == dataset].dropna(subset=["cv_score_mean"])
    row = sub.loc[sub.cv_score_mean.idxmax()]
    stub = _row_stub("Sig+aug (best_config)", dataset)
    stub.update(
        config=str(row.config),
        N=int(row.N),
        feature_dim=int(row.n_features),
        nnz_features=int(row.n_nonzero_features),
        test_accuracy=float(row.test_accuracy),
        test_f1_macro=float(row.test_f1_macro),
        fit_time_s=float(row.fit_predict_time_s),
    )
    return stub


def rformer_row(df: pd.DataFrame, dataset: str) -> Optional[dict]:
    """Single RFormer row per dataset (may be missing for ECG200 / RS)."""
    sub = df[df.dataset == dataset]
    if sub.empty:
        return None
    row = sub.iloc[0]
    stub = _row_stub("RFormer", dataset)
    stub.update(
        config=str(row.config),
        N=int(row.N),
        feature_dim=int(row.n_params),
        test_accuracy=float(row.test_accuracy),
        test_f1_macro=float(row.test_f1_macro),
        fit_time_s=float(row.train_time_s),
    )
    return stub


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def build_comparison() -> pd.DataFrame:
    """Aggregate all methods × all datasets from the four source CSVs."""
    baseline_df = pd.read_csv(RESULTS_DIR / "5_2_1_baseline_stats.csv")
    sig_lr_df = pd.read_csv(RESULTS_DIR / "5_2_2_signature_lr.csv")
    aug_df = pd.read_csv(RESULTS_DIR / "5_3_1_augmentations.csv")
    rformer_df = pd.read_csv(RESULTS_DIR / "5_4_1_rformer.csv")

    rows = []
    for ds in DATASETS:
        rows.append(baseline_row(baseline_df, ds))
        rows.append(sig_lr_row(sig_lr_df, ds, load_optimal_N(DATASET_KEYS[ds])))
        rows.append(aug_row(aug_df, ds))
        r = rformer_row(rformer_df, ds)
        if r is not None:
            rows.append(r)

    return pd.DataFrame(rows)[
        ["dataset", "method", "config", "N",
         "feature_dim", "nnz_features",
         "test_accuracy", "test_f1_macro", "fit_time_s"]
    ]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _values_by_method(
    df: pd.DataFrame, dataset: str, col: str
) -> np.ndarray:
    """Return a value per method for one dataset (NaN if missing)."""
    sub = df[df.dataset == dataset]
    out = []
    for m in METHODS:
        vals = sub[sub.method == m][col].values
        out.append(float(vals[0]) if len(vals) else np.nan)
    return np.array(out)


def plot_results(df: pd.DataFrame) -> plt.Figure:
    """Two panels: accuracy progression per dataset, wall-clock cost per method."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Axes 1-3 synthesis — test accuracy and cost per method",
        fontsize=13, y=1.02,
    )

    # Left: accuracy progression across methods, one line per dataset.
    ax = axes[0]
    x = np.arange(len(METHODS))
    for ds in DATASETS:
        y = _values_by_method(df, ds, "test_accuracy")
        ax.plot(
            x, y, marker="o", linewidth=2,
            color=DATASET_COLORS[ds], label=ds,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Test accuracy", fontsize=11)
    ax.set_title("Accuracy progression across axes", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0.55, 1.02)
    ax.legend(loc="lower right", fontsize=9)

    # Right: fit / train time (log) grouped by method, per dataset.
    ax = axes[1]
    n_methods = len(METHODS)
    width = 0.8 / n_methods
    x = np.arange(len(DATASETS))
    for i, method in enumerate(METHODS):
        y = []
        for ds in DATASETS:
            vals = df[(df.dataset == ds) & (df.method == method)]["fit_time_s"].values
            y.append(float(vals[0]) if len(vals) else np.nan)
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(
            x + offset, y, width,
            label=method, color=METHOD_COLORS[method],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=10)
    ax.set_yscale("log")
    ax.set_ylabel("Fit / train time (s, log)", fontsize=11)
    ax.set_title(
        "Wall-clock cost per method (CPU 1-3, T4 GPU 4)",
        fontsize=11,
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save + main
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame, fig: plt.Figure, name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    fig.savefig(RESULTS_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    logger.info("Saved %s.csv and %s.png", name, name)


def main() -> None:
    logger.info("Aggregating chap5 results across Axes 1-3...")
    df = build_comparison()
    logger.info("Summary:\n%s", df.to_string(index=False))
    fig = plot_results(df)
    save_results(df, fig, NAME)
    plt.close(fig)
    logger.info("Done.")


if __name__ == "__main__":
    main()
