"""
Experiment 5.4.1 — Multi-view signature + Rough Transformer (RFormer).

Chapter 5, Axe 3: Rough Transformer.
Splits each series into overlapping windows, computes a truncated
signature per window with the `lin+time` augmentation and depth N*
inherited from Axe 1, and feeds the resulting sequence of window
signatures into a small Transformer encoder (RFormer-style). The tokens
seen by self-attention are signatures of local sub-paths rather than
raw samples — following the design of Arroyo et al. (NeurIPS 2024,
AlvaroArroyo/RFormer).

Datasets     : CharTraj, NATOPS
Augmentation : lin+time, hard-coded (see METHODOLOGICAL NOTE below)
Truncation N : loaded from results/chap5/optimal_N_{dataset}.txt (Axe 1)

METHODOLOGICAL NOTE
-------------------
RFormer uses `lin+time` exclusively, independently of the best_config
selected by Axe 2 (5_3_1). This deliberate choice isolates the
architectural contribution of the Transformer from the effect of the
augmentation: measuring both effects at once would prevent us from
concluding on what the attention mechanism itself adds relative to the
signature+LASSO baseline at the same `lin+time` representation.
See thesis section 5.5.4 for the discussion.

Note: this file implements the RFormer idea from scratch (PyTorch
Transformer encoder over signature tokens). It is a pragmatic
reproduction — the core design (signatures-as-tokens) is preserved, but
it does not claim byte-level fidelity to the AlvaroArroyo/RFormer repo.
Swap `RFormer` below for their model class if strict reproduction is
required.

# Colab setup (Pattern A — one script does everything)
# ---------------------------------------------------------------------
#     !git clone https://github.com/<you>/path-signature-benchmark
#     %cd path-signature-benchmark
#     !pip install -q iisignature torch scikit-learn matplotlib pandas
#     !python experiments/chap5/5_4_1_rformer.py
#
#     # commit results back to the repo:
#     !git add results/chap5/5_4_1_rformer.*
#     !git -c user.name="you" -c user.email="you@x" commit \\
#          -m "Axe 3 results"
#     !git push
# ---------------------------------------------------------------------

Input : data/processed/{chartraj,natops}_{X,y}_{train,test}.npy
        results/chap5/optimal_N_{chartraj,natops}.txt
Output: results/chap5/5_4_1_rformer.csv
        results/chap5/5_4_1_rformer.png
"""

import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import iisignature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/chap5")
NAME = "5_4_1_rformer"

DATASETS = ["chartraj", "natops"]
DISPLAY_NAMES = {
    "chartraj": "CharTraj",
    "natops":   "NATOPS",
}

# Note: RFormer uses lin+time exclusively to isolate the architectural
# contribution of the Transformer from the choice of augmentation. This is a
# deliberate methodological decision (see thesis section 5.5.4).
CONFIG = "lin+time"

# Multi-view (windowing) configuration.
N_WINDOWS = 8
WINDOW_OVERLAP = 0.5

# Transformer hyperparameters.
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
N_EPOCHS = 100


# ---------------------------------------------------------------------------
# Path transformations (mirror 5_3_1)
# ---------------------------------------------------------------------------

def lin_time(x: np.ndarray) -> np.ndarray:
    """Prepend a normalized-time channel to (T, d) → (T, d+1)."""
    T = x.shape[0]
    t = np.linspace(0.0, 1.0, T, dtype=np.float64).reshape(-1, 1)
    return np.hstack([t, x.astype(np.float64)])


def lin_time_lead_lag(x: np.ndarray) -> np.ndarray:
    """lin+time then lead-lag with lag=1: (T, d+1) → (T-1, 2*(d+1))."""
    x_time = lin_time(x)
    return np.hstack([x_time[:-1], x_time[1:]])


def rectilinear_time(x: np.ndarray) -> np.ndarray:
    """Rectilinear interpolation on the time-augmented path."""
    x_time = lin_time(x)
    T = x_time.shape[0]
    out = []
    for i in range(T - 1):
        step1 = x_time[i].copy()
        step1[0] = x_time[i + 1, 0]
        out.extend([x_time[i], step1, x_time[i + 1].copy()])
    return np.array(out, dtype=np.float64)


def lin_time_cumsum(x: np.ndarray) -> np.ndarray:
    """Cumulative sum on the raw series, then time-augment: (T, d) → (T, d+1)."""
    x_cum = np.cumsum(x.astype(np.float64), axis=0)
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


def load_optimal_N(name: str) -> int:
    """Read N* from Axe 1 output file. Augmentation is hard-coded to CONFIG."""
    return int((RESULTS_DIR / f"optimal_N_{name}.txt").read_text().strip())


def _to_series_list(X: np.ndarray) -> List[np.ndarray]:
    """Normalise input into a list of 2-D (T_i, d) arrays."""
    if isinstance(X, np.ndarray) and X.dtype == object:
        return [x.astype(np.float64) for x in X]
    if X.ndim == 2:
        return [X[i, :, None].astype(np.float64) for i in range(X.shape[0])]
    return [X[i].astype(np.float64) for i in range(X.shape[0])]


# ---------------------------------------------------------------------------
# Resampling + windowing + multi-view signatures
# ---------------------------------------------------------------------------

def resample_series(x: np.ndarray, T_target: int) -> np.ndarray:
    """Linearly interpolate a (T, d) series to (T_target, d)."""
    T_orig, d = x.shape
    if T_orig == T_target:
        return x.astype(np.float64)
    old_t = np.linspace(0.0, 1.0, T_orig)
    new_t = np.linspace(0.0, 1.0, T_target)
    out = np.empty((T_target, d), dtype=np.float64)
    for i in range(d):
        out[:, i] = np.interp(new_t, old_t, x[:, i])
    return out


def make_windows(
    x: np.ndarray, n_windows: int, overlap: float
) -> List[np.ndarray]:
    """Split a (T, d) series into `n_windows` overlapping windows.

    Window size and stride are chosen so that the first window starts at 0
    and the last window ends at (or before) T-1. Windows have equal length.
    """
    T = x.shape[0]
    # T = w + (n-1) * stride,  stride = w * (1 - overlap)
    #   → w = T / (1 + (n-1) * (1 - overlap))
    window_size = max(2, int(round(T / (1 + (n_windows - 1) * (1 - overlap)))))
    stride = max(1, int(round(window_size * (1 - overlap))))
    windows = []
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        if end > T:
            end = T
            start = max(0, T - window_size)
        windows.append(x[start:end])
    return windows


def multi_view_signature(
    x: np.ndarray,
    config: str,
    depth: int,
    n_windows: int,
    overlap: float,
) -> np.ndarray:
    """Apply the config transform to each window then compute signatures.

    Returns:
        Array of shape (n_windows, sig_dim).
    """
    transform = TRANSFORMS[config]
    windows = make_windows(x, n_windows, overlap)
    return np.stack(
        [iisignature.sig(transform(w), depth) for w in windows]
    )


def prepare_multiview(
    X: np.ndarray,
    config: str,
    depth: int,
    n_windows: int,
    overlap: float,
    T_target: int,
) -> np.ndarray:
    """Build the (n_samples, n_windows, sig_dim) mvsig tensor."""
    series = [resample_series(x, T_target) for x in _to_series_list(X)]
    return np.stack(
        [multi_view_signature(x, config, depth, n_windows, overlap) for x in series]
    )


# ---------------------------------------------------------------------------
# RFormer model
# ---------------------------------------------------------------------------

class RFormer(nn.Module):
    """Small Transformer encoder over a sequence of window signatures.

    Each token is one window's truncated signature (sig_dim). A learned
    positional embedding gives the encoder access to window order. Global
    average pooling over tokens then a linear head produces class logits.

    This is a from-scratch implementation of the "signatures-as-tokens"
    idea from Arroyo et al. (2024). Swap for the official model class if
    strict reproduction is required.
    """

    def __init__(
        self,
        sig_dim: int,
        n_windows: int,
        n_classes: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(sig_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_windows, sig_dim) → (batch, n_classes)."""
        z = self.input_proj(x) + self.pos_embed
        z = self.transformer(z)
        z = z.mean(dim=1)
        return self.head(z)


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def _standardize_mvsig(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Feature-wise standardization computed on train only."""
    _, n_windows, sig_dim = X_train.shape
    flat = X_train.reshape(-1, sig_dim)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8
    Xtr = (X_train - mean) / std
    Xte = (X_test - mean) / std
    return Xtr, Xte


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    device: torch.device,
) -> Dict[str, float]:
    """Train an RFormer for N_EPOCHS and return test-set metrics."""
    _, n_windows, sig_dim = X_train.shape
    Xtr_n, Xte_n = _standardize_mvsig(X_train, X_test)

    Xtr = torch.from_numpy(Xtr_n).float().to(device)
    ytr = torch.from_numpy(y_train).long().to(device)
    Xte = torch.from_numpy(Xte_n).float().to(device)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
    model = RFormer(sig_dim, n_windows, n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "  RFormer: sig_dim=%d, n_windows=%d, n_params=%d",
        sig_dim, n_windows, n_params,
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    t0 = time.perf_counter()
    for epoch in range(N_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                pred = model(Xte).argmax(dim=1).cpu().numpy()
            logger.info(
                "    epoch %3d: test_acc=%.3f, lr=%.2e",
                epoch + 1, accuracy_score(y_test, pred),
                sched.get_last_lr()[0],
            )
    train_time = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(dim=1).cpu().numpy()
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_f1_macro": f1_score(y_test, pred, average="macro"),
        "train_time_s":  train_time,
        "n_params":      n_params,
    }


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

def run_experiment() -> pd.DataFrame:
    """Loop over CharTraj + NATOPS, prep multi-view sigs, train RFormer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cpu":
        logger.warning(
            "No CUDA GPU detected — training will be slow. "
            "Run this script on Colab T4 (see docstring)."
        )

    rows = []
    for ds in DATASETS:
        logger.info("=== %s ===", DISPLAY_NAMES[ds])
        config = CONFIG
        N = load_optimal_N(ds)
        logger.info(
            "  config=%s (hard-coded), N*=%d (from Axe 1)", config, N,
        )

        X_train, y_train, X_test, y_test = load_dataset(ds)
        n_classes = int(np.unique(y_train).size)

        train_list = _to_series_list(X_train)
        T_target = int(np.median([x.shape[0] for x in train_list]))
        logger.info(
            "  n_train=%d, n_test=%d, n_classes=%d, T_target=%d, "
            "n_windows=%d (overlap=%.0f%%)",
            len(train_list), len(_to_series_list(X_test)), n_classes,
            T_target, N_WINDOWS, WINDOW_OVERLAP * 100,
        )

        prep_t0 = time.perf_counter()
        X_train_mv = prepare_multiview(
            X_train, config, N, N_WINDOWS, WINDOW_OVERLAP, T_target
        )
        X_test_mv = prepare_multiview(
            X_test, config, N, N_WINDOWS, WINDOW_OVERLAP, T_target
        )
        prep_time = time.perf_counter() - prep_t0
        logger.info(
            "  mvsig: train=%s, test=%s (prep=%.1fs)",
            X_train_mv.shape, X_test_mv.shape, prep_time,
        )

        metrics = train_and_evaluate(
            X_train_mv, y_train, X_test_mv, y_test, n_classes, device
        )
        rows.append({
            "dataset":         DISPLAY_NAMES[ds],
            "config":          config,
            "N":               N,
            "n_windows":       N_WINDOWS,
            "sig_dim":         X_train_mv.shape[2],
            "test_accuracy":   metrics["test_accuracy"],
            "test_f1_macro":   metrics["test_f1_macro"],
            "n_params":        metrics["n_params"],
            "prep_time_s":     prep_time,
            "train_time_s":    metrics["train_time_s"],
        })
        logger.info(
            "  → test_acc=%.3f, test_f1=%.3f, train_time=%.1fs",
            metrics["test_accuracy"], metrics["test_f1_macro"],
            metrics["train_time_s"],
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot + save
# ---------------------------------------------------------------------------

def plot_results(df: pd.DataFrame) -> plt.Figure:
    """One row × 2 subplots: test accuracy and F1 macro per dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        "Axe 3 — RFormer (multi-view signature + Transformer encoder)",
        fontsize=13, y=1.02,
    )
    colors = ["#1f77b4", "#ff7f0e"]
    for i, metric in enumerate(["test_accuracy", "test_f1_macro"]):
        ax = axes[i]
        names = df["dataset"].values
        vals = df[metric].values
        ax.bar(names, vals, color=colors[:len(names)])
        for j, v in enumerate(vals):
            ax.text(
                j, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10,
            )
        ax.set_ylim(0, 1.05)
        ax.set_title(
            "Test accuracy" if metric == "test_accuracy" else "F1 (macro)",
            fontsize=11,
        )
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def save_results(df: pd.DataFrame, fig: plt.Figure, name: str) -> None:
    """Save results DataFrame to CSV and figure to PNG."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{name}.csv"
    png_path = RESULTS_DIR / f"{name}.png"
    df.to_csv(csv_path, index=False)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    logger.info("Results saved to %s and %s", csv_path, png_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Axe 3 pipeline end-to-end."""
    logger.info("Running Axe 3 (RFormer) experiment...")
    df = run_experiment()
    logger.info("Summary:\n%s", df.to_string(index=False))
    fig = plot_results(df)
    save_results(df, fig, NAME)
    plt.close(fig)
    logger.info("Done.")


if __name__ == "__main__":
    main()
