"""
Experiment 5.3.1 — Multi-view signature + Rough Transformer (RFormer).

Chapter 5, Axe 3: Rough Transformer.
From-scratch reproduction of the Rough Transformer (Moreno-Pino et al.,
NeurIPS 2024); it is not the original code. Each series is embedded once
with lin+time (time added once, normalized over the whole series), then
cut into L = N_WINDOWS consecutive windows whose bounds
0 = t_0 < t_1 < ... < t_L = T-1 are proportional to the series' own length
(no resampling, so CharTraj's variable lengths are kept). Token k is the
multi-view signature

    [ S(X)_{[0, t_k]}  (global view)  ,  S(X)_{[t_{k-1}, t_k]}  (local view) ]

truncated at depth N, so token_dim = 2 * sig_dim(d+1, N). The L tokens go
through a small pre-norm Transformer encoder (with a final LayerNorm),
mean-pooled, then a linear head.
No positional embedding is added: the time channel already gives each
token its position (level 1 of the global view contains t_k).

Hyperparameters are fixed a priori (thesis table tab:rformer_hparams),
never tuned; 5 seeds; no early stopping; the test set is only used once,
after training (training logs the training loss only).

LASSO control: GridSearchCV LASSO (same protocol as 5_1_2, liblinear, so
one-vs-rest on these multiclass datasets; scikit-learn stays at 1.7.2
because >= 1.8 raises on it) on the flattened tokens (L * token_dim
features), to separate the effect of the multi-view representation from
that of the Transformer.

Fallback: the fallback depth is the largest N < N* whose flattened control
fits within MAX_FEATURE_DIM (NATOPS, N* = 3: N = 2). If the control at N*
exceeds MAX_FEATURE_DIM or runs out of memory, or if RFormer runs out of
memory at N*, both the control and RFormer (5 seeds) are run at the
fallback depth, and that depth is where they are compared (with LASSO on
the whole-path signature at the same N, read by 5_4_1 from 5_1_2).
RFormer is still trained at N* first, as long as memory allows.

Memory: the multi-view tokens are cast to float32 right after they are
computed. A MemoryError, or a RuntimeError whose message reports an
allocation failure (torch on the CPU raises "DefaultCPUAllocator: can't
allocate memory"), is logged as a WARNING and written as NaN row(s); any
other exception stops the script. At the end the script logs the number of
rows skipped for memory reasons and exits with code 1 (after saving its
outputs) if a required row is missing or NaN: the control and the 5
RFormer seeds at the comparison depth `compare_N` of each dataset.

Datasets     : CharTraj, NATOPS
Augmentation : lin+time, hard-coded (see METHODOLOGICAL NOTE below)
Truncation N : LASSO N* from results/chap5/optimal_N_{dataset}_LR-L1.txt

METHODOLOGICAL NOTE
-------------------
RFormer uses `lin+time` exclusively, independently of the best_config
selected by Axe 2 (5_2_1). This deliberate choice isolates the
architectural contribution of the Transformer from the effect of the
augmentation: measuring both effects at once would prevent us from
concluding on what the attention mechanism itself adds relative to the
signature+LASSO baseline at the same `lin+time` representation.
See thesis section 5.5.4 for the discussion.

Bootstrap CIs and McNemar tests are computed by 5_4_1 from the saved
per-sample predictions.

Hardware: the thesis results are trained on the CPU (Intel i5-7300U, the
same machine as the linear models), so the costs of all methods are
comparable; the script uses a GPU only if torch detects one. It logs
os.cpu_count() and torch.get_num_threads().

Control rows also carry the diagnostics n_convergence_warnings (CV folds ×
C grid + refit) and c_at_grid_edge (selected C is the smallest or largest
in CS); they are empty for RFormer rows.

Input : data/processed/{chartraj,natops}_{X,y}_{train,test}.npy
        results/chap5/optimal_N_{chartraj,natops}_LR-L1.txt
Output: results/chap5/5_3_1_rformer.csv
            (one row per (dataset, model, N, seed); the control has no seed;
            N_star, fallback_N, compare_N, control_fallback and
            rformer_oom_at_N_star are repeated on every row of a dataset)
        results/chap5/5_3_1_rformer.png
        results/chap5/5_3_1_rformer_predictions.csv
            (per-sample test predictions, every row of the CSV)
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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from embeddings import TRANSFORMS
from runinfo import log_versions

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
NAME = "5_3_1_rformer"

DATASETS = ["chartraj", "natops"]
DISPLAY_NAMES = {
    "chartraj": "CharTraj",
    "natops":   "NATOPS",
}

# Note: RFormer uses lin+time exclusively to isolate the architectural
# contribution of the Transformer from the choice of augmentation. This is a
# deliberate methodological decision (see thesis section 5.5.4).
CONFIG = "lin+time"

# Multi-view windowing: L consecutive windows, bounds proportional to length.
N_WINDOWS = 8

# LASSO control (same protocol as 5_1_2).
CS = np.logspace(-3, 3, 7)
MAX_FEATURE_DIM = 50_000
# Substrings of a RuntimeError message that report an allocation failure
# (torch CPU allocator, CUDA, or the OS); compared in lower case.
OOM_MESSAGES = (
    "not enough memory", "can't allocate memory",
    "cannot allocate memory", "out of memory",
)

# Transformer hyperparameters, fixed a priori (thesis tab:rformer_hparams).
D_MODEL = 20
N_HEADS = 2
N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
N_EPOCHS = 100
SEEDS = [0, 1, 2, 3, 4]

MODEL_LABELS = {"RFormer": "RFormer", "LR-L1-control": "LASSO de contrôle"}


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
    """Read the LASSO N* written by 5_1_2. Augmentation is hard-coded to CONFIG."""
    path = RESULTS_DIR / f"optimal_N_{name}_LR-L1.txt"
    value = path.read_text().strip()
    if not value.isdigit():
        raise ValueError(
            f"{path} contains {value!r}, not a depth: 5_1_2 found no valid LASSO N* "
            f"for {name}. Rerun 5_1_2 (axe1.sh) and check its log."
        )
    return int(value)


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


def control_dim(d: int, depth: int) -> int:
    """Flattened multi-view dimension: L tokens × [global, local] signatures."""
    return N_WINDOWS * 2 * sig_dim(d, depth)


def fallback_depth(d: int, N_star: int) -> Optional[int]:
    """Largest N < N* whose flattened control fits within MAX_FEATURE_DIM (None if none)."""
    fits = [N for N in range(1, N_star) if control_dim(d, N) <= MAX_FEATURE_DIM]
    return max(fits) if fits else None


def _is_oom(exc: BaseException) -> bool:
    """MemoryError, or a RuntimeError whose message reports an allocation failure."""
    if isinstance(exc, MemoryError):
        return True
    return isinstance(exc, RuntimeError) and any(
        m in str(exc).lower() for m in OOM_MESSAGES
    )


# ---------------------------------------------------------------------------
# Multi-view signatures
# ---------------------------------------------------------------------------

def window_bounds(T: int, n_windows: int) -> np.ndarray:
    """Indices 0 = t_0 < ... < t_L = T-1, proportional to the series length."""
    bounds = np.rint(np.linspace(0, T - 1, n_windows + 1)).astype(int)
    if np.any(np.diff(bounds) <= 0):
        raise ValueError(f"Series of length {T} too short for {n_windows} windows")
    return bounds


def multi_view_signature(x: np.ndarray, depth: int, n_windows: int) -> np.ndarray:
    """Tokens [global sig over [0, t_k], local sig over [t_{k-1}, t_k]], k = 1..L.

    Time is added once to the whole series (CONFIG transform), before windowing.

    Returns:
        Array of shape (n_windows, 2 * sig_dim).
    """
    path = TRANSFORMS[CONFIG](x)
    bounds = window_bounds(path.shape[0], n_windows)
    # format=2: row i is the signature of path[0 : i+2], i.e. of [0, t] at t = i+1.
    prefix = iisignature.sig(path, depth, 2)
    tokens = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        global_view = prefix[b - 1]
        local_view = iisignature.sig(path[a:b + 1], depth)
        tokens.append(np.concatenate([global_view, local_view]))
    return np.stack(tokens)


def prepare_multiview(X: np.ndarray, depth: int, n_windows: int) -> np.ndarray:
    """Build the (n_samples, n_windows, 2 * sig_dim) multi-view tensor, in float32.

    Each series' tokens are cast to float32 as soon as they are computed, so
    the full tensor never exists in float64 (half the memory).
    """
    return np.stack([
        multi_view_signature(x, depth, n_windows).astype(np.float32)
        for x in _to_series_list(X)
    ])


# ---------------------------------------------------------------------------
# LASSO control
# ---------------------------------------------------------------------------

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


def fit_lasso(
    X_train_f: np.ndarray,
    y_train: np.ndarray,
    X_test_f: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[dict, np.ndarray]:
    """Grid-search C on Pipeline(scaler, LASSO) by CV; evaluate best_estimator_ on test."""
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
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1_macro": f1_score(y_test, y_pred, average="macro"),
        "cv_score":      float(search.best_score_),
        "cv_std":        float(search.cv_results_["std_test_accuracy"][search.best_index_]),
        "C_best":        float(search.best_params_["clf__C"]),
        "fit_time_s":    elapsed,
        **_fit_diagnostics(search),
    }
    return metrics, y_pred


def warm_up() -> None:
    """Untimed dummy fit, so the first timed fit does not pay joblib's worker start-up."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(50, 3))
    y = np.arange(50) % 2
    fit_lasso(X, y, X, y)


# ---------------------------------------------------------------------------
# RFormer model
# ---------------------------------------------------------------------------

class RFormer(nn.Module):
    """Small Transformer encoder over a sequence of multi-view signature tokens.

    Each token is one window's [global, local] truncated signature. No
    positional embedding: the time channel already encodes the position.
    Pre-norm encoder layers, then a final LayerNorm.
    Mean pooling over tokens, then a linear head, produces class logits.
    """

    def __init__(
        self,
        token_dim: int,
        n_classes: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(token_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        # Pre-norm layers leave the residual stream unnormalised: a final
        # LayerNorm closes the encoder, as in the standard pre-LN Transformer.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_windows, token_dim) → (batch, n_classes)."""
        z = self.transformer(self.input_proj(x))
        return self.head(z.mean(dim=1))


def _standardize_mvsig(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Feature-wise standardization computed on train only (pooled over windows)."""
    token_dim = X_train.shape[2]
    flat = X_train.reshape(-1, token_dim)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def train_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    seed: int,
    device: torch.device,
) -> Tuple[dict, np.ndarray]:
    """Train one RFormer for N_EPOCHS (no test evaluation), then predict the test set."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr_n, Xte_n = _standardize_mvsig(X_train, X_test)
    Xtr = torch.from_numpy(Xtr_n).float().to(device)
    ytr = torch.from_numpy(y_train).long().to(device)
    Xte = torch.from_numpy(Xte_n).float().to(device)

    loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = RFormer(X_train.shape[2], n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    t0 = time.perf_counter()
    for epoch in range(N_EPOCHS):
        model.train()
        total, count = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            count += len(yb)
        sched.step()
        train_loss = total / count
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info("      epoch %3d: train_loss=%.4f", epoch + 1, train_loss)
    model.eval()
    with torch.no_grad():
        y_pred = model(Xte).argmax(dim=1).cpu().numpy()
    fit_time = time.perf_counter() - t0  # training + test predict
    return {
        "n_params":         n_params,
        "final_train_loss": train_loss,
        "fit_time_s":       fit_time,
    }, y_pred


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

def _pred_frame(
    ds: str, model: str, N: int, seed, y_test: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame({
        "dataset": DISPLAY_NAMES[ds],
        "model":   model,
        "N":       N,
        "seed":    pd.array([seed] * len(y_test), dtype="Int64"),
        "sample":  np.arange(len(y_test)),
        "y_true":  y_test,
        "y_pred":  y_pred,
    })


RESULT_COLUMNS = [
    "dataset", "model", "N", "N_star", "fallback_N", "compare_N",
    "control_fallback", "rformer_oom_at_N_star", "seed",
    "n_windows", "token_dim", "n_features", "n_params",
    "cv_score", "cv_std", "C_best", "final_train_loss",
    "test_accuracy", "test_f1_macro", "feat_time_s", "fit_time_s", "device",
    "n_convergence_warnings", "c_at_grid_edge",
]


def _nan_row(common: dict, model: str, seed, device: str) -> dict:
    """Row of a fit skipped for memory reasons: metrics left NaN."""
    return {**common, "model": model, "seed": seed, "device": device,
            "test_accuracy": np.nan, "test_f1_macro": np.nan, "fit_time_s": np.nan}


def _run_at_depth(
    ds: str,
    N: int,
    data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    n_classes: int,
    d: int,
    do_control: bool,
    do_rformer: bool,
    device: torch.device,
    rows: List[dict],
    pred_rows: List[pd.DataFrame],
) -> Tuple[bool, bool]:
    """Multi-view tokens at depth N, then the LASSO control and/or RFormer × seeds.

    Appends one row per fit (NaN metrics when it ran out of memory) and the
    predictions of each successful fit. The token arrays are local, so they
    are freed when this returns.

    Returns:
        (control ran out of memory, RFormer ran out of memory for some seed).
    """
    X_train, y_train, X_test, y_test = data
    common = {
        "dataset":     DISPLAY_NAMES[ds],
        "N":           N,
        "n_windows":   N_WINDOWS,
        "token_dim":   2 * sig_dim(d, N),
        "n_features":  control_dim(d, N),
        "feat_time_s": np.nan,
    }
    try:
        t0 = time.perf_counter()
        Xtr_mv = prepare_multiview(X_train, N, N_WINDOWS)
        Xte_mv = prepare_multiview(X_test, N, N_WINDOWS)
        common["feat_time_s"] = time.perf_counter() - t0
    except (MemoryError, RuntimeError) as exc:
        if not _is_oom(exc):
            raise
        logger.warning(
            "  N=%d skipped: out of memory in the multi-view signatures (%s); "
            "rows written as NaN", N, type(exc).__name__,
        )
        if do_control:
            rows.append(_nan_row(common, "LR-L1-control", None, "cpu"))
        if do_rformer:
            rows.extend(_nan_row(common, "RFormer", seed, device.type) for seed in SEEDS)
        return do_control, do_rformer
    logger.info(
        "  N=%d: tokens train=%s, test=%s, float32 %.2f GB (sig=%.1fs)",
        N, Xtr_mv.shape, Xte_mv.shape, (Xtr_mv.nbytes + Xte_mv.nbytes) / 1e9,
        common["feat_time_s"],
    )

    control_oom = False
    if do_control:
        try:
            metrics, y_pred = fit_lasso(
                Xtr_mv.reshape(len(Xtr_mv), -1), y_train,
                Xte_mv.reshape(len(Xte_mv), -1), y_test,
            )
        except (MemoryError, RuntimeError) as exc:
            if not _is_oom(exc):
                raise
            control_oom = True
            logger.warning("    LASSO control N=%d skipped: out of memory (%s)",
                           N, type(exc).__name__)
            rows.append(_nan_row(common, "LR-L1-control", None, "cpu"))
        else:
            rows.append({**common, "model": "LR-L1-control", "seed": None,
                         "device": "cpu", **metrics})
            pred_rows.append(_pred_frame(ds, "LR-L1-control", N, None, y_test, y_pred))
            logger.info(
                "    LASSO control: acc=%.3f  cv=%.3f±%.3f  C=%g  time=%.1fs  conv_warn=%d",
                metrics["test_accuracy"], metrics["cv_score"], metrics["cv_std"],
                metrics["C_best"], metrics["fit_time_s"],
                metrics["n_convergence_warnings"],
            )

    rformer_oom = False
    if do_rformer:
        for seed in SEEDS:
            logger.info("    RFormer N=%d seed %d", N, seed)
            try:
                metrics, y_pred = train_and_predict(
                    Xtr_mv, y_train, Xte_mv, n_classes, seed, device
                )
            except (MemoryError, RuntimeError) as exc:
                if not _is_oom(exc):
                    raise
                rformer_oom = True
                logger.warning("    RFormer N=%d seed %d skipped: out of memory (%s)",
                               N, seed, type(exc).__name__)
                rows.append(_nan_row(common, "RFormer", seed, device.type))
                continue
            metrics["test_accuracy"] = accuracy_score(y_test, y_pred)
            metrics["test_f1_macro"] = f1_score(y_test, y_pred, average="macro")
            rows.append({**common, "model": "RFormer", "seed": seed,
                         "device": device.type, **metrics})
            pred_rows.append(_pred_frame(ds, "RFormer", N, seed, y_test, y_pred))
            logger.info(
                "    → seed %d: acc=%.3f  f1=%.3f  n_params=%d  train=%.1fs",
                seed, metrics["test_accuracy"], metrics["test_f1_macro"],
                metrics["n_params"], metrics["fit_time_s"],
            )
    return control_oom, rformer_oom


def run_experiment() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loop over CharTraj + NATOPS: multi-view signatures, LASSO control, RFormer × seeds.

    Per dataset: RFormer is trained at N*, and the control too if its
    flattened dimension fits within MAX_FEATURE_DIM. If the control does not
    fit (or runs out of memory), or RFormer runs out of memory, both are run
    again at the fallback depth, which becomes the comparison depth compare_N.

    Returns:
        df: One row per (dataset, model, N, seed); the control has no seed.
        preds: Per-sample test predictions for every successful fit.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Device: %s%s", device,
        f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda"
        else f" ({torch.get_num_threads()} torch threads)",
    )

    rows: List[dict] = []
    pred_rows: List[pd.DataFrame] = []
    for ds in DATASETS:
        logger.info("=== %s ===", DISPLAY_NAMES[ds])
        N_star = load_optimal_N(ds)
        data = load_dataset(ds)
        n_classes = int(np.unique(data[1]).size)
        d = _to_series_list(data[0][:1])[0].shape[1] + 1   # +1: time channel
        fb_N = fallback_depth(d, N_star)
        first_row = len(rows)

        control_fallback = control_dim(d, N_star) > MAX_FEATURE_DIM
        logger.info(
            "  config=%s (hard-coded), LASSO N*=%d, n_classes=%d, L=%d windows, "
            "control dim at N*=%d, fallback N=%s",
            CONFIG, N_star, n_classes, N_WINDOWS, control_dim(d, N_star), fb_N,
        )
        if control_fallback:
            logger.info(
                "  control at N*: %d features > MAX_FEATURE_DIM=%d → control at N=%s",
                control_dim(d, N_star), MAX_FEATURE_DIM, fb_N,
            )

        control_oom, rformer_oom = _run_at_depth(
            ds, N_star, data, n_classes, d, not control_fallback, True,
            device, rows, pred_rows,
        )
        control_fallback = control_fallback or control_oom
        compare_N: Optional[int] = N_star
        if control_fallback or rformer_oom:
            compare_N = fb_N
            if fb_N is None:
                logger.error(
                    "  no fallback depth: no N < N*=%d fits within MAX_FEATURE_DIM", N_star
                )
            else:
                logger.info(
                    "  fallback: control and RFormer at N=%d (control fallback=%s, "
                    "RFormer out of memory at N*=%s)", fb_N, control_fallback, rformer_oom,
                )
                _run_at_depth(
                    ds, fb_N, data, n_classes, d, True, True, device, rows, pred_rows,
                )

        for r in rows[first_row:]:
            r.update(
                N_star=N_star, fallback_N=fb_N, compare_N=compare_N,
                control_fallback=control_fallback, rformer_oom_at_N_star=rformer_oom,
            )

    df = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    for col in ("seed", "fallback_N", "compare_N", "n_params",
                "n_convergence_warnings", "c_at_grid_edge"):
        df[col] = pd.to_numeric(df[col]).astype("Int64")
    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    return df, preds


def required_row_problems(df: pd.DataFrame) -> List[str]:
    """Required rows missing or NaN: the control and the 5 RFormer seeds at compare_N."""
    problems = []
    for ds in DATASETS:
        sub = df[df.dataset == DISPLAY_NAMES[ds]]
        if sub.empty or sub.compare_N.isna().all():
            problems.append(f"{DISPLAY_NAMES[ds]}: no comparison depth (no valid fallback N)")
            continue
        N = int(sub.compare_N.iloc[0])
        at_N = sub[sub.N == N]
        ctrl = at_N[at_N.model == "LR-L1-control"]
        if len(ctrl) != 1 or ctrl.test_accuracy.isna().any():
            problems.append(f"{DISPLAY_NAMES[ds]}: no valid LASSO control row at N={N}")
        rf = at_N[(at_N.model == "RFormer") & at_N.test_accuracy.notna()]
        if sorted(rf.seed.dropna().astype(int)) != SEEDS:
            problems.append(
                f"{DISPLAY_NAMES[ds]}: RFormer at N={N} has valid rows for seeds "
                f"{sorted(rf.seed.dropna().astype(int))}, expected {SEEDS}"
            )
    return problems


# ---------------------------------------------------------------------------
# Plot + save
# ---------------------------------------------------------------------------

def plot_results(df: pd.DataFrame) -> plt.Figure:
    """Test accuracy per dataset: RFormer (mean ± std, seeds as dots) and the LASSO control.

    Rows skipped for memory reasons (NaN) are not plotted.
    """
    datasets = list(dict.fromkeys(df["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5),
                             squeeze=False)
    fig.suptitle(
        "Axe 3 — RFormer (signature multi-vues + Transformer) vs LASSO de contrôle",
        fontsize=13, y=1.02,
    )
    for ax, ds in zip(axes.ravel(), datasets):
        sub = df[(df.dataset == ds) & df.test_accuracy.notna()]   # skip out-of-memory rows
        bars = []
        for N in sorted(sub[sub.model == "RFormer"].N.unique()):
            acc = sub[(sub.model == "RFormer") & (sub.N == N)].test_accuracy.values
            bars.append((f"RFormer\nN={N}", acc, "#1f77b4"))
        for _, r in sub[sub.model == "LR-L1-control"].iterrows():
            bars.append((f"LASSO de contrôle\nN={r.N}", np.array([r.test_accuracy]), "#ff7f0e"))
        for i, (label, acc, color) in enumerate(bars):
            mean = acc.mean()
            std = acc.std(ddof=1) if len(acc) > 1 else 0.0
            ax.bar(i, mean, 0.6, color=color, alpha=0.8,
                   yerr=std if len(acc) > 1 else None, capsize=4)
            if len(acc) > 1:
                ax.scatter(np.full(len(acc), i), acc, color="black", s=12, zorder=3)
            ax.text(i, max(mean + std, acc.max()) + 0.003, f"{mean:.3f}" + (f"\n±{std:.3f}" if len(acc) > 1 else ""),
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(ds, fontsize=11)
        if not bars:   # every fit of this dataset ran out of memory
            ax.text(0.5, 0.5, "aucun résultat\n(mémoire insuffisante)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            continue
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([b[0] for b in bars], fontsize=9)
        lo = min(acc.min() for _, acc, _ in bars)
        ax.set_ylim(max(0.0, lo - 0.1), 1.02)
        ax.set_ylabel("Taux de bonne classification (test)")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout()
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Axe 3 pipeline end-to-end."""
    log_versions(logger)
    logger.info("Running Axe 3 (RFormer) experiment...")
    logger.info("CPU cores available: %s (os.cpu_count()); GridSearchCV n_jobs=-1", os.cpu_count())
    logger.info("torch threads: %d (torch.get_num_threads())", torch.get_num_threads())
    t0 = time.perf_counter()
    warm_up()
    df, preds = run_experiment()
    logger.info("Summary:\n%s", df.drop(columns=["device"]).to_string(index=False))
    logger.info("Total runtime: %.1f min", (time.perf_counter() - t0) / 60)
    fig = plot_results(df)
    save_results(df, preds, fig, NAME)
    plt.close(fig)
    logger.info(
        "Rows skipped for memory reasons (NaN in the CSV): %d",
        int(df.test_accuracy.isna().sum()),
    )
    problems = required_row_problems(df)
    if problems:
        for msg in problems:
            logger.error("Required rows incomplete: %s", msg)
        sys.exit(1)
    logger.info("Done.")


if __name__ == "__main__":
    main()
