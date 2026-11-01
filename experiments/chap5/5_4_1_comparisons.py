"""
Experiment 5.4.1 — Final comparison across Axes 1-3, with uncertainty.

Pure aggregation of the earlier CSVs and per-sample test predictions: no
model is fitted, and nothing is selected here. Every N, configuration and
model comes from the CV selections of 5_1_2 / 5_2_1 (via their files) and
from 5_3_1's comparison depth; test accuracy is only reported.

Rows of 5_4_1_comparisons.csv (one per axis × dataset × method):
  Axe 1 (all datasets): baseline Ridge, baseline LASSO (5_1_1); signature
        Ridge at its N*, signature LASSO at its N* (5_1_2, lin+time).
  Axe 2 (all datasets): the four configurations under LASSO, each at its own
        CV-selected N* (5_2_1); `selected` marks the CV-selected one.
  Axe 3 (CharTraj, NATOPS): signature LASSO at its N* and Ridge at its N*
        (descriptive reference only), the LASSO control at compare_N,
        RFormer at N* (one row per seed + one seed-averaged row), and, when
        5_3_1 fell back (compare_N < N*), RFormer and signature LASSO at
        compare_N as well.

Uncertainty, for every row: 95% bootstrap CI on test accuracy (N_BOOT = 2000
resamples of the test set, percentile method, fixed seed BOOT_SEED). The
resampled indices are drawn once per dataset and shared by all its rows.
RFormer: each seed row has its own CI; the seed-averaged row has the CI of
the seed-averaged accuracy (at each draw, every seed's accuracy is computed
on the same indices, then averaged over seeds), next to mean ± std.

McNemar tests (exact: two-sided binomial test on the discordant pairs, p = 1
when there are none), only for the planned comparisons, always at the same N
on both sides; one Holm family per axis (raw and adjusted p-values):
  Axe 1: signature Ridge (N*) vs baseline Ridge, signature LASSO (N*) vs
         baseline LASSO, per dataset.
  Axe 2: selected configuration vs lin+time, per dataset; "not_applicable"
         (no p-value, outside the Holm family) when lin+time is selected.
  Axe 3: RFormer@N* vs LASSO@N* (per seed) and control@compare_N vs
         LASSO@compare_N; if compare_N < N*, also RFormer@compare_N vs
         LASSO@compare_N (per seed). The RFormer rows report how many of the
         5 seeds give p < 0.05, on raw and on Holm-adjusted p-values.
A test whose input row was skipped for memory reasons (NaN upstream) is
written with status "missing" and left out of the Holm family.

Cost of a method = feat_time_s + fit_time_s (total_time_s); feat_time_s is
never summed across models or seeds. Every row also carries n_star_at_cap,
cap_reason, c_at_grid_edge and n_convergence_warnings from its source row
(Axe 3 rows take the cap flags of the 5_1_2 LASSO row at the same N; the
RFormer rows have no grid-search diagnostics).

Rows skipped upstream for memory reasons are kept with status
"out_of_memory" and NaN metrics, never dropped. A missing input (file, N*
row, selected row or its predictions) stops the script with exit code 1.

Input : results/chap5/{5_1_1_baseline_stats,5_1_2_signature_lr,
                       5_2_1_augmentations,5_3_1_rformer}.csv
        results/chap5/{same names}_predictions.csv
        results/chap5/optimal_N_{ecg200,racketsports,chartraj,natops}_{LR-L2,LR-L1}.txt
Output: results/chap5/5_4_1_comparisons.csv   (one row per axis × dataset × method)
        results/chap5/5_4_1_tests.csv         (one row per McNemar test)
        results/chap5/5_4_1_comparisons.png
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import binomtest

from runinfo import log_versions

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/chap5")
NAME = "5_4_1_comparisons"

DATASETS = ["ecg200", "racketsports", "chartraj", "natops"]
AXE3_DATASETS = ["chartraj", "natops"]
DISPLAY_NAMES = {
    "ecg200":       "ECG200",
    "racketsports": "RacketSports",
    "chartraj":     "CharTraj",
    "natops":       "NATOPS",
}
MODELS = ["LR-L2", "LR-L1"]
MODEL_LABELS = {"LR-L2": "Ridge", "LR-L1": "LASSO"}
# Same order as 5_2_1 (ties between configurations go to the earlier one).
CONFIGS = ["lin+time", "lin+time+lead-lag", "rectilinear+time", "lin+time+cumsum"]
SEEDS = [0, 1, 2, 3, 4]

N_BOOT = 2000
BOOT_SEED = 42
CI_LEVEL = 0.95
ALPHA = 0.05

AXIS_COLORS = {"Axe 1": "#1f77b4", "Axe 2": "#ff7f0e", "Axe 3": "#2ca02c"}

COMPARISON_COLUMNS = [
    "axis", "dataset", "method", "model", "config", "N", "seed", "n_seeds",
    "selected", "status", "n_features", "n_nonzero_features", "cv_score",
    "test_accuracy", "test_accuracy_std", "ci_low", "ci_high", "test_f1_macro",
    "feat_time_s", "fit_time_s", "total_time_s",
    "n_star_at_cap", "cap_reason", "c_at_grid_edge", "n_convergence_warnings",
    "mcnemar_vs", "p_raw", "p_holm", "seeds_p05_raw", "seeds_p05_holm",
]
TEST_COLUMNS = [
    "axis", "dataset", "method_a", "N_a", "method_b", "N_b", "seed", "status",
    "n_test", "b", "c", "p_raw", "p_holm", "significant_raw", "significant_holm",
]


class MissingInput(Exception):
    """An input the comparison needs is missing: stop with exit code 1."""


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _read_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        raise MissingInput(f"{path} not found: run the stage that writes it first")
    return pd.read_csv(path)


def load_optimal_N(ds: str, model: str) -> int:
    """N* of one model written by 5_1_2 (optimal_N_{ds}_{model}.txt)."""
    path = RESULTS_DIR / f"optimal_N_{ds}_{model}.txt"
    if not path.exists():
        raise MissingInput(f"{path} not found: run axe1.sh (5_1_2) first")
    value = path.read_text().strip()
    if not value.isdigit():
        raise MissingInput(f"{path} contains {value!r}: 5_1_2 found no valid N*")
    return int(value)


def load_inputs() -> Dict[str, pd.DataFrame]:
    """The four result CSVs and their per-sample prediction CSVs."""
    names = {
        "base": "5_1_1_baseline_stats",
        "sig":  "5_1_2_signature_lr",
        "aug":  "5_2_1_augmentations",
        "rf":   "5_3_1_rformer",
    }
    out = {}
    for key, name in names.items():
        out[key] = _read_csv(f"{name}.csv")
        out[f"{key}_pred"] = _read_csv(f"{name}_predictions.csv")
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def boot_indices(n: int) -> np.ndarray:
    """N_BOOT resamples (with replacement) of n test indices, fixed seed."""
    return np.random.default_rng(BOOT_SEED).integers(0, n, size=(N_BOOT, n))


def boot_ci(correct: np.ndarray, idx: np.ndarray) -> Tuple[float, float]:
    """Percentile CI of the accuracy; `correct` may be a seed-averaged float vector."""
    accs = correct[idx].mean(axis=1)
    tail = 100 * (1 - CI_LEVEL) / 2
    lo, hi = np.percentile(accs, [tail, 100 - tail])
    return float(lo), float(hi)


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> Tuple[int, int, float]:
    """Exact McNemar test: b = A right & B wrong, c = A wrong & B right."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return b, c, p


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as p)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

class Builder:
    """Collects comparison rows, their per-sample correctness, and the tests."""

    def __init__(self, inputs: Dict[str, pd.DataFrame]) -> None:
        self.inp = inputs
        self.rows: List[dict] = []
        self.correct: Dict[int, np.ndarray] = {}   # row index → correctness vector
        self.tests: List[dict] = []
        self.test_rows: List[int] = []              # row index of method_a per test
        self.y_ref: Dict[str, np.ndarray] = {}
        self.idx: Dict[str, np.ndarray] = {}

    # -- predictions ------------------------------------------------------
    def _y_ref(self, ds: str) -> np.ndarray:
        """Test labels of a dataset, from the baseline predictions (reference order)."""
        if ds not in self.y_ref:
            p = self.inp["base_pred"]
            sub = p[(p.dataset == DISPLAY_NAMES[ds]) & (p.model == "LR-L2")]
            if sub.empty:
                raise MissingInput(f"no 5_1_1 predictions for {DISPLAY_NAMES[ds]}")
            self.y_ref[ds] = sub.sort_values("sample").y_true.to_numpy()
            self.idx[ds] = boot_indices(len(self.y_ref[ds]))
        return self.y_ref[ds]

    def _correctness(self, key: str, ds: str, what: str, **match) -> np.ndarray:
        """Per-sample correctness of one fit, checked against the reference labels."""
        p = self.inp[key]
        mask = p.dataset == DISPLAY_NAMES[ds]
        for col, val in match.items():
            mask &= p[col] == val
        sub = p[mask].sort_values("sample")
        y_ref = self._y_ref(ds)
        if len(sub) != len(y_ref):
            raise MissingInput(
                f"predictions of {what} ({DISPLAY_NAMES[ds]}): {len(sub)} rows, "
                f"expected {len(y_ref)}"
            )
        if not np.array_equal(sub.y_true.to_numpy(), y_ref):
            raise MissingInput(f"predictions of {what} ({DISPLAY_NAMES[ds]}): y_true differs")
        return sub.y_pred.to_numpy() == y_ref

    # -- rows ---------------------------------------------------------------
    def add(self, correct: Optional[np.ndarray], ds: str,
            ci: Optional[Tuple[float, float]] = None, **fields) -> int:
        """Append one comparison row.

        The CI comes from `correct` (per-sample correctness), or is given as
        `ci` (seed-averaged RFormer row). With neither, the row was skipped
        upstream for memory reasons: its metrics are NaN.
        """
        row = {c: np.nan for c in COMPARISON_COLUMNS}
        row.update(dataset=DISPLAY_NAMES[ds], status="ok", cap_reason="")
        row.update(fields)
        if correct is not None:
            self._y_ref(ds)
            ci = boot_ci(correct.astype(float), self.idx[ds])
        if ci is None:
            if row["status"] == "ok":
                row["status"] = "out_of_memory"
            for col in ("test_accuracy", "test_accuracy_std", "ci_low", "ci_high",
                        "test_f1_macro"):
                row[col] = np.nan
        else:
            row["ci_low"], row["ci_high"] = ci
        row["total_time_s"] = row["feat_time_s"] + row["fit_time_s"]
        self.rows.append(row)
        i = len(self.rows) - 1
        if correct is not None:
            self.correct[i] = correct
        return i

    def test(self, axis: str, ds: str, i_a: Optional[int], i_b: Optional[int],
             seed=None, status: Optional[str] = None) -> None:
        """Record one McNemar test between rows i_a and i_b (same N on both sides)."""
        ra = self.rows[i_a] if i_a is not None else {}
        rb = self.rows[i_b] if i_b is not None else {}
        t = {c: np.nan for c in TEST_COLUMNS}
        t.update(
            axis=axis, dataset=DISPLAY_NAMES[ds],
            method_a=ra.get("method", ""), N_a=ra.get("N", np.nan),
            method_b=rb.get("method", ""), N_b=rb.get("N", np.nan), seed=seed,
            n_test=len(self._y_ref(ds)),
        )
        if status is None:
            if i_a in self.correct and i_b in self.correct:
                t["b"], t["c"], t["p_raw"] = mcnemar_exact(self.correct[i_a], self.correct[i_b])
                status = "tested"
            else:
                status = "missing"   # an input row was skipped for memory reasons
        t["status"] = status
        self.tests.append(t)
        self.test_rows.append(i_a)

    # -- source-row helpers --------------------------------------------------
    @staticmethod
    def _diag(r: pd.Series) -> dict:
        return {
            "c_at_grid_edge":         r.get("c_at_grid_edge", np.nan),
            "n_convergence_warnings": r.get("n_convergence_warnings", np.nan),
        }

    @staticmethod
    def _cap(r: pd.Series) -> dict:
        reason = r.get("cap_reason", "")
        return {
            "n_star_at_cap": r.get("n_star_at_cap", np.nan),
            "cap_reason":    "" if pd.isna(reason) else reason,
        }

    def _sig_row(self, ds: str, model: str, N: int) -> pd.Series:
        df = self.inp["sig"]
        sub = df[(df.dataset == DISPLAY_NAMES[ds]) & (df.model == model) & (df.N == N)]
        if len(sub) != 1:
            raise MissingInput(f"5_1_2: no row for {DISPLAY_NAMES[ds]} / {model} / N={N}")
        return sub.iloc[0]

    def add_signature(self, axis: str, ds: str, model: str, N: int,
                      expect_star: bool) -> int:
        """Signature (lin+time) row from 5_1_2 at depth N."""
        r = self._sig_row(ds, model, N)
        if expect_star and not bool(r.is_N_star):
            raise MissingInput(
                f"optimal_N_{ds}_{model}.txt says N*={N}, but 5_1_2's CSV does not mark "
                f"that row as N*: the files come from different runs"
            )
        ok = not pd.isna(r.accuracy)
        correct = (self._correctness("sig_pred", ds, f"5_1_2 {model} N={N}",
                                     model=model, N=N) if ok else None)
        return self.add(
            correct, ds, axis=axis, method=f"Signature {MODEL_LABELS[model]}",
            model=model, config="lin+time", N=N, n_features=r.n_features,
            cv_score=r.cv_score, test_accuracy=r.accuracy, test_f1_macro=r.f1_macro,
            feat_time_s=r.feat_time_s, fit_time_s=r.fit_time_s,
            **self._cap(r), **self._diag(r),
        )


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

def axe1(b: Builder, ds: str, n_star: Dict[str, int]) -> None:
    base = b.inp["base"]
    for model in MODELS:
        sub = base[(base.dataset == DISPLAY_NAMES[ds]) & (base.model == model)]
        if len(sub) != 1:
            raise MissingInput(f"5_1_1: no row for {DISPLAY_NAMES[ds]} / {model}")
        r = sub.iloc[0]
        i_base = b.add(
            b._correctness("base_pred", ds, f"5_1_1 {model}", model=model), ds,
            axis="Axe 1", method=f"Baseline {MODEL_LABELS[model]}", model=model,
            config="stats", n_features=r.n_features, cv_score=r.cv_score,
            test_accuracy=r.accuracy, test_f1_macro=r.f1_macro,
            feat_time_s=r.feat_time_s, fit_time_s=r.fit_time_s,
            n_star_at_cap=np.nan, **b._diag(r),
        )
        i_sig = b.add_signature("Axe 1", ds, model, n_star[model], expect_star=True)
        b.test("Axe 1", ds, i_sig, i_base)


def axe2(b: Builder, ds: str) -> None:
    aug = b.inp["aug"]
    sub = aug[aug.dataset == DISPLAY_NAMES[ds]]
    idx = {}
    selected = None
    for config in CONFIGS:
        star = sub[(sub.config == config) & sub.is_config_N_star.astype(bool)]
        if len(star) != 1:
            raise MissingInput(f"5_2_1: no N* row for {DISPLAY_NAMES[ds]} / {config}")
        r = star.iloc[0]
        N = int(r.N)
        ok = not pd.isna(r.test_accuracy)
        correct = (b._correctness("aug_pred", ds, f"5_2_1 {config} N={N}",
                                  config=config, N=N) if ok else None)
        is_sel = bool(r.is_selected)
        if is_sel:
            selected = config
        idx[config] = b.add(
            correct, ds, axis="Axe 2", method=f"LASSO {config}", model="LR-L1",
            config=config, N=N, selected=is_sel, n_features=r.n_features,
            n_nonzero_features=r.n_nonzero_features, cv_score=r.cv_score,
            test_accuracy=r.test_accuracy, test_f1_macro=r.test_f1_macro,
            feat_time_s=r.feat_time_s, fit_time_s=r.fit_time_s,
            **b._cap(r), **b._diag(r),
        )
    if selected is None:
        raise MissingInput(f"5_2_1: no selected configuration for {DISPLAY_NAMES[ds]}")
    if selected == "lin+time":
        b.test("Axe 2", ds, idx[selected], idx["lin+time"], status="not_applicable")
    else:
        b.test("Axe 2", ds, idx[selected], idx["lin+time"])


def _rformer_rows(b: Builder, ds: str, rf: pd.DataFrame, N: int,
                  cap: dict) -> Tuple[Dict[int, int], int]:
    """Per-seed RFormer rows + one seed-averaged row at depth N."""
    sub = rf[(rf.model == "RFormer") & (rf.N == N)]
    seed_rows: Dict[int, int] = {}
    correct_seeds = []
    for seed in SEEDS:
        s = sub[sub.seed == seed]
        if len(s) != 1:
            raise MissingInput(f"5_3_1: no RFormer row for {DISPLAY_NAMES[ds]} N={N} seed {seed}")
        r = s.iloc[0]
        ok = not pd.isna(r.test_accuracy)
        correct = (b._correctness("rf_pred", ds, f"RFormer N={N} seed {seed}",
                                  model="RFormer", N=N, seed=seed) if ok else None)
        if correct is not None:
            correct_seeds.append(correct)
        seed_rows[seed] = b.add(
            correct, ds, axis="Axe 3", method="RFormer", model="RFormer",
            config="lin+time multi-view", N=N, seed=seed, n_seeds=1,
            n_features=r.n_features, test_accuracy=r.test_accuracy,
            test_f1_macro=r.test_f1_macro, feat_time_s=r.feat_time_s,
            fit_time_s=r.fit_time_s, **cap,
        )
    n_ok = len(correct_seeds)
    agg = dict(
        axis="Axe 3", method="RFormer (mean over seeds)", model="RFormer",
        config="lin+time multi-view", N=N, n_seeds=n_ok,
        n_features=sub.n_features.iloc[0], feat_time_s=sub.feat_time_s.iloc[0], **cap,
    )
    if n_ok == len(SEEDS):
        acc = sub.test_accuracy.to_numpy(dtype=float)
        agg.update(
            test_accuracy=acc.mean(), test_accuracy_std=acc.std(ddof=1),
            test_f1_macro=sub.test_f1_macro.mean(), fit_time_s=sub.fit_time_s.mean(),
        )
        # Seed-averaged CI: same resampled indices for every seed, then averaged.
        b._y_ref(ds)
        ci = boot_ci(np.mean(correct_seeds, axis=0), b.idx[ds])
        i_agg = b.add(None, ds, ci=ci, **agg)
    else:
        i_agg = b.add(None, ds, status=f"out_of_memory ({len(SEEDS) - n_ok}/{len(SEEDS)} seeds)",
                      **agg)
    return seed_rows, i_agg


def axe3(b: Builder, ds: str, n_star: Dict[str, int]) -> List[Tuple[Dict[int, int], int]]:
    rf_all = b.inp["rf"]
    rf = rf_all[rf_all.dataset == DISPLAY_NAMES[ds]]
    if rf.empty:
        raise MissingInput(f"5_3_1: no rows for {DISPLAY_NAMES[ds]}")
    N_star = int(rf.N_star.iloc[0])
    if N_star != n_star["LR-L1"]:
        raise MissingInput(
            f"5_3_1 used N*={N_star} for {DISPLAY_NAMES[ds]}, but 5_1_2's LASSO N* is "
            f"{n_star['LR-L1']}: rerun axe3.sh"
        )
    if rf.compare_N.isna().all():
        raise MissingInput(f"5_3_1: no comparison depth for {DISPLAY_NAMES[ds]}")
    compare_N = int(rf.compare_N.iloc[0])

    def lasso_cap(N: int) -> dict:
        return Builder._cap(b._sig_row(ds, "LR-L1", N))

    i_lasso = {N_star: b.add_signature("Axe 3", ds, "LR-L1", N_star, expect_star=True)}
    b.add_signature("Axe 3", ds, "LR-L2", n_star["LR-L2"], expect_star=True)  # descriptive
    if compare_N != N_star:
        i_lasso[compare_N] = b.add_signature("Axe 3", ds, "LR-L1", compare_N, expect_star=False)

    ctrl = rf[(rf.model == "LR-L1-control") & (rf.N == compare_N)]
    if len(ctrl) != 1:
        raise MissingInput(f"5_3_1: no LASSO control row for {DISPLAY_NAMES[ds]} N={compare_N}")
    r = ctrl.iloc[0]
    ok = not pd.isna(r.test_accuracy)
    correct = (b._correctness("rf_pred", ds, f"control N={compare_N}",
                              model="LR-L1-control", N=compare_N) if ok else None)
    i_ctrl = b.add(
        correct, ds, axis="Axe 3", method="LASSO control", model="LR-L1",
        config="lin+time multi-view (flattened)", N=compare_N, n_features=r.n_features,
        cv_score=r.cv_score, test_accuracy=r.test_accuracy, test_f1_macro=r.test_f1_macro,
        feat_time_s=r.feat_time_s, fit_time_s=r.fit_time_s,
        **lasso_cap(compare_N), **Builder._diag(r),
    )

    groups = []
    for N in sorted({N_star, compare_N}, reverse=True):
        seed_rows, i_agg = _rformer_rows(b, ds, rf, N, lasso_cap(N))
        for seed, i_seed in seed_rows.items():
            b.test("Axe 3", ds, i_seed, i_lasso[N], seed=seed)
        groups.append((seed_rows, i_agg))
    b.test("Axe 3", ds, i_ctrl, i_lasso[compare_N])
    return groups


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the comparison table and the McNemar tests (Holm within each axis)."""
    b = Builder(load_inputs())
    n_star = {ds: {m: load_optimal_N(ds, m) for m in MODELS} for ds in DATASETS}
    rformer_groups = []
    for ds in DATASETS:
        axe1(b, ds, n_star[ds])
    for ds in DATASETS:
        axe2(b, ds)
    for ds in AXE3_DATASETS:
        rformer_groups += axe3(b, ds, n_star[ds])

    tests = pd.DataFrame(b.tests, columns=TEST_COLUMNS)
    for axis in AXIS_COLORS:
        fam = (tests.axis == axis) & (tests.status == "tested")
        if fam.any():
            tests.loc[fam, "p_holm"] = holm(tests.loc[fam, "p_raw"].to_numpy())
    tested = tests.status == "tested"
    tests["significant_raw"] = (tests.p_raw < ALPHA).where(tested)
    tests["significant_holm"] = (tests.p_holm < ALPHA).where(tested)

    # Copy each test's p-values onto the row of method_a.
    for t, i_a in zip(tests.itertuples(), b.test_rows):
        if t.status == "not_applicable":
            b.rows[i_a]["mcnemar_vs"] = "not applicable (lin+time selected)"
            continue
        b.rows[i_a]["mcnemar_vs"] = f"{t.method_b} (N={'' if pd.isna(t.N_b) else int(t.N_b)})"
        b.rows[i_a]["p_raw"], b.rows[i_a]["p_holm"] = t.p_raw, t.p_holm
    # RFormer seed-averaged rows: how many of the 5 seeds are significant.
    for seed_rows, i_agg in rformer_groups:
        p_raw = np.array([b.rows[i]["p_raw"] for i in seed_rows.values()], dtype=float)
        p_holm = np.array([b.rows[i]["p_holm"] for i in seed_rows.values()], dtype=float)
        if np.isnan(p_raw).all():   # no seed could be tested (out of memory): no count
            continue
        b.rows[i_agg]["seeds_p05_raw"] = int(np.sum(p_raw < ALPHA))
        b.rows[i_agg]["seeds_p05_holm"] = int(np.sum(p_holm < ALPHA))
        b.rows[i_agg]["mcnemar_vs"] = b.rows[next(iter(seed_rows.values()))]["mcnemar_vs"]

    df = pd.DataFrame(b.rows, columns=COMPARISON_COLUMNS)
    for col in ("N", "seed", "n_seeds", "n_features", "n_nonzero_features",
                "n_star_at_cap", "c_at_grid_edge", "n_convergence_warnings",
                "seeds_p05_raw", "seeds_p05_holm"):
        df[col] = pd.to_numeric(df[col]).round().astype("Int64")
    df["selected"] = df.selected.map({True: True, False: False})
    for col in ("N_a", "N_b", "seed", "b", "c"):
        tests[col] = pd.to_numeric(tests[col]).astype("Int64")
    return df, tests


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _label_fr(r: pd.Series) -> str:
    """French figure label of one comparison row."""
    N = "" if pd.isna(r.N) else f"N={int(r.N)}"
    model = MODEL_LABELS.get(r.model, r.model)
    if r.method.startswith("Baseline"):
        text = f"Référence statistique, {model}"
    elif r.method.startswith("Signature"):
        text = f"Signature lin+time, {model} ({N})"
    elif r.axis == "Axe 2":
        text = f"LASSO {r.config} ({N})" + ("  ★ choisie par VC" if r.selected is True else "")
    elif r.method == "LASSO control":
        text = f"LASSO de contrôle multi-vues ({N})"
    else:
        text = f"RFormer ({N}, moyenne sur {len(SEEDS)} graines)"
    return f"{r.axis} · {text}"


def plot_results(df: pd.DataFrame) -> plt.Figure:
    """Per dataset: test accuracy with 95% bootstrap CI, and cost (log), one line per method.

    Per-seed RFormer rows are summarised by their seed-averaged row. A row
    repeated in several axes (same model, configuration, N and accuracy) is
    shown once. Rows skipped for memory reasons are shown as missing.
    """
    fig_rows = df[df.seed.isna()].copy()
    fig_rows["acc_key"] = fig_rows.test_accuracy.round(12)
    fig_rows = fig_rows.drop_duplicates(subset=["dataset", "model", "config", "N", "acc_key"])

    n = len(DATASETS)
    heights = [max(len(fig_rows[fig_rows.dataset == DISPLAY_NAMES[ds]]), 1) for ds in DATASETS]
    fig, axes = plt.subplots(
        n, 2, figsize=(15, 0.42 * sum(heights) + 2.5),
        gridspec_kw={"height_ratios": heights, "width_ratios": [3, 2]}, squeeze=False,
    )
    fig.suptitle(
        "Synthèse des axes 1 à 3 — taux de bonne classification (test, IC bootstrap à 95 %) "
        "et coût par méthode",
        fontsize=13, y=1.0,
    )
    for i, ds in enumerate(DATASETS):
        sub = fig_rows[fig_rows.dataset == DISPLAY_NAMES[ds]].reset_index(drop=True)
        y = np.arange(len(sub))[::-1]
        ax_a, ax_c = axes[i, 0], axes[i, 1]
        trans_a = mtransforms.blended_transform_factory(ax_a.transAxes, ax_a.transData)
        trans_c = mtransforms.blended_transform_factory(ax_c.transAxes, ax_c.transData)
        for yy, (_, r) in zip(y, sub.iterrows()):
            color = AXIS_COLORS[r.axis]
            if pd.isna(r.test_accuracy):
                for ax, tr in ((ax_a, trans_a), (ax_c, trans_c)):
                    ax.text(0.02, yy, "manquant (mémoire insuffisante)", transform=tr,
                            va="center", fontsize=8, color="grey", style="italic")
                continue
            marker = "D" if r.selected is True else "o"
            ax_a.errorbar(
                r.test_accuracy, yy,
                xerr=[[r.test_accuracy - r.ci_low], [r.ci_high - r.test_accuracy]],
                fmt=marker, color=color, ecolor=color, capsize=3, markersize=6,
            )
            ax_a.text(r.ci_high, yy, f"  {r.test_accuracy:.3f}", va="center", fontsize=8)
            if not pd.isna(r.total_time_s):
                ax_c.barh(yy, r.total_time_s, color=color, alpha=0.8, height=0.6)
        ax_a.set_yticks(y)
        ax_a.set_yticklabels([_label_fr(r) for _, r in sub.iterrows()], fontsize=8)
        ax_c.set_yticks(y)
        ax_c.set_yticklabels([])
        ax_a.set_title(DISPLAY_NAMES[ds], fontsize=11, loc="left")
        ax_c.set_xscale("log")
        for ax in (ax_a, ax_c):
            ax.set_ylim(-0.7, len(sub) - 0.3)
            ax.grid(True, axis="x", which="both", linestyle=":", alpha=0.5)
            ax.set_axisbelow(True)
        if i == n - 1:
            ax_a.set_xlabel("Taux de bonne classification (test)")
            ax_c.set_xlabel(
                "Temps de calcul des caractéristiques + d'ajustement (s, échelle log)"
            )
    handles = [Line2D([0], [0], marker="o", color=c, linestyle="", label=a)
               for a, c in AXIS_COLORS.items()]
    handles.append(Line2D([0], [0], marker="D", color="black", linestyle="",
                          label="configuration choisie par VC (axe 2)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.text(
        0.5, -0.045,
        f"Barres d'erreur : IC bootstrap à {int(CI_LEVEL * 100)} % ({N_BOOT} rééchantillonnages "
        "du jeu de test, méthode des percentiles). RFormer : moyenne sur les graines et IC de "
        "la moyenne. Coût mesuré sur le même CPU (i5-6300U).",
        ha="center", fontsize=8,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save + main
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame, tests: pd.DataFrame, fig: plt.Figure, name: str) -> None:
    """Save the comparison table, the tests table and the figure."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{name}.csv"
    tests_path = RESULTS_DIR / "5_4_1_tests.csv"
    png_path = RESULTS_DIR / f"{name}.png"
    df.to_csv(csv_path, index=False)
    tests.to_csv(tests_path, index=False)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    logger.info("Results saved to %s, %s and %s", csv_path, tests_path, png_path)


def main() -> None:
    """Aggregate Axes 1-3, compute CIs and McNemar tests, save CSVs and figure."""
    log_versions(logger)
    logger.info("Aggregating chap5 results across Axes 1-3...")
    try:
        df, tests = run_experiment()
    except MissingInput as exc:
        logger.error("Missing input: %s", exc)
        sys.exit(1)
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        logger.info(
            "Comparisons:\n%s",
            df[["axis", "dataset", "method", "N", "seed", "status", "test_accuracy",
                "ci_low", "ci_high", "total_time_s"]].to_string(index=False),
        )
        logger.info(
            "Tests:\n%s",
            tests[["axis", "dataset", "method_a", "N_a", "method_b", "N_b", "seed",
                   "status", "b", "c", "p_raw", "p_holm"]].to_string(index=False),
        )
    n_oom = int(df.status.str.startswith("out_of_memory").sum())
    if n_oom:
        logger.warning("%d row(s) skipped upstream for memory reasons (status out_of_memory)",
                       n_oom)
    fig = plot_results(df)
    save_results(df, tests, fig, NAME)
    plt.close(fig)
    logger.info("Done.")


if __name__ == "__main__":
    main()
