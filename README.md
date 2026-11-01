# Path Signature Benchmark

Experiments for Chapter 5 of a Master 2 thesis (CNAM) on **path signatures
for time-series classification**. Four datasets, three axes:

| Axe | Script | Question |
|-----|--------|----------|
| 1 | `5_1_1`, `5_1_2` | Signature (lin+time) + logistic regression vs. statistical baseline; optimal truncation depth N* |
| 2 | `5_2_1` | Effect of path augmentations (time, lead-lag, rectilinear, cumsum) |
| 3 | `5_3_1` | Rough Transformer: windowed signatures as tokens for a Transformer encoder |
| — | `5_4_1` | Final comparison table (accuracy + computational cost) |

Datasets (UCR/UEA archive, downloaded automatically via `sktime`):

| Dataset | Train / Test | Channels d | Classes | Length T |
|---------|--------------|-----------|---------|----------|
| ECG200 | 100 / 100 | 1 | 2 | 96 |
| RacketSports | 151 / 152 | 6 | 4 | 30 |
| CharacterTrajectories | 1422 / 1436 | 3 | 20 | 60–180 |
| NATOPS | 180 / 180 | 24 | 6 | 51 |

## Installation

Python 3.11. With conda:

```bash
git clone https://github.com/cnammastermlballa-dotcom/path-signature-benchmark.git
cd path-signature-benchmark
conda env create -f environment.yml
conda activate path-sig-benchmark
```

Or with pip in a virtual environment:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install numpy==2.3.5          # iisignature needs numpy at build time
pip install -r requirements.txt
```

`torch` is only needed for Axe 3 (`axe3.sh`).

### Versions

The thesis results were produced with:

| Package | Version |
|---------|---------|
| Python | 3.11.17 |
| numpy | 2.3.5 |
| scikit-learn | 1.7.2 |
| iisignature | 0.24 |
| torch | 2.15.0+cpu (CPU build; Axe 3 only) |

Every script logs these versions on its first line (`INFO — Versions: …`), so
each `<stage>_run.log` records the environment of the run. Keep
`scikit-learn==1.7.2`: LASSO uses liblinear, which fits one-vs-rest on
multiclass data, and scikit-learn 1.8 and later raise an error on it.

## Running

The pipeline is split into one script per stage, at the repository root. Each
can be launched from any directory (it moves to the repository root itself)
and shares its setup through `common.sh`:

- it stops if `iisignature` cannot be imported, i.e. if the
  `path-sig-benchmark` environment is not active;
- it checks that the files it needs from earlier stages exist, and otherwise
  names the script to run first;
- it writes `results/chap5/<stage>_run.log`: the git commit (flagged if the
  working tree has uncommitted changes), the start time, the full output of
  each script (versions line, per-N accuracy, CV scores, fit times, warnings),
  and, on success, the end time.

Run the stages in this order:

| Script | Thesis section | Runtime |
|--------|----------------|---------|
| `./data.sh` | 5.1 | ~ 1–2 min |
| `./axe1.sh` | 5.2 | ~ 36 min |
| `./axe2.sh` | 5.3 | ~ 90 min |
| `./axe3.sh` | 5.4 | ~ 15 min |
| `./compare.sh` | 5.5 | < 1 min |

`axe2.sh` and `axe3.sh` are independent of each other: once `axe1.sh` has run,
they can be launched in either order.

Runtimes were measured on a laptop CPU (Intel i5-7300U, 2 cores / 4 threads); a
faster machine will be quicker. Some single fits are long (e.g. the CharTraj
LASSO fits, ~10 min), so a pause in the output does not mean the run is stuck.

To run everything in one go (~2.4 h on the laptop above), including the
RFormer training:

```bash
./run_all.sh
./run_all.sh --skip-rformer   # skips axe3.sh (RFormer + LASSO control)
```

With `--skip-rformer`, `compare.sh` runs only if a `5_3_1_rformer.csv` from an
earlier `axe3.sh` run is present in `results/chap5/`; otherwise it is skipped.

`run_all.sh` calls the five scripts in order and stops at the first error; the
`=== 5_x_y_... ===` header above the Python traceback shows which script
failed. Each stage's output is both printed and appended to its
`results/chap5/<stage>_run.log`, so no `tee` is needed.

A single Python script can still be run on its own from the repository root,
e.g. `python experiments/chap5/5_1_2_signature_lr.py`, but without the
prerequisite checks or the run log.

### `data.sh` — preprocessing and exploration (section 5.1)

```bash
./data.sh
```

- **Runs:** `5_0_1_preprocess.py`, then the loaders `5_0_2_load_ecg200.py`,
  `5_0_3_load_chartraj.py`, `5_0_4_load_racketsports.py`,
  `5_0_5_load_natops.py`.
- **Prerequisites:** none, but network access: `5_0_1` downloads the four
  datasets via `sktime`. The loaders only read `data/processed/`.
- **Outputs:**
  - `data/processed/{ecg200,racketsports,chartraj,natops}_{X,y}_{train,test}.npy`
    (labels remapped to 0..K-1). These files are committed, so a fresh clone
    can skip `data.sh` and start with `axe1.sh`.
  - `results/chap5/5_0_{2,3,4,5}_load_*.{csv,png}`: per-dataset statistics
    and plots.
  - `results/chap5/data_run.log`
- **Runtime:** ~ 1–2 min (depends on the downloads).

### `axe1.sh` — signature vs. statistical baseline (section 5.2)

```bash
./axe1.sh
```

- **Runs:** `5_1_1_baseline_stats.py` (statistical features + Ridge / LASSO),
  then `5_1_2_signature_lr.py` (lin+time signature, N = 1..10, + Ridge /
  LASSO; selects one N* per model by cross-validation).
- **Prerequisites:** `data/processed/*.npy` (from `data.sh`).
- **Outputs** in `results/chap5/`:
  - `5_1_1_baseline_stats.{csv,png}`, `5_1_1_baseline_stats_predictions.csv`
  - `5_1_2_signature_lr.{csv,png}`, `5_1_2_signature_lr_predictions.csv`
  - `optimal_N_{dataset}_{LR-L2,LR-L1}.txt`: N* for Ridge and for LASSO
  - `axe1_run.log`
- **Runtime:** `5_1_1` < 1 min; `5_1_2` ~ 36 min (35.6 min measured on
  2026-11-01 on an Intel i5-7300U, 2 cores / 4 threads; CharTraj takes
  ~ 94 % of it).

### `axe2.sh` — path augmentations (section 5.3)

```bash
./axe2.sh
```

- **Runs:** `5_2_1_augmentations.py`: lin+time, lead-lag, rectilinear and
  cumsum under LASSO. N is selected by cross-validation separately for each
  configuration (N = 1..10); `5_2_1` does **not** read the N* of `axe1.sh`.
- **Prerequisites:** `data/processed/*.npy` (from `data.sh`) only.
- **Outputs** in `results/chap5/`:
  - `5_2_1_augmentations.csv` (one row per dataset × configuration × N) and
    `5_2_1_augmentations_predictions.csv`
  - `5_2_1_augmentations*.png`, including test accuracy vs. log(number of
    features)
  - `best_config_{dataset}.txt`: configuration selected by CV score
  - `axe2_run.log`
- **Runtime:** ~ 90 min (87.6 min measured; CharTraj ≈ 95 % of it).

### `axe3.sh` — Rough Transformer (section 5.4)

```bash
./axe3.sh
```

- **Runs:** `5_3_1_rformer.py`. It **trains a Transformer** (Rough Transformer
  on multi-view signatures, 5 seeds) on CharacterTrajectories and NATOPS, plus
  a LASSO control on the same flattened tokens. The thesis results are trained
  on the CPU (i5-7300U, the same machine as the linear models), so the costs
  of all methods are comparable; a GPU is not needed. The script would use a
  GPU if `torch` detected one, but its timings would then no longer be
  comparable. If the control's dimension at N* is too large (NATOPS), or the
  control or RFormer runs out of memory at N*, both are run again at the
  fallback depth: the largest N < N* whose control fits within
  `MAX_FEATURE_DIM` (NATOPS: N=2). That depth (`compare_N` in the CSV) is
  where they are compared with LASSO.
- **Prerequisites:** `torch` installed; `data/processed/{chartraj,natops}_*.npy`
  (from `data.sh`); `results/chap5/optimal_N_{chartraj,natops}_LR-L1.txt`
  (from `axe1.sh`).
- **Outputs** in `results/chap5/`:
  - `5_3_1_rformer.{csv,png}` (one row per dataset × model × N × seed) and
    `5_3_1_rformer_predictions.csv`
  - `axe3_run.log`
- **Runtime:** ~ 15 min on the i5-7300U (~ 10 min of it is the CharTraj LASSO
  control, ~ 35 s per RFormer seed).

### `compare.sh` — final comparison (section 5.5)

```bash
./compare.sh
```

- **Runs:** `5_4_1_comparisons.py`, a pure aggregation of the earlier CSVs and
  per-sample predictions (no model is fitted, nothing is selected on test).
- **Prerequisites:** from `axe1.sh`: `5_1_1_baseline_stats.csv`,
  `5_1_2_signature_lr.csv`, their `_predictions.csv` and
  `optimal_N_{dataset}_{LR-L2,LR-L1}.txt`; from `axe2.sh`:
  `5_2_1_augmentations.csv` and its predictions; from `axe3.sh`:
  `5_3_1_rformer.csv` and its predictions.
- **Outputs** in `results/chap5/`:
  - `5_4_1_comparisons.csv`: one row per axis × dataset × method (Axe 1:
    baselines and signatures, Ridge and LASSO at their own N*; Axe 2: the four
    configurations at their own N*; Axe 3: LASSO, Ridge, the LASSO control and
    RFormer per seed and averaged over seeds, at N* and, after a fallback, at
    `compare_N`), with test accuracy, 95% bootstrap CI (2000 resamples,
    percentile method), cost, cap flags and fit diagnostics. Rows skipped for
    memory reasons are kept with `status = out_of_memory`.
  - `5_4_1_tests.csv`: one row per planned exact McNemar test, raw and
    Holm-adjusted p-values (one Holm family per axis).
  - `5_4_1_comparisons.png`, `compare_run.log`.
- **Runtime:** < 1 min. Exits with code 1 if an input is missing or
  inconsistent (e.g. N* files and CSVs from different runs).

### Optional: Axe 3 on Google Colab

The thesis results come from the CPU run above. `notebooks/demo_google_colab.ipynb`
can also run Axe 3 on a Colab GPU (Runtime → T4 GPU), but its timings are not
comparable with the CPU timings of the other methods. The repository is cloned from GitHub, so the code you want to run must
be pushed first. `axe3.sh` needs `optimal_N_{chartraj,natops}_LR-L1.txt`: run
`./axe1.sh` locally, upload these two files to `results/chap5/` in the Colab
session, then:

```bash
!pip install -q iisignature
!./axe3.sh
```

The notebook itself still copies these files from the former reference results
folder, which no longer exists, and needs to be updated.

## Outputs

All scripts write to `results/chap5/` (git-ignored, created on the first run):
one CSV + PNG per experiment script (plus `5_4_1_tests.csv`), per-sample test predictions
(`*_predictions.csv`), `optimal_N_*.txt`, `best_config_*.txt` and one
`<stage>_run.log` per stage script.

Linear models use fixed seeds (`random_state=42`) and should reproduce exactly;
RFormer is seeded (seeds 0–4), but results may vary slightly (±0.01) across
machines or `torch` versions. Timings depend on hardware.

The model CSVs (`5_1_1`, `5_1_2`, `5_2_1`, `5_3_1`) share the same cost columns:
`n_features`, `feat_time_s` (feature computation, train + test, shared by the
models fitted on the same features) and `fit_time_s` (GridSearchCV + refit +
test predict, or RFormer training + test predict). They also carry diagnostics:
`n_convergence_warnings` and `c_at_grid_edge` for every grid search, and, in
`5_1_2` / `5_2_1`, `n_star_at_cap` / `cap_reason` when the CV-selected N is
the largest N evaluated (starred in the figures).

Out of memory: in `5_1_2`, `5_2_1` and `5_3_1`, a `MemoryError` (or, in
`5_3_1`, a torch `RuntimeError` reporting an allocation failure) is logged as
a WARNING and that row is written with NaN metrics (in `5_1_2` / `5_2_1`,
`cap_reason` is then `fit_failed` on the N* just below it); the script goes
on. Any other error stops the script. Each script ends by logging how many
rows were skipped for memory reasons, and exits with code 1 (after saving its
outputs, so `run_all.sh` stops) if a required row is missing or NaN: an N*
row in `5_1_2`, a configuration's N* or the selected configuration in
`5_2_1`, the control or one of the 5 RFormer seeds at `compare_N` in `5_3_1`.

## Repository layout

```
data.sh, axe1.sh, axe2.sh, axe3.sh, compare.sh   one script per stage
common.sh            shared setup, sourced by the stage scripts
run_all.sh           runs the five stage scripts in order
experiments/chap5/   experiment scripts (5_0_1 … 5_4_1)
data/processed/      numpy arrays produced by 5_0_1_preprocess.py (committed)
results/chap5/       your run's outputs (git-ignored, created by the scripts)
notebooks/           optional Colab notebook for Axe 3
```
