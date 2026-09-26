# Path Signature Benchmark

Experiments for Chapter 5 of a Master 2 thesis (CNAM) on **path signatures
for time-series classification**. Four datasets, three axes:

| Axe | Script | Question |
|-----|--------|----------|
| 1 | `5_2_1`, `5_2_2` | Signature (lin+time) + logistic regression vs. statistical baseline; optimal truncation depth N* |
| 2 | `5_3_1` | Effect of path augmentations (time, lead-lag, rectilinear, cumsum) |
| 3 | `5_4_1` | Rough Transformer: windowed signatures as tokens for a Transformer encoder |
| — | `5_5_1` | Final comparison table (accuracy + computational cost) |

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

`torch` is only needed for Axe 3 (`5_4_1_rformer.py`).

## Running

All scripts must be run **from the repository root** (they use relative paths).

Whole pipeline on CPU. RFormer is not retrained: its reference results are
copied from `results/reference/chap5/` so the final comparison can be built.

```bash
./run_all.sh
```

Progress (per-N accuracy, CV scores, fit times) is only printed to the
terminal. To also keep it in a file:

```bash
./run_all.sh 2>&1 | tee run_all.log
```

Some steps are slow on CPU (e.g. CharTraj LR-L1 at N=2 took ~10 min in the
reference run), so a pause in the output does not mean the run is stuck.

Including RFormer training (GPU strongly recommended):

```bash
./run_all.sh --rformer
```

Or a single script, e.g.:

```bash
python experiments/chap5/5_2_2_signature_lr.py
```

Script order matters: `5_0` produces `data/processed/` (already committed, so it
can be skipped), `5_2_2` writes `optimal_N_*.txt` used by `5_3_1` and `5_4_1`,
`5_3_1` writes `best_config_*.txt`, and `5_5_1` aggregates all CSVs.
To run a later script alone without rerunning the earlier ones, seed its
inputs from the reference results first:

```bash
mkdir -p results/chap5
cp results/reference/chap5/*.txt results/chap5/          # for 5_3_1 / 5_4_1
cp results/reference/chap5/*.csv results/chap5/          # for 5_5_1
```

### Axe 3 on Google Colab

Open `notebooks/demo_google_colab.ipynb` in Colab (Runtime → T4 GPU), or run:

```bash
!git clone https://github.com/cnammastermlballa-dotcom/path-signature-benchmark.git
%cd path-signature-benchmark
!pip install -q iisignature torch scikit-learn matplotlib pandas
!mkdir -p results/chap5 && cp results/reference/chap5/optimal_N_*.txt results/chap5/
!python experiments/chap5/5_4_1_rformer.py
```

## Outputs

Scripts write to `results/chap5/` (git-ignored, created on the first run): one
CSV + PNG per script, plus `optimal_N_{dataset}.txt` and
`best_config_{dataset}.txt`.

The results reported in the thesis are committed in `results/reference/chap5/`,
with the same file names, so you can compare your run against them, e.g.:

```bash
diff results/chap5/5_5_1_comparison.csv results/reference/chap5/5_5_1_comparison.csv
```

Timing columns will differ with hardware.

Expected test accuracy (`5_5_1_comparison.csv`):

| Dataset | Baseline stats | Sig + LR | Sig + augmentation | RFormer |
|---------|----------------|----------|--------------------|---------|
| ECG200 | 0.71 | 0.83 | 0.82 | – |
| RacketSports | 0.77 | 0.84 | 0.84 | – |
| CharacterTrajectories | 0.94 | 0.98 | 0.99 | 0.99 |
| NATOPS | 0.84 | 0.91 | 0.91 | 0.93 |

Linear models use fixed seeds (`random_state=42`) and should reproduce exactly;
RFormer on GPU may vary slightly (±0.01) due to non-deterministic CUDA kernels.
Timings depend on hardware.

## Repository layout

```
experiments/chap5/   experiment scripts (5_0 … 5_5_1)
data/processed/      numpy arrays produced by 5_0_preprocess.py
results/reference/   reference results of the thesis (committed)
results/chap5/       your run's outputs (git-ignored, created by the scripts)
notebooks/           Colab notebook for the Axe 3 GPU run
```