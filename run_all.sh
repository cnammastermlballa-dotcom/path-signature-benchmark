#!/usr/bin/env bash
# Run the full Chapter 5 pipeline from the repository root.
# Outputs go to results/chap5/ (git-ignored); reference results of the
# thesis are in results/reference/chap5/.
# Usage: ./run_all.sh            (CPU steps; copies reference RFormer results)
#        ./run_all.sh --rformer  (also trains RFormer, GPU recommended)
set -euo pipefail
cd "$(dirname "$0")"

RUN_RFORMER=false
[[ "${1:-}" == "--rformer" ]] && RUN_RFORMER=true

run() {
    echo "=== $1 ==="
    python "experiments/chap5/$1"
}

run 5_0_preprocess.py
run 5_1_1_load_ecg200.py
run 5_1_2_load_chartraj.py
run 5_1_3_load_racketsports.py
run 5_1_4_load_natops.py
run 5_2_1_baseline_stats.py
run 5_2_2_signature_lr.py
run 5_3_1_augmentations.py
if $RUN_RFORMER; then
    run 5_4_1_rformer.py
else
    echo "=== 5_4_1_rformer.py skipped: copying reference RFormer results ==="
    cp results/reference/chap5/5_4_1_rformer.csv results/reference/chap5/5_4_1_rformer.png results/chap5/
fi
run 5_5_1_comparison.py

echo "Done. Outputs in results/chap5/"