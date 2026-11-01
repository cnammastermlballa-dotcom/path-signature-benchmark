#!/usr/bin/env bash
# Thesis section 5.2 (Axe 1): statistical baseline (5_1_1) and truncated
# signature + logistic regression (5_1_2), which writes the per-model N*.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mapfile -t inputs < <(data_files "${DATASETS[@]}")
require data.sh "${inputs[@]}"
start_log axe1

run 5_1_1_baseline_stats.py
run 5_1_2_signature_lr.py

finish_log
echo "axe1.sh done. Outputs in $RESULTS_DIR/"
