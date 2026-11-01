#!/usr/bin/env bash
# Thesis section 5.5: final comparison (5_4_1), pure aggregation of the CSVs
# and per-sample predictions written by axe1.sh, axe2.sh and axe3.sh:
# bootstrap CIs, McNemar tests with Holm correction per axis.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require axe1.sh "$RESULTS_DIR"/5_1_1_baseline_stats{,_predictions}.csv \
    "$RESULTS_DIR"/5_1_2_signature_lr{,_predictions}.csv \
    "$RESULTS_DIR"/optimal_N_{ecg200,racketsports,chartraj,natops}_{LR-L2,LR-L1}.txt
require axe2.sh "$RESULTS_DIR"/5_2_1_augmentations{,_predictions}.csv
require axe3.sh "$RESULTS_DIR"/5_3_1_rformer{,_predictions}.csv
start_log compare

run 5_4_1_comparisons.py

finish_log
echo "compare.sh done. Outputs in $RESULTS_DIR/"
