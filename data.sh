#!/usr/bin/env bash
# Thesis section 5.1: preprocessing (5_0_1) and per-dataset exploration (5_0_2 … 5_0_5).
# 5_0_1 downloads the datasets via sktime and writes data/processed/;
# 5_0_2 … 5_0_5 only read data/processed/.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log data

run 5_0_1_preprocess.py
run 5_0_2_load_ecg200.py
run 5_0_3_load_chartraj.py
run 5_0_4_load_racketsports.py
run 5_0_5_load_natops.py

finish_log
echo "data.sh done. Outputs in $DATA_DIR/ and $RESULTS_DIR/"
