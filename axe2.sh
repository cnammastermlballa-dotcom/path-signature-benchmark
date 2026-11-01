#!/usr/bin/env bash
# Thesis section 5.3 (Axe 2): path augmentations under LASSO (5_2_1).
# 5_2_1 selects its own N per configuration by CV: it does not read
# 5_1_2's optimal_N files, so it only needs data/processed/.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mapfile -t inputs < <(data_files "${DATASETS[@]}")
require data.sh "${inputs[@]}"
start_log axe2

run 5_2_1_augmentations.py

finish_log
echo "axe2.sh done. Outputs in $RESULTS_DIR/"
