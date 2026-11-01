#!/usr/bin/env bash
# Thesis section 5.4 (Axe 3): trains the Rough Transformer (5 seeds) and the
# LASSO control on multi-view signatures (5_3_1), CharTraj and NATOPS only.
# Uses the GPU if torch sees one, otherwise the CPU.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mapfile -t inputs < <(data_files chartraj natops)
require data.sh "${inputs[@]}"
require axe1.sh "$RESULTS_DIR"/optimal_N_{chartraj,natops}_LR-L1.txt
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: torch cannot be imported; it is required by 5_3_1_rformer.py." >&2
    exit 1
fi
start_log axe3

run 5_3_1_rformer.py

finish_log
echo "axe3.sh done. Outputs in $RESULTS_DIR/"
