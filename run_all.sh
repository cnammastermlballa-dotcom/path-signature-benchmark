#!/usr/bin/env bash
# Run the full Chapter 5 pipeline: the five stage scripts, in order.
# Outputs go to results/chap5/ (git-ignored). Stops at the first error.
#
# Usage: ./run_all.sh [--skip-rformer]
#   RFormer is trained by default (axe3.sh, on the CPU unless torch sees a GPU).
#   --skip-rformer  skips axe3.sh (RFormer and its LASSO control). compare.sh
#                   then runs only if results/chap5/5_3_1_rformer.csv is left
#                   over from an earlier axe3.sh run.
set -euo pipefail
cd "$(dirname "$0")"

skip_rformer=false
case "${1:-}" in
    "") ;;
    --skip-rformer) skip_rformer=true ;;
    *) echo "Usage: $0 [--skip-rformer]" >&2; exit 2 ;;
esac

./data.sh
./axe1.sh
./axe2.sh
if $skip_rformer; then
    echo "=== axe3.sh skipped (--skip-rformer) ==="
else
    ./axe3.sh
fi
if [[ -e results/chap5/5_3_1_rformer.csv ]]; then
    if $skip_rformer; then
        echo "Note: compare.sh uses 5_3_1_rformer.csv from an earlier axe3.sh run."
    fi
    ./compare.sh
else
    echo "=== compare.sh skipped: no results/chap5/5_3_1_rformer.csv (run ./axe3.sh) ==="
fi

echo "Done. Outputs in results/chap5/"
