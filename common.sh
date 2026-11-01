# Shared setup for the Chapter 5 stage scripts (data.sh, axe1.sh, axe2.sh,
# axe3.sh, compare.sh). Sourced, not executed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RESULTS_DIR=results/chap5
DATA_DIR=data/processed
DATASETS=(ecg200 racketsports chartraj natops)
mkdir -p "$RESULTS_DIR"

if ! python -c "import iisignature" 2>/dev/null; then
    echo "ERROR: iisignature cannot be imported by '$(command -v python || echo python)'." >&2
    echo "Activate the environment first: conda activate path-sig-benchmark" >&2
    echo "(or the virtualenv where requirements.txt is installed)." >&2
    exit 1
fi

run() {
    echo "=== $1 ==="
    python "experiments/chap5/$1"
}

# require <script to run first> <file>...
# Exits with a clear message if any of the files is missing.
require() {
    local stage=$1; shift
    local missing=() f
    for f in "$@"; do
        [[ -e $f ]] || missing+=("$f")
    done
    if (( ${#missing[@]} )); then
        echo "ERROR: missing input(s) from an earlier stage:" >&2
        printf '  %s\n' "${missing[@]}" >&2
        echo "Run ./$stage first." >&2
        exit 1
    fi
}

# data_files <dataset>...: the 4 .npy files of each dataset in data/processed/.
data_files() {
    local ds
    for ds in "$@"; do
        printf '%s\n' "$DATA_DIR/${ds}_"{X,y}_{train,test}.npy
    done
}

# start_log <stage>: writes results/chap5/<stage>_run.log with the git commit
# and start time; finish_log appends the end time on success.
start_log() {
    LOG_FILE="$RESULTS_DIR/$1_run.log"
    local commit dirty=""
    commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
    [[ -n $(git status --porcelain 2>/dev/null) ]] && dirty=" (with uncommitted changes)"
    {
        echo "stage:    $1"
        echo "commit:   $commit$dirty"
        echo "started:  $(date -Iseconds)"
    } > "$LOG_FILE"
}

finish_log() {
    echo "finished: $(date -Iseconds)" >> "$LOG_FILE"
}
