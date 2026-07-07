#!/usr/bin/env bash
# Convenience launcher. Each solution has its own venv; this picks the right one
# and runs from the right directory, so it works from anywhere.
#
#   ./run.sh 1        raw agent (solution 1)
#   ./run.sh 2        RAG document agent (solution 2)
#   ./run.sh ingest   (re)index solution 2's documents/
#   ./run.sh inspect  show what's in solution 2's Chroma base
set -e
here="$(cd "$(dirname "$0")" && pwd)"

run() {   # $1 = solution dir, $2 = script to run
    local dir="$here/$1"
    if [ ! -x "$dir/.venv/bin/python" ]; then
        echo "No venv in $1/. Set it up first:"
        echo "  cd $1 && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
        exit 1
    fi
    cd "$dir" && exec .venv/bin/python "$2"
}

case "$1" in
    1)      run solution-1-raw-agent agent.py ;;
    2)      run solution-2-rag agent.py ;;
    ingest)  run solution-2-rag ingest.py ;;
    inspect) run solution-2-rag inspect_db.py ;;
    *)       echo "Usage: ./run.sh [1|2|ingest|inspect]"; exit 1 ;;
esac
