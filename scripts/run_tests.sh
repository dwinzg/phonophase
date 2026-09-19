#!/bin/bash
# Run the test suite with no dependencies beyond the runtime ones.
# Override the interpreter with PYTHON=python3 scripts/run_tests.sh
# Use pytest instead once requirements-dev.txt is installed.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3

status=0
for f in "$ROOT"/tests/test_*.py; do
    echo "=== $(basename "$f") ==="
    "$PY" "$f" || status=1
done
exit $status
