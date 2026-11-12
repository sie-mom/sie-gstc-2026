#!/usr/bin/env bash
# End-to-end runner script for Example 1 publication figures.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Ensure repository root is on PYTHONPATH
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Prefer virtual environment python if available
if [[ -f "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Error: Python 3 not found." >&2
    exit 1
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_example_1.py" "$@"

