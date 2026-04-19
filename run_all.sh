#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv/bin/python. Create the virtual environment and install requirements first." >&2
  exit 1
fi

.venv/bin/python scripts/task1_baseline.py "$@"
.venv/bin/python scripts/task2_collect_scores.py "$@"
.venv/bin/python scripts/task3_local_cp.py
