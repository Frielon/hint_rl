#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_INFERENCE_PYTHON="/shared_home/xutao.ma/miniconda3/envs/inference/bin/python"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  SELECTED_PYTHON="${PYTHON_BIN}"
elif [[ -x "${DEFAULT_INFERENCE_PYTHON}" ]]; then
  SELECTED_PYTHON="${DEFAULT_INFERENCE_PYTHON}"
else
  SELECTED_PYTHON="python3"
fi

exec "${SELECTED_PYTHON}" "${SCRIPT_DIR}/generate_reference_solutions.py" "$@"
