#!/usr/bin/env bash
# Initialize the requested Conda installation and activate the project runtime.
# This file is sourced by run_api.sh so activation affects the API process.

CONDA_BIN="${CONDA_BIN:-/share/homes/teinhonglo/anaconda3/bin/conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-teval_py310}"

if [[ ! -x "$CONDA_BIN" ]]; then
  echo "Error: Conda executable not found or not executable: $CONDA_BIN" >&2
  echo "Set CONDA_BIN=/path/to/conda if Conda is installed elsewhere." >&2
  return 1 2>/dev/null || exit 1
fi

# Equivalent to:
# eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
if ! CONDA_HOOK="$("$CONDA_BIN" shell.bash hook)"; then
  echo "Error: failed to initialize Conda using $CONDA_BIN." >&2
  return 1 2>/dev/null || exit 1
fi
eval "$CONDA_HOOK"
if ! conda activate "$CONDA_ENV_NAME"; then
  echo "Error: failed to activate Conda environment: $CONDA_ENV_NAME" >&2
  return 1 2>/dev/null || exit 1
fi

echo "Conda environment activated: $CONDA_ENV_NAME"
