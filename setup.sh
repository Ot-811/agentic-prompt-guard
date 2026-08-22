#!/usr/bin/env bash
# Portable bootstrap for the Agentic Prompt Guard.
# Creates a local virtual environment and installs dependencies.
# Usage:  bash setup.sh   (then follow the printed instructions)
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV_DIR=".venv"

echo "==> Using interpreter: $($PYTHON --version 2>&1)"

# Require Python 3.9+ (the code uses built-in generic type hints).
$PYTHON - <<'PY'
import sys
if sys.version_info < (3, 9):
    sys.exit("ERROR: Python 3.9+ is required (found %d.%d)." % sys.version_info[:2])
PY

if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating virtual environment in $VENV_DIR"
    $PYTHON -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
python -m pip install --upgrade pip >/dev/null

echo "==> Installing dependencies"
python -m pip install -r requirements.txt

echo "==> Running test suite"
python -m pytest -q

cat <<'EOF'

==> Setup complete.

Activate the environment in your shell with:
    source .venv/bin/activate

Then try:
    python -m guard.cli check "Write a catchy social post for the new biologic." --no-llm
    python -m guard.cli eval data/seed_dataset.csv --no-llm
    python generate_dataset.py --rows 1000

The pipeline runs fully offline using heuristic fallbacks. To use the LLM path,
install Ollama (https://ollama.com), run `ollama pull llama3`, and drop --no-llm.
EOF
