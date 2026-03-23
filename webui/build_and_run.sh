#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm is not installed or not in PATH."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: python is not installed or not in PATH."
  exit 1
fi

echo "[1/3] Installing Python dependencies..."
"$PYTHON_BIN" -m pip install -r backend/requirements.txt

echo "[2/3] Installing frontend dependencies and building..."
pushd frontend >/dev/null
npm install
npm run build
popd >/dev/null

echo "[3/3] Starting web server at http://localhost:8000 ..."
exec "$PYTHON_BIN" backend/app.py
