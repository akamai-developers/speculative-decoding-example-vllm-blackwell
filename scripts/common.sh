#!/usr/bin/env bash
set -euo pipefail

set -a
source config.env
set +a

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

INSTANCE_IP=$(hostname -I | awk '{print $1}')

activate_venv() {
  if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
  fi

  echo "Activating virtual environment..."
  source .venv/bin/activate
}