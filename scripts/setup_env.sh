#!/usr/bin/env bash
set -euo pipefail

source ./scripts/common.sh
activate_venv

echo "Installing/Updating Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "Downloading models..."
./scripts/helpers/_download_models.sh

echo "Setup complete."