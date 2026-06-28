#!/usr/bin/env bash
set -euo pipefail

source ./scripts/common.sh
activate_venv

echo "Starting baseline vLLM server..."

nohup ./scripts/helpers/_baseline.sh \
    > "$LOG_DIR/baseline.log" 2>&1 &

echo "Waiting for baseline server..."
sleep 30

echo "========================================"
echo "Baseline server ready"
echo "========================================"
echo "http://${INSTANCE_IP}:8000"