#!/usr/bin/env bash
set -euo pipefail

source ./scripts/common.sh
activate_venv

echo "Starting speculative vLLM server..."

nohup ./scripts/helpers/_speculative.sh \
    > "$LOG_DIR/speculative.log" 2>&1 &

echo "Waiting for speculative server..."
sleep 30

echo "========================================"
echo "Speculative server ready"
echo "========================================"
echo "http://${INSTANCE_IP}:8001"