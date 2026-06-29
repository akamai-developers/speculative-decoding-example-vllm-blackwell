#!/usr/bin/env bash
set -euo pipefail

source ./scripts/common.sh
activate_venv

#Start both vLLM servers (Staggered to prevent profiling race conditions)
echo "Starting baseline vLLM server..."
nohup ./scripts/helpers/_baseline.sh > "$LOG_DIR/baseline.log" 2>&1 &

echo "⏳ Allowing baseline server to initialize and lock VRAM..."
sleep 30

echo "Starting speculative vLLM server..."
nohup ./scripts/helpers/_speculative.sh > "$LOG_DIR/speculative.log" 2>&1 &

echo "⏳ Waiting for speculative vLLM server to initialize..."
sleep 30

echo "========================================"
echo "vLLM servers ready"
echo "========================================"
echo "Baseline vLLM:    http://${INSTANCE_IP}:8000"
echo "Speculative vLLM: http://${INSTANCE_IP}:8001"