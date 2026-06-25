#!/usr/bin/env bash
set -euo pipefail

source ./scripts/common.sh
activate_venv

echo "Starting Streamlit dashboard..."
nohup ./scripts/helpers/_streamlit.sh > "$LOG_DIR/dashboard.log" 2>&1 &

echo "Starting Prometheus, Grafana, and DCGM Exporter..."
docker compose -f app/docker-compose.yml up -d

echo "========================================"
echo "Monitoring started"
echo "========================================"
echo "Dashboard:     http://${INSTANCE_IP}:8501"
echo "Grafana:       http://${INSTANCE_IP}:3000"
echo "Prometheus:    http://${INSTANCE_IP}:9090"
echo "DCGM Exporter: http://${INSTANCE_IP}:9400/metrics"