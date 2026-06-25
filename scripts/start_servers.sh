#!/usr/bin/env bash
set -euo pipefail

# Automatically export all variables defined in config.env
set -a
source config.env
set +a

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

# ----------------------------------------
# 1. Environment Setup & Dependencies
# ----------------------------------------
if [ ! -d ".venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing/Updating Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# ----------------------------------------
# 2. Hugging Face authentication
# ----------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "Run: export HF_TOKEN='hf_xxx'"
  exit 1
fi

echo "Authenticating with Hugging Face..."
hf auth login --token "$HF_TOKEN"

# ----------------------------------------
# 3. Model download
# ----------------------------------------
echo "Downloading models..."
./scripts/download_models.sh

# ----------------------------------------
# 4. Start both vLLM servers (Staggered to prevent profiling race conditions)
# ----------------------------------------
echo "Starting baseline vLLM server..."
nohup ./scripts/run_baseline.sh > "$LOG_DIR/baseline.log" 2>&1 &

# Give the baseline server 30 seconds to load weights and allocate its KV cache
echo "⏳ Allowing baseline server to initialize and lock VRAM..."
sleep 30

echo "Starting speculative vLLM server..."
nohup ./scripts/run_speculative.sh > "$LOG_DIR/speculative.log" 2>&1 &

echo "⏳ Waiting for speculative vLLM server to initialize..."
sleep 30

# ----------------------------------------
# 5. Start Streamlit dashboard
# ----------------------------------------
echo "Starting Streamlit dashboard..."
nohup ./scripts/run_dashboard.sh > "$LOG_DIR/dashboard.log" 2>&1 &

# ----------------------------------------
# 6. Start Prometheus + Grafana 
# ----------------------------------------
echo "Starting Prometheus and Grafana containers..."
docker-compose -f app/docker-compose.yaml up -d

# ----------------------------------------
# 6. Access running services
# ----------------------------------------
INSTANCE_IP=$(hostname -I | awk '{print $1}')

echo "========================================"
echo "Demo environment started successfully"
echo "========================================"
echo "Baseline vLLM:    http://${INSTANCE_IP}:8000"
echo "Speculative vLLM: http://${INSTANCE_IP}:8001"
echo "Dashboard:        http://${INSTANCE_IP}:8501"
echo "Grafana:          http://${INSTANCE_IP}:3000"
echo "Prometheus:       http://${INSTANCE_IP}:9090"