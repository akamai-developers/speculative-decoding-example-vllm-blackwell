#!/usr/bin/env bash
set -euo pipefail

set -a
source .env.example
set +a

echo "🚀 Variables loaded! Launching demo"

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

# ----------------------------------------
# 1. Create Python virtual environment
# ----------------------------------------
if [ ! -d ".venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt


# ----------------------------------------
# 2. Hugging Face authentication
# ----------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "Run:export HF_TOKEN='hf_xxx'"
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
# 4. Start both vLLM servers
# ----------------------------------------
echo "Starting baseline vLLM server..."
nohup ./scripts/run_baseline.sh \
  > "$LOG_DIR/baseline.log" 2>&1 &

echo "Starting speculative vLLM server..."
nohup ./scripts/run_speculative.sh \
  > "$LOG_DIR/speculative.log" 2>&1 &

# Give vLLM time to initialize
echo "Waiting for vLLM servers to initialize..."
sleep 15


# ----------------------------------------
# 5. Start Streamlit dashboard
# ----------------------------------------
echo "Starting Streamlit dashboard..."
nohup ./scripts/run_dashboard.sh \
  > "$LOG_DIR/dashboard.log" 2>&1 &


# ----------------------------------------
# 6. Access running services
# ----------------------------------------
INSTANCE_IP=$(hostname -I | awk '{print $1}')

echo "========================================"
echo "Demo environment started successfully"
echo "========================================"
echo ""
echo "Baseline vLLM:    http://${INSTANCE_IP}:8000"
echo "Speculative vLLM: http://${INSTANCE_IP}:8001"
echo "Dashboard: http://${INSTANCE_IP}:8501"
