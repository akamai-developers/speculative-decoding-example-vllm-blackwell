#!/usr/bin/env bash
set -euo pipefail

# Change to project directory
cd /root/vllm-speculative-demo

# Make all project scripts executable
chmod +x scripts/*.sh


# Create and activate Python virtual environment
if [ ! -d ".venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv
else
  echo "Using existing Python virtual environment..."
fi

source .venv/bin/activate


# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt


# Authenticate with Hugging Face
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "Run: export HF_TOKEN='hf_xxx'"
  exit 1
fi

hf auth login --token "$HF_TOKEN"


# Download model weights
echo "Downloading models..."
./scripts/download-models.sh


# Create log directory
mkdir -p logs


# Start vLLM servers in background
echo "Starting baseline vLLM server..."
nohup ./scripts/run-baseline.sh > logs/baseline.log 2>&1 &

echo "Starting speculative vLLM server..."
nohup ./scripts/run-speculative.sh > logs/speculative.log 2>&1 &


# Start Streamlit dashboard in background
echo "Starting dashboard..."
nohup ./scripts/run-dashboard.sh > logs/dashboard.log 2>&1 &


# Print service URLs
INSTANCE_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "Demo started successfully."
echo "Baseline vLLM:    http://${INSTANCE_IP}:8000"
echo "Speculative vLLM: http://${INSTANCE_IP}:8001"
echo "Dashboard:        http://${INSTANCE_IP}:8501"